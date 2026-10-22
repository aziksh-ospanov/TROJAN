from typing import Dict, Any, List, Union
import openai, backoff
import json, sys
import string

# Transient errors worth retrying with exponential backoff.
_RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    ConnectionError,
)

_BACKOFF_KWARGS = dict(
    wait_gen=backoff.expo,
    exception=_RETRYABLE_EXCEPTIONS,
    max_tries=6,
    max_time=120,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(
        f"    ↻ Retry {details['tries']}/{6} after {details['wait']:.1f}s "
        f"({type(details['exception']).__name__})"
    ),
)

# Models that use the Responses API instead of Chat Completions.
RESPONSES_API_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5.4"}


class BaseLLM:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _uses_responses_api(self, model: str) -> bool:
        return (model or "") in RESPONSES_API_MODELS

    # ── Chat Completions API (DeepSeek, Gemini, etc.) ─────────────

    @backoff.on_exception(**_BACKOFF_KWARGS)
    def _call_chat_completions(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    @backoff.on_exception(**_BACKOFF_KWARGS)
    def _call_responses(self, **kwargs):
        return self.client.responses.create(**kwargs)

    @staticmethod
    def _extract_chat_usage(response) -> dict:
        if not response.usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens":     response.usage.prompt_tokens or 0,
            "completion_tokens": response.usage.completion_tokens or 0,
            "total_tokens":      response.usage.total_tokens or 0,
        }

    def _generate_via_chat(self, prompt: str, **kwargs) -> dict:
        """Standard Chat Completions path."""
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 8192,
            "n": 1,
            "stop": None,
            "top_p": 0.95,
            "timeout": 300,
            "seed": 42,
        }
        params.update(kwargs)

        res   = self._call_chat_completions(**params)
        usage = self._extract_chat_usage(res)

        if params.get("n", 1) > 1:
            return {
                "content":        [c.message.content for c in res.choices],
                "usage":          usage,
                "model":          res.model,
                "finish_reasons": [c.finish_reason for c in res.choices],
            }

        return {
            "content":       res.choices[0].message.content,
            "usage":         usage,
            "model":         res.model,
            "finish_reason": res.choices[0].finish_reason,
        }

    # ── Responses API (GPT-5 family) ──────────────────────────────

    def _generate_via_responses(self, prompt: str, **kwargs) -> dict:
        """OpenAI Responses API path for GPT-5 and similar models."""
        model = kwargs.get("model", self.model_name)

        api_params: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "text":  {"verbosity": "high"},
            "reasoning": {"effort": "high"},
        }

        if "temperature" in kwargs:
            api_params["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            api_params["max_output_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            api_params["top_p"] = kwargs["top_p"]

        response = self._call_responses(**api_params)

        content = ""
        for item in (response.output or []):
            for block in (getattr(item, "content", None) or []):
                text = getattr(block, "text", None)
                if text:
                    content += text

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if response.usage:
            inp = getattr(response.usage, "input_tokens", 0) or 0
            out = getattr(response.usage, "output_tokens", 0) or 0
            usage = {
                "prompt_tokens":     inp,
                "completion_tokens": out,
                "total_tokens":      inp + out,
            }

        return {
            "content":       content,
            "usage":         usage,
            "model":         model,
            "finish_reason": "stop",
        }

    # ── Unified entry point ───────────────────────────────────────

    def generate_response(self, prompt: str, **kwargs) -> dict:
        """
        Route to the correct API based on the model name.

        Returns:
            {
                "content":       str,
                "usage":         {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
                "model":         str,
                "finish_reason": str,
            }
        """
        model = kwargs.get("model", self.model_name) or ""

        if self._uses_responses_api(model):
            return self._generate_via_responses(prompt, **kwargs)

        return self._generate_via_chat(prompt, **kwargs)


class TemplatedGenerator(BaseLLM):
    def __init__(self, template: str, api_key: str, base_url: str, model_name: str):
        """
        Args:
            template: The string template with {variable} placeholders.
            api_key: OpenAI/DeepSeek API Key.
            base_url: API Base URL.
            model_name: The model to use.
        """
        super().__init__(api_key, base_url, model_name)
        self.template = template

    def _get_template_vars(self) -> set:
        """Introspects the template to find required keys."""
        formatter = string.Formatter()
        return {
            field_name
            for _, field_name, _, _ in formatter.parse(self.template)
            if field_name is not None
        }

    def _render_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        Validates keys, converts complex types to JSON strings, and fills template.
        """
        required_keys = self._get_template_vars()
        provided_keys = set(inputs.keys())

        missing = required_keys - provided_keys
        if missing:
            raise ValueError(f"Missing required template variables: {missing}")

        clean_data = {}
        for key, value in inputs.items():
            if key in required_keys:
                if isinstance(value, (dict, list)):
                    clean_data[key] = json.dumps(value, indent=2)
                else:
                    clean_data[key] = str(value)

        return self.template.format(**clean_data)

    def generate_from_template(
        self,
        variables: Dict[str, Any],
        **generation_kwargs,
    ) -> dict:
        """
        1. Renders the template using 'variables'.
        2. Calls the parent generate_response with specific generation settings.

        Returns the same rich dict as generate_response (content, usage, model, …).
        """
        final_prompt = self._render_prompt(variables)
        return self.generate_response(final_prompt, **generation_kwargs)
