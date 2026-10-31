"""Thin LLM wrapper: Anthropic API or a local Ollama server, with JSON retries."""
import time

from .config import Settings, settings
from .utils import extract_json


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None, cfg: Settings = settings):
        self.provider = provider or cfg.llm_provider
        self.model = model or cfg.llm_model
        self.cfg = cfg
        self.calls = 0  # incremented per request; used to report cost per query
        self._client = None

    def complete(self, system: str, user: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
        self.calls += 1
        if self.provider == "anthropic":
            if self._client is None:
                import anthropic

                self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

        if self.provider == "ollama":
            import requests

            start = time.time()


            payload = {
                "model": self.model,
                "keep_alive":"30m",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }


            if "relevance grader" in system:
                payload["messages"][0]["content"] += (
                    "\nIMPORTANT: Your entire response must be an object with a "
                    "'grades' array. Never return a single grade object by itself."
)
                payload["format"] = {
                    "type": "object",
                    "properties": {
                        "grades": {
                            "type": "array",
                            "minItems":1,
                            "maxItems":8,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "relevant": {"type": "boolean"},
                                },
                                "required": ["id", "relevant"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["grades"],
                    "additionalProperties": False,
                }
            elif "single valid JSON object" in system:
                payload["format"] = "json"


            r = requests.post(
                f"{self.cfg.ollama_host}/api/chat",
                json=payload,
                timeout=300,
            )


            r.raise_for_status()

            elapsed = time.time() - start
            result = r.json()["message"]["content"]
            if "relevance grader" in system:
                print(f"[GRADER RAW] {result[:1000]!r}")

            print(
                f"[LLM] call={self.calls} "
                f"tokens_limit={max_tokens} "
                f"time={elapsed:.2f}s"
            )

            return result

        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")


    def complete_json(self, system: str, user: str, max_tokens: int = 1000, retries: int = 2) -> dict:
        last_err: Exception | None = None
        sys_json = system + "\n\nRespond with a single valid JSON object and nothing else."

        for attempt in range(retries + 1):
            raw = self.complete(sys_json, user, max_tokens=max_tokens)
            try:
                return extract_json(raw)
            except ValueError as e:
                last_err = e
                print(
                    f"[JSON DEBUG] attempt={attempt + 1}/{retries + 1} "
                    f"error={e} response={raw[:500]!r}"
                )

        raise ValueError(
            f"LLM did not return valid JSON after {retries + 1} tries: {last_err}"
        )
