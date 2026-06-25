"""A minimal, framework-agnostic inference service (u8). It wraps a trained model
and tokenizer behind a request/response handler, so the same object can sit behind
FastAPI, a queue worker, or a test, without depending on any web framework."""
from __future__ import annotations
from typing import Any
from .generate import generate


class InferenceService:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer

    def complete(self, prompt: str, max_new_tokens: int = 50, **opts: Any) -> str:
        return generate(self.model, self.tokenizer, prompt,
                        max_new_tokens=max_new_tokens, **opts)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Map a JSON-style request to a response dict. This is the function a
        FastAPI route (or any transport) would call."""
        if "prompt" not in request:
            raise ValueError("request must include 'prompt'")
        completion = self.complete(
            request["prompt"],
            max_new_tokens=request.get("max_new_tokens", 50),
            temperature=request.get("temperature", 1.0),
            seed=request.get("seed"),
        )
        return {"prompt": request["prompt"], "completion": completion}
