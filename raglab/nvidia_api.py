"""Small, inspectable NVIDIA NIM HTTP client (no orchestration/SDK dependency).

Exact model IDs only: discovery is diagnostic, never automatic substitution.
Retries are bounded, honor Retry-After, and never retry authentication errors.
Only final assistant content is consumed; reasoning is not used as an answer.
"""

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
KIMI_MODEL = "moonshotai/kimi-k3"
DEEPSEEK_MODEL = "deepseek-ai/deepseek-v4-pro-0813"
RIVA_MODEL = "nvidia/riva-translate-4b-instruct-v2"
TRANSLATION_MODELS = (KIMI_MODEL, DEEPSEEK_MODEL, RIVA_MODEL)
ANSWER_MODELS = (KIMI_MODEL, DEEPSEEK_MODEL)
BASE_URL = "https://integrate.api.nvidia.com/v1"

_PACE_LOCK = threading.Lock()
_LAST_REQUEST = {}


class NvidiaAPIError(RuntimeError):
    def __init__(self, message, status_code=0, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

    @property
    def retryable(self):
        return self.status_code in {0, 408, 429, 500, 502, 503, 504}


def safe_error(value):
    """Provider errors can echo requests: redact key-shaped strings defensively."""
    text = str(value)
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"(?:nvapi-|sk-)[A-Za-z0-9_-]{12,}", "[REDACTED]", text)[:600]


def retry_after_seconds(value):
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (ValueError, TypeError):
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


def final_content(content):
    """Drop explicit reasoning, NOT just its delimiters. Empty output fails."""
    if not isinstance(content, str):
        raise NvidiaAPIError("NVIDIA returned no final assistant text", 422)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S | re.I)
    content = re.sub(r"```(?:thinking|reasoning)\b.*?```", "", content,
                     flags=re.S | re.I)
    if "<think>" in content.lower():
        raise NvidiaAPIError("NVIDIA returned an unclosed reasoning block", 422)
    if not content.strip():
        raise NvidiaAPIError("NVIDIA returned empty final assistant text", 422)
    return content.strip()


def chat_payload(model, messages, max_tokens=2048):
    """Use each model's documented parameters, not a universal thinking flag."""
    payload = {"model": model, "messages": messages, "temperature": 0,
               "max_tokens": max_tokens, "stream": False}
    if model == KIMI_MODEL:
        payload.update(reasoning_effort="low", seed=0)
    elif model.startswith("deepseek-ai/") or model.startswith("moonshotai/kimi-k2"):
        payload["chat_template_kwargs"] = {"thinking": False}
    # Riva expects a language-pair system message; it has no thinking flag.
    return payload


def read_event_stream(response, deadline):
    """Collect final content from OpenAI-style SSE; discard reasoning deltas."""
    content, model, usage, finish, complete = [], None, {}, None, False
    for line in response:
        if time.monotonic() > deadline:
            raise TimeoutError("NVIDIA streaming response exceeded its wall-clock budget")
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            complete = True
            break
        data = json.loads(payload)
        if data.get("error"):
            raise NvidiaAPIError("NVIDIA stream error: " + safe_error(data["error"]), 502)
        model = data.get("model") or model
        usage = data.get("usage") or usage
        for choice in data.get("choices", []):
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str):
                content.append(text)
            finish = choice.get("finish_reason") or finish
    if not complete or not finish:
        raise NvidiaAPIError("NVIDIA stream ended without a complete final answer", 502)
    return {"model": model, "usage": usage,
            "choices": [{"message": {"content": "".join(content)}, "finish_reason": finish}]}


class NvidiaClient:
    def __init__(self, *, base_url=BASE_URL, timeout=120, attempts=3,
                 min_interval=1.6, max_retry_delay=30, api_key=None, stream=False):
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or os.environ.get("NVIDIA_API_KEY", "")).strip()
        self.timeout = float(timeout)
        self.attempts = int(attempts)
        self.min_interval = float(min_interval)
        self.max_retry_delay = float(max_retry_delay)
        if self.timeout <= 0 or self.attempts < 1 or self.min_interval < 0:
            raise ValueError("NVIDIA timeout/attempts must be positive; pacing nonnegative")
        self.stream = stream
        self.calls = 0
        self.events = []

    def _pace(self):
        with _PACE_LOCK:
            delay = self.min_interval - (time.monotonic() - _LAST_REQUEST.get(self.base_url, 0))
            if delay > 0:
                time.sleep(delay)
            _LAST_REQUEST[self.base_url] = time.monotonic()

    def request(self, path, payload=None):
        if not self.api_key:
            raise NvidiaAPIError("NVIDIA_API_KEY is not set; configure .env or the Actions secret", 401)
        url = path if path.startswith("https://") else f"{self.base_url}/{path.lstrip('/')}"
        if not url.startswith("https://"):
            raise ValueError("NVIDIA endpoints must use HTTPS")
        started = time.monotonic()
        for attempt in range(self.attempts):
            self._pace()
            req = urllib.request.Request(
                url, data=None if payload is None else json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json",
                         "Accept": "text/event-stream" if (payload or {}).get("stream") else "application/json"},
                method="GET" if payload is None else "POST")
            self.calls += 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if getattr(resp, "status", 200) == 202:
                        raise NvidiaAPIError("NVIDIA returned pending (202), not a completed result", 202)
                    data = (read_event_stream(resp, started + self.timeout * (attempt + 1))
                            if (payload or {}).get("stream") else
                            json.loads(resp.read().decode("utf-8")))
                if not isinstance(data, dict):
                    raise NvidiaAPIError("NVIDIA returned a non-object JSON response", 422)
                seconds = round(time.monotonic() - started, 3)
                self.events.append({"model": (payload or {}).get("model"),
                                    "attempts": attempt + 1, "seconds": seconds,
                                    "usage": data.get("usage") or {}})
                # Per-response timing stays correct even with concurrent calls.
                data["_request_seconds"] = seconds
                return data
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                detail = safe_error(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw)
                error = NvidiaAPIError(f"NVIDIA API HTTP {exc.code}: {detail or exc.reason}",
                                       exc.code, retry_after_seconds(exc.headers.get("Retry-After")))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                error = NvidiaAPIError(f"NVIDIA connection error: {safe_error(exc)}")
            except NvidiaAPIError as exc:
                error = exc
            except json.JSONDecodeError as exc:
                raise NvidiaAPIError("NVIDIA returned invalid JSON", 422) from exc
            if not error.retryable or attempt + 1 == self.attempts:
                raise error
            delay = error.retry_after
            if delay is None:
                base = 15 if error.status_code == 429 else 1
                delay = min(self.max_retry_delay, base * 2 ** attempt + random.uniform(0, 0.25))
            if delay > self.max_retry_delay:
                # Don't retry BEFORE a long Retry-After; defer to a later run.
                raise error
            print(f"[nvidia] retry {attempt + 1}/{self.attempts - 1} in {delay:.1f}s "
                  f"(HTTP {error.status_code or 'network'})", flush=True)
            time.sleep(delay)

    def chat(self, model, messages, *, max_tokens=2048):
        payload = chat_payload(model, messages, max_tokens)
        if self.stream and model in ANSWER_MODELS:
            payload["stream"] = True
        data = self.request("chat/completions", payload)
        served = data.get("model")
        if served and served != model:
            raise NvidiaAPIError(f"Requested {model}, but NVIDIA reported model {served}; refusing substitution", 422)
        try:
            choice = data["choices"][0]
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise NvidiaAPIError(f"Incomplete NVIDIA output: {choice['finish_reason']}", 422)
            content = final_content(choice["message"].get("content"))
        except (KeyError, IndexError, TypeError) as exc:
            raise NvidiaAPIError("Malformed NVIDIA chat response (no assistant choice)", 422) from exc
        return {"text": content, "requested_model": model, "served_model": served,
                "usage": data.get("usage") or {},
                "finish_reason": choice.get("finish_reason"),
                "seconds": data.get("_request_seconds")}

    def models(self):
        data = self.request("models")
        return sorted(m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m)
