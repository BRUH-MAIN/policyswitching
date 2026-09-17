"""VLM client for the navigation pipeline: one image + one prompt -> text (or JSON).

Talks to any OpenAI-compatible chat endpoint; the default is the local llama.cpp
server started by scripts/vlm_server.sh (Gemma-4-E4B). SARO used LLaVA-34B on a
GPU server -- the interface is the same, only the model differs.

Every call is appended to a JSONL transcript (prompt, raw reply, parsed value,
latency, token counts, image file), so a closed-loop run can be audited after
the fact: which question the robot asked, what it saw, and what it was told.
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass
class VLMReply:
  text: str
  parsed: Any = None
  latency_s: float = 0.0
  prompt_tokens: int = 0
  completion_tokens: int = 0
  error: str | None = None


class VLM(Protocol):
  def ask(self, image: np.ndarray, prompt: str, schema: dict | None = None, max_tokens: int = 256, tag: str = "") -> VLMReply: ...


@dataclass
class OpenAICompatVLM:
  base_url: str = "http://127.0.0.1:8091/v1"
  model: str = "local"
  temperature: float = 0.0
  timeout_s: float = 120.0
  image_format: str = "PNG"
  transcript: Path | None = None
  image_dir: Path | None = None
  """If set, every queried image is saved here and referenced from the transcript."""
  retries: int = 2
  _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
  _count: int = 0

  def ask(self, image: np.ndarray, prompt: str, schema: dict | None = None, max_tokens: int = 256, tag: str = "") -> VLMReply:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format=self.image_format)
    b64 = base64.b64encode(buf.getvalue()).decode()
    body: dict = {
      "model": self.model,
      "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/{self.image_format.lower()};base64,{b64}"}},
        {"type": "text", "text": prompt},
      ]}],
      "max_tokens": max_tokens,
      "temperature": self.temperature,
      # Gemma 4 otherwise spends max_tokens in reasoning_content and returns "".
      "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
      body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}

    reply = VLMReply(text="")
    t0 = time.perf_counter()
    for attempt in range(self.retries + 1):
      try:
        req = urllib.request.Request(
          f"{self.base_url}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
          r = json.load(resp)
        msg = r["choices"][0]["message"]
        usage = r.get("usage", {})
        reply = VLMReply(
          text=(msg.get("content") or "").strip(),
          prompt_tokens=int(usage.get("prompt_tokens", 0)),
          completion_tokens=int(usage.get("completion_tokens", 0)),
        )
        if schema is not None:
          reply.parsed = parse_json(reply.text)
          if reply.parsed is None:
            reply.error = "unparseable JSON"
            continue
        break
      except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        reply = VLMReply(text="", error=f"{type(e).__name__}: {e}")
        if attempt < self.retries:
          time.sleep(0.5)
    reply.latency_s = time.perf_counter() - t0
    self._log(image, prompt, schema, reply, tag)
    return reply

  def _log(self, image: np.ndarray, prompt: str, schema: dict | None, reply: VLMReply, tag: str) -> None:
    if self.transcript is None:
      return
    with self._lock:
      idx = self._count
      self._count += 1
      image_file = None
      if self.image_dir is not None:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        image_file = f"q{idx:05d}.jpg"
        Image.fromarray(image).save(self.image_dir / image_file, quality=90)
      self.transcript.parent.mkdir(parents=True, exist_ok=True)
      with open(self.transcript, "a") as f:
        f.write(json.dumps({"idx": idx, "tag": tag, "image": image_file, "prompt": prompt,
                            "schema": schema is not None, **asdict(reply)}) + "\n")


def parse_json(text: str) -> Any:
  """Parse a JSON answer, tolerating a ```json fence or leading prose."""
  s = text.strip()
  fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
  if fence:
    s = fence.group(1).strip()
  for candidate in (s, _first_json_span(s)):
    if candidate:
      try:
        return json.loads(candidate)
      except json.JSONDecodeError:
        pass
  return None


def _first_json_span(s: str) -> str | None:
  m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
  return m.group(1) if m else None


def health(base_url: str = "http://127.0.0.1:8091") -> bool:
  try:
    with urllib.request.urlopen(f"{base_url}/health", timeout=3) as r:
      return r.status == 200
  except (urllib.error.URLError, TimeoutError):
    return False
