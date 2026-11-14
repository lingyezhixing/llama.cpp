#!/usr/bin/env python3
"""The audio chat and OpenAI transcription paths still work with Qwen3-ASR.

The ASR endpoints are additive: /v1/chat/completions with audio input and
/v1/audio/transcriptions keep their old behavior, also while an offline ASR job
is running. Set QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ and QWEN3_ASR_AUDIO.
"""

import base64
import json

import pytest

from asr_utils import AUDIO, MODEL, MMPROJ, loop_wav, multipart, request, run_server, submit, wait_job

pytestmark = pytest.mark.skipif(
    not MODEL or not MMPROJ or not AUDIO,
    reason="QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ and QWEN3_ASR_AUDIO are not set",
)


@pytest.fixture(scope="module")
def server():
    with run_server() as port:
        yield port


def wav(seconds: float) -> bytes:
    with open(AUDIO, "rb") as f:
        return loop_wav(f.read(), seconds)


def chat_with_audio(port: int, data: bytes) -> tuple[int, dict]:
    body = {
        "messages": [{
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": base64.b64encode(data).decode()},
            }],
        }],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    return request(port, "POST", "/v1/chat/completions", json.dumps(body).encode(),
                   {"Content-Type": "application/json"})


def test_chat_completions_with_audio(server):
    status, res = chat_with_audio(server, wav(10))
    assert status == 200, res

    content = res["choices"][0]["message"]["content"]
    assert content.strip()
    assert res["usage"]["prompt_tokens"] > 0
    assert res["usage"]["completion_tokens"] > 0


def test_audio_transcriptions_oai(server):
    body, ctype = multipart({"file": ("a.wav", wav(10))})
    status, res = request(server, "POST", "/v1/audio/transcriptions", body, {"Content-Type": ctype})

    assert status == 200, res
    assert res["type"] == "transcript.text.done"
    assert res["text"].strip()
    assert res["usage"]["total_tokens"] > 0


def test_chat_with_audio_during_an_offline_job(server):
    """the job and the chat path share the audio encoder, not the ASR context"""
    status, job = submit(server, wav(30))
    assert status == 202, job

    status, res = chat_with_audio(server, wav(10))
    assert status == 200, res
    assert res["choices"][0]["message"]["content"].strip()

    done = wait_job(server, job["id"])
    assert done["status"] == "succeeded", done
    assert done["result"]["text"].strip()
