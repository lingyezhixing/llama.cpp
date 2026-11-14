#!/usr/bin/env python3
"""Realtime ASR WebSocket protocol, driven by the fake runner.

Set QWEN3_ASR_MODEL and QWEN3_ASR_MMPROJ. The runner is deterministic, so no
audio fixture is needed.
"""

import json
import struct

import pytest

from asr_utils import MODEL, MMPROJ, run_server

pytestmark = pytest.mark.skipif(
    not MODEL or not MMPROJ,
    reason="QWEN3_ASR_MODEL / QWEN3_ASR_MMPROJ are not set",
)

FAKE_ENV = {"LLAMA_SERVER_ASR_FAKE_RUNNER": "1"}


def pcm16(n_samples: int, value: int = 1000) -> bytes:
    return struct.pack("<" + "h" * n_samples, *([value] * n_samples))


@pytest.fixture(scope="module")
def server():
    with run_server([], FAKE_ENV) as port:
        yield port


def connect(port: int):
    from websockets.sync.client import connect as ws_connect
    return ws_connect(f"ws://127.0.0.1:{port}/v1/asr/realtime", open_timeout=15)


def test_session_is_append_only(server):
    with connect(server) as ws:
        ws.send(json.dumps({"type": "session.start", "language": "English"}))
        started = json.loads(ws.recv())
        assert started["type"] == "session.started"
        assert started["sample_rate"] == 16000

        ws.send(pcm16(1600))
        first = json.loads(ws.recv())
        assert first == {"type": "transcript.delta", "text": "hello"}

        ws.send(pcm16(1600))
        ws.send(json.dumps({"type": "session.finish"}))

        deltas = []
        finished = None
        for _ in range(4):
            message = json.loads(ws.recv())
            if message["type"] == "transcript.delta":
                deltas.append(message["text"])
            elif message["type"] == "session.finished":
                finished = message
                break

        assert finished is not None
        assert "".join(["hello"] + deltas) == "hello world"
        assert finished["audio_ms"] == 200
        assert finished["usage"]["decode_steps"] >= 2
        assert finished["language"] == "English"
        assert finished["language_code"] == "en"


def test_second_session_is_busy(server):
    with connect(server) as first:
        first.send(json.dumps({"type": "session.start"}))
        assert json.loads(first.recv())["type"] == "session.started"

        with connect(server) as second:
            message = json.loads(second.recv())
            assert message["type"] == "error"
            assert message["code"] == "realtime_busy"

        first.send(json.dumps({"type": "session.cancel"}))
        cancelled = json.loads(first.recv())
        assert cancelled["type"] == "session.cancelled"
        # a cancelled session still reports what it did
        assert cancelled["usage"]["audio_ms"] == 0
        assert cancelled["language"] == "English"


def test_invalid_sample_rate(server):
    with connect(server) as ws:
        ws.send(json.dumps({"type": "session.start", "sample_rate": 4000}))
        message = json.loads(ws.recv())
        assert message["type"] == "error"
        assert message["code"] == "invalid_audio_format"


def test_audio_before_start(server):
    with connect(server) as ws:
        ws.send(pcm16(160))
        message = json.loads(ws.recv())
        assert message["type"] == "error"
        assert message["code"] == "invalid_state"
