#!/usr/bin/env python3
"""Realtime transcription of a one minute stream, gated on the model and a WAV.

Set QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ and QWEN3_ASR_AUDIO (16 kHz mono PCM16
WAV). QWEN3_ASR_DEVICE (e.g. CUDA0) runs it on the GPU.
"""

import json
import struct

import pytest

from asr_utils import AUDIO, MODEL, MMPROJ, loop_wav, run_server

pytestmark = pytest.mark.skipif(
    not MODEL or not MMPROJ or not AUDIO,
    reason="QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ and QWEN3_ASR_AUDIO are not set",
)

FRAME_SAMPLES = 1600  # 100 ms at 16 kHz


@pytest.fixture(scope="module")
def server():
    with run_server() as port:
        yield port


def frames(wav: bytes, seconds: float):
    pcm = wav[44:]  # canonical 44 byte WAV header, written by loop_wav()
    n_samples = int(seconds * 16000)
    for start in range(0, n_samples, FRAME_SAMPLES):
        chunk = pcm[start * 2:(start + FRAME_SAMPLES) * 2]
        if not chunk:
            break
        yield chunk


def test_one_minute_stream(server):
    from websockets.sync.client import connect

    with open(AUDIO, "rb") as f:
        wav = loop_wav(f.read(), 60)

    deltas = []
    language_events = []
    finished = None

    with connect(f"ws://127.0.0.1:{server}/v1/asr/realtime", open_timeout=15, max_size=None) as ws:
        ws.send(json.dumps({"type": "session.start", "language": "English", "stability": "balanced"}))
        started = json.loads(ws.recv())
        assert started["type"] == "session.started"

        # stream faster than real time; the deltas queue up on the socket
        for frame in frames(wav, 60):
            ws.send(frame)

        ws.send(json.dumps({"type": "session.finish"}))
        while finished is None:
            message = json.loads(ws.recv())
            if message["type"] == "transcript.delta":
                deltas.append(message["text"])
            elif message["type"] == "transcript.language":
                language_events.append(message)
            elif message["type"] == "session.finished":
                finished = message

    assert finished is not None
    joined = "".join(deltas)
    assert joined.strip(), "no text was produced"
    # a minute of speech is far more than the first sentence
    assert len(joined) > 200, joined

    usage = finished["usage"]
    assert usage["decode_steps"] >= 30
    assert usage["committed_output_tokens"] > 0
    assert usage["committed_output_tokens"] <= usage["output_tokens"]
    assert finished["audio_ms"] >= 59000

    for event in language_events:
        assert event["language"]
        assert event["language_code"]

@pytest.mark.slow
@pytest.mark.heavy
def test_ten_minute_stream(server):
    """long stream: resets keep the context bounded and the deltas append-only"""
    from websockets.sync.client import connect

    with open(AUDIO, "rb") as f:
        wav = loop_wav(f.read(), 600)

    deltas = []
    finished = None

    with connect(f"ws://127.0.0.1:{server}/v1/asr/realtime", open_timeout=15, max_size=None) as ws:
        ws.send(json.dumps({"type": "session.start", "stability": "balanced"}))
        assert json.loads(ws.recv())["type"] == "session.started"

        for frame in frames(wav, 600):
            ws.send(frame)

        ws.send(json.dumps({"type": "session.finish"}))
        while finished is None:
            message = json.loads(ws.recv())
            if message["type"] == "transcript.delta":
                deltas.append(message["text"])
            elif message["type"] == "session.finished":
                finished = message
            elif message["type"] == "error":
                raise AssertionError(message)

    joined = "".join(deltas)
    assert joined.strip()
    assert finished["audio_ms"] >= 599000

    # the same sentence repeated by the loop should not be glued twice at a cut
    assert "  " not in joined
    assert len(joined) > 500

    # the whole stream went through several segment resets
    assert finished["usage"]["decode_steps"] >= 500
