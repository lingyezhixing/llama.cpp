#!/usr/bin/env python3
"""Offline Qwen3-ASR transcription, gated on a converted model and a WAV fixture.

Set QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ and QWEN3_ASR_AUDIO (16 kHz mono PCM16
WAV). Set QWEN3_ASR_DEVICE (e.g. CUDA0) to run on the GPU; otherwise the server
stays on the CPU.
"""

import os

import pytest

import asr_utils
from asr_utils import AUDIO, MODEL, MMPROJ, loop_wav, run_server, submit, submit_file, wait_job
from asr_utils import write_looped_wav

pytestmark = pytest.mark.skipif(
    not MODEL or not MMPROJ or not AUDIO,
    reason="QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ and QWEN3_ASR_AUDIO are not set",
)


@pytest.fixture(scope="module")
def wav30() -> bytes:
    with open(AUDIO, "rb") as f:
        data = f.read()
    return loop_wav(data, 30)


@pytest.fixture(scope="module")
def server():
    with run_server() as port:
        yield port


def test_transcribe_30s(server, wav30):
    status, job = submit(server, wav30)
    assert status == 202, job

    done = wait_job(server, job["id"])

    assert done["status"] == "succeeded", done
    text = done["result"]["text"]
    assert text.strip(), done
    # the metadata prefix must be stripped
    assert "language " not in text[:16] or "<asr_text>" not in text
    assert done["result"]["language"] in ("English", "", "Chinese")

    usage = done["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["chunks"] >= 1
    assert done["progress"]["audio_processed_ms"] == done["progress"]["audio_total_ms"]


def test_transcribe_70s(server):
    with open(AUDIO, "rb") as f:
        wav = loop_wav(f.read(), 70)

    status, job = submit(server, wav)
    assert status == 202, job

    done = wait_job(server, job["id"])
    assert done["status"] == "succeeded", done
    assert done["result"]["text"].strip()
    assert done["progress"]["audio_total_ms"] >= 69000


def test_transcribe_280s(server):
    with open(AUDIO, "rb") as f:
        wav = loop_wav(f.read(), 280)

    status, job = submit(server, wav)
    assert status == 202, job

    done = wait_job(server, job["id"])

    # either the whole audio is processed, or the job fails loudly
    if done["status"] == "succeeded":
        assert done["progress"]["audio_processed_ms"] == done["progress"]["audio_total_ms"]
    else:
        assert done["error"]["code"] in ("output_limit_reached", "context_too_small")


def test_language_hint_is_accepted(server, wav30):
    status, job = submit(server, wav30, language="English")
    assert status == 202, job

    done = wait_job(server, job["id"])
    assert done["status"] == "succeeded", done
    assert done["result"]["language_code"] in ("en", "")


@pytest.mark.slow
@pytest.mark.heavy
def test_two_hour_offline(server, tmp_path):
    """a two hour upload: more than 100 MiB, many chunks, no payload limit"""
    with open(AUDIO, "rb") as f:
        source = f.read()

    path = write_looped_wav(source, 2 * 60 * 60, str(tmp_path / "two_hours.wav"))
    size = os.path.getsize(path)
    assert size > 100 * 1024 * 1024, size

    # one clip's text estimates how much text the looped audio should produce
    unit_job = submit(server, loop_wav(source, 30))[1]
    unit = wait_job(server, unit_job["id"])
    assert unit["status"] == "succeeded", unit
    unit_words = len(unit["result"]["text"].split())
    assert unit_words > 0, unit

    status, job = submit_file(server, path)
    assert status == 202, job

    done = wait_job(server, job["id"], timeout=3600)

    assert done["status"] == "succeeded", done
    assert "error" not in done
    # every chunk was transcribed: the ranges tile the audio
    assert done["progress"]["audio_processed_ms"] == done["progress"]["audio_total_ms"]
    assert done["usage"]["chunks"] >= 20

    text = done["result"]["text"]
    assert text.strip()

    # 240 loops; a chunk boundary that swallowed text shows up as a deficit
    words = len(text.split())
    assert 0.85 * unit_words * 240 <= words <= 1.3 * unit_words * 240, (words, unit_words)


def test_unknown_language_is_rejected(server, wav30):
    status, body = submit(server, wav30, language="Klingon")
    assert status == 400
    assert body["error"]["code"] == "unsupported_language"
