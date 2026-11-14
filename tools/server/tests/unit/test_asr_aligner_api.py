#!/usr/bin/env python3
"""Offline transcription with timestamps, gated on the forced aligner.

Set QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ, QWEN3_ASR_ALIGNER_MODEL,
QWEN3_ASR_ALIGNER_MMPROJ and QWEN3_ASR_AUDIO (16 kHz mono PCM16 WAV).
"""

import pytest

from asr_utils import (
    ALIGNER_MMPROJ, ALIGNER_MODEL, AUDIO, MODEL, MMPROJ,
    loop_wav, run_server, submit, wait_job,
)

pytestmark = pytest.mark.skipif(
    not MODEL or not MMPROJ or not AUDIO or not ALIGNER_MODEL or not ALIGNER_MMPROJ,
    reason="the ASR model, the aligner and QWEN3_ASR_AUDIO are required",
)

ALIGNER_ARGS = [
    "--asr-aligner-model", ALIGNER_MODEL,
    "--asr-aligner-mmproj", ALIGNER_MMPROJ,
    "--asr-aligner-ctx", "8192",
]


@pytest.fixture(scope="module")
def wav30() -> bytes:
    with open(AUDIO, "rb") as f:
        return loop_wav(f.read(), 30)


@pytest.fixture(scope="module")
def server():
    with run_server(ALIGNER_ARGS) as port:
        yield port


def test_timestamps_produce_items(server, wav30):
    status, job = submit(server, wav30, timestamps="true")
    assert status == 202, job

    done = wait_job(server, job["id"])
    assert done["status"] == "succeeded", done

    items = done["result"]["items"]
    assert items, done
    assert done["result"]["text"].strip()

    total = done["progress"]["audio_total_ms"]
    assert total >= 29000

    previous_start = -1
    for item in items:
        assert item["text"].strip(), item
        assert 0 <= item["start_ms"] <= total, item
        assert item["start_ms"] <= item["end_ms"] <= total, item
        assert item["start_ms"] >= previous_start, item
        previous_start = item["start_ms"]

    aligner = done["usage"]["aligner"]
    assert aligner["input_tokens"] > 0
    assert aligner["timestamp_tokens"] > 0
    assert aligner["chunks"] >= 1


def test_items_are_not_returned_without_timestamps(server, wav30):
    status, job = submit(server, wav30)
    assert status == 202, job

    done = wait_job(server, job["id"])
    assert done["status"] == "succeeded", done
    assert "items" not in done["result"]
    assert "aligner" not in done["usage"]
