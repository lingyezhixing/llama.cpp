#!/usr/bin/env python3
"""Realtime priority over offline jobs: a running job pauses, then resumes.

Set QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ and QWEN3_ASR_AUDIO (16 kHz mono PCM16
WAV). QWEN3_ASR_DEVICE (e.g. CUDA0) runs it on the GPU.
"""

import json
import time

import pytest

from asr_utils import AUDIO, MODEL, MMPROJ, loop_wav, request, run_server, submit, wait_job

pytestmark = pytest.mark.skipif(
    not MODEL or not MMPROJ or not AUDIO,
    reason="QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ and QWEN3_ASR_AUDIO are not set",
)

FRAME_SAMPLES = 1600  # 100 ms at 16 kHz
STREAM_SECONDS = 10
PAUSE_HOLD_S = 3.0  # keeps the job paused long enough to time it


@pytest.fixture(scope="module")
def server():
    with run_server() as port:
        yield port


def wait_status(port: int, job_id: str, wanted: tuple, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    job = {}
    while time.time() < deadline:
        status, job = request(port, "GET", f"/v1/asr/transcriptions/{job_id}")
        assert status == 200, job
        if job["status"] in wanted:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {wanted}, last had {job.get('status')!r}")


def test_realtime_preempts_offline(server):
    from websockets.sync.client import connect

    with open(AUDIO, "rb") as f:
        wav = loop_wav(f.read(), 280)

    # a run without a realtime session gives the reference text and chunk count
    status, baseline_job = submit(server, wav)
    assert status == 202, baseline_job
    baseline = wait_job(server, baseline_job["id"])
    assert baseline["status"] == "succeeded", baseline

    status, job = submit(server, wav)
    assert status == 202, job
    job_id = job["id"]

    wait_status(server, job_id, ("running",))
    t_running = time.time()

    deltas = []
    finished = None

    with connect(f"ws://127.0.0.1:{server}/v1/asr/realtime", open_timeout=15, max_size=None) as ws:
        ws.send(json.dumps({"type": "session.start", "stability": "balanced"}))
        assert json.loads(ws.recv())["type"] == "session.started"

        t_session = time.time()
        paused = wait_status(server, job_id, ("paused",))
        t_paused = time.time()

        # the job stops at the first safe point after the flag: the rest of the
        # audio encoder window and the prefill of the chunk in flight
        print(f"pause latency {t_paused - t_session:.2f} s")
        assert t_paused - t_session < 8.0, t_paused - t_session

        assert paused["paused_reason"] == "realtime_priority", paused
        assert paused["timing"]["paused_at"]
        assert paused["progress"]["audio_processed_ms"] < paused["progress"]["audio_total_ms"]

        # hold the session so the wall clock of the pause is measurable
        time.sleep(PAUSE_HOLD_S)

        pcm = wav[44:]  # canonical 44 byte WAV header, written by loop_wav()
        for start in range(0, STREAM_SECONDS * 16000, FRAME_SAMPLES):
            ws.send(pcm[start * 2:(start + FRAME_SAMPLES) * 2])

        ws.send(json.dumps({"type": "session.finish"}))
        while finished is None:
            message = json.loads(ws.recv())
            if message["type"] == "transcript.delta":
                deltas.append(message["text"])
            elif message["type"] == "session.finished":
                finished = message
            elif message["type"] == "error":
                raise AssertionError(message)

    assert finished is not None
    assert "".join(deltas).strip(), "the realtime session produced no text"

    # the offline job is resumed as soon as the session releases the context
    resumed = wait_status(server, job_id, ("running",))
    t_resumed = time.time()
    assert "paused_at" not in resumed["timing"]

    done = wait_job(server, job_id, timeout=900)
    t_done = time.time()

    assert done["status"] == "succeeded", done
    assert done["progress"]["audio_processed_ms"] == done["progress"]["audio_total_ms"]
    # every chunk was transcribed once: the interrupted one is re-run, not skipped
    assert done["usage"]["chunks"] == baseline["usage"]["chunks"]

    text = done["result"]["text"].strip()
    reference = baseline["result"]["text"].strip()
    assert text
    assert abs(len(text) - len(reference)) <= max(20, len(reference) // 10), (text, reference)

    # processing_ms covers the two running spans and not the pause between them
    paused_window = t_resumed - t_paused
    assert paused_window > PAUSE_HOLD_S, paused_window

    wall_ms = (t_done - t_running) * 1000.0
    processing_ms = done["timing"]["processing_ms"]
    assert processing_ms > 0
    assert processing_ms <= wall_ms - paused_window * 1000.0 + 3000, (
        processing_ms, wall_ms, paused_window)
