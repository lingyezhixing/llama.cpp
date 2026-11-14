#!/usr/bin/env python3
"""Qwen3-ASR capability checks, gated on converted GGUFs.

Set QWEN3_ASR_MODEL and QWEN3_ASR_MMPROJ to run. The aligner checks additionally
need QWEN3_ASR_ALIGNER_MODEL and QWEN3_ASR_ALIGNER_MMPROJ. The server starts with
-ngl from N_GPU_LAYERS (default 0, i.e. CPU only).
"""

import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

MODEL          = os.environ.get("QWEN3_ASR_MODEL")
MMPROJ         = os.environ.get("QWEN3_ASR_MMPROJ")
ALIGNER_MODEL  = os.environ.get("QWEN3_ASR_ALIGNER_MODEL")
ALIGNER_MMPROJ = os.environ.get("QWEN3_ASR_ALIGNER_MMPROJ")

pytestmark = pytest.mark.skipif(
    not MODEL or not MMPROJ,
    reason="QWEN3_ASR_MODEL / QWEN3_ASR_MMPROJ are not set",
)


def server_binary() -> str:
    path = os.environ.get("LLAMA_SERVER_BIN")
    if path:
        return path
    for rel in ("build/bin/Release/llama-server.exe", "build/bin/llama-server"):
        candidate = os.path.join(REPO_ROOT, rel)
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError("llama-server not found, set LLAMA_SERVER_BIN")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port: int, method: str, path: str, body: bytes = b"", headers: dict | None = None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body if body else None,
        method=method,
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            return res.status, json.loads(res.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def wait_ready(port: int, proc: subprocess.Popen, log_path: str) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            with open(log_path, "r", encoding="utf-8", errors="replace") as log:
                raise RuntimeError(f"server exited early:\n{log.read()[-4000:]}")
        try:
            status, _ = request(port, "GET", "/health")
            if status == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server did not become ready, log: {log_path}")


def make_wav() -> bytes:
    """the fake runner only checks that the upload exists"""
    return b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 8 + b"data\x00\x00\x00\x00"


def multipart(fields: dict) -> tuple[bytes, str]:
    boundary = "----asrtestboundary"
    chunks = []
    for name, value in fields.items():
        if isinstance(value, tuple):
            filename, data = value
            chunks.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: audio/wav\r\n\r\n".encode() + data + b"\r\n")
        else:
            chunks.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n".encode())
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


FAKE_ENV = {"LLAMA_SERVER_ASR_FAKE_RUNNER": "1"}


@contextlib.contextmanager
def run_server(extra_args: list[str], env_overrides: dict | None = None):
    port = free_port()
    log_file = tempfile.NamedTemporaryFile(prefix="asr-cap-", suffix=".log", delete=False)
    log_file.close()

    env = dict(os.environ)
    if "N_GPU_LAYERS" not in env:
        env["CUDA_VISIBLE_DEVICES"] = ""  # the capability tests need no GPU
    env.update(env_overrides or {})

    cmd = [
        server_binary(),
        "-m", MODEL,
        "--mmproj", MMPROJ,
        "-c", "2048",
        "-np", "1",
        "-ngl", env.get("N_GPU_LAYERS", "0"),
        "--port", str(port),
        "--log-file", log_file.name,
    ] + extra_args

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(port, proc, log_file.name)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        os.unlink(log_file.name)


def test_props_reports_asr():
    with run_server([]) as port:
        status, props = request(port, "GET", "/props")
        assert status == 200
        assert "asr" in props, props.get("model_path")
        asr = props["asr"]
        assert asr["offline"] is True
        assert asr["realtime"] is True
        assert asr["audio_formats"] == ["wav", "mp3", "flac"]
        assert len(asr["languages"]) == 30
        assert asr["timestamps"] is False  # no aligner flags


def test_routes_are_wired():
    with run_server([]) as port:
        # a request without a file part is rejected before a job is created
        status, body = request(
            port, "POST", "/v1/asr/transcriptions",
            body=b"--x\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nChinese\r\n--x--\r\n",
            headers={"Content-Type": "multipart/form-data; boundary=x"},
        )
        assert status == 400
        assert body["error"]["code"] == "file_required"

        status, body = request(port, "GET", "/v1/asr/transcriptions/asr_missing")
        assert status == 404
        assert body["error"]["code"] == "job_not_found"

        status, body = request(port, "DELETE", "/v1/asr/transcriptions/asr_missing")
        assert status == 404
        assert body["error"]["code"] == "job_not_found"


def test_single_aligner_flag_fails_startup():
    with pytest.raises(RuntimeError):
        with run_server(["--asr-aligner-model", ALIGNER_MODEL or "missing.gguf"]):
            pass


@pytest.mark.skipif(not ALIGNER_MODEL or not ALIGNER_MMPROJ,
                    reason="QWEN3_ASR_ALIGNER_MODEL / QWEN3_ASR_ALIGNER_MMPROJ are not set")
def test_props_reports_timestamps_with_aligner():
    with run_server([
        "--asr-aligner-model", ALIGNER_MODEL,
        "--asr-aligner-mmproj", ALIGNER_MMPROJ,
        "--asr-aligner-ctx", "8192",
    ]) as port:
        status, props = request(port, "GET", "/props")
        assert status == 200
        assert props["asr"]["timestamps"] is True

#
# offline job API, driven by the fake runner
#

def wait_job(port: int, job_id: str, want: tuple[str, ...], timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    job = {}
    while time.time() < deadline:
        status, job = request(port, "GET", f"/v1/asr/transcriptions/{job_id}")
        assert status == 200, job
        if job["status"] in want:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} stuck in {job.get('status')!r}, wanted {want}")


def test_offline_job_succeeds():
    with run_server([], {**FAKE_ENV, "LLAMA_SERVER_ASR_FAKE_STEP_MS": "150"}) as port:
        body, ctype = multipart({
            "file":     ("a.wav", make_wav()),
            "language": "Chinese",
            "context":  "meeting notes",
        })
        status, job = request(port, "POST", "/v1/asr/transcriptions", body, {"Content-Type": ctype})

        assert status == 202, job
        assert job["status"] == "queued"
        assert job["object"] == "asr.transcription"
        assert job["progress"]["audio_total_ms"] == 180000

        job_id = job["id"]
        job = wait_job(port, job_id, ("succeeded", "failed"))

        assert job["status"] == "succeeded", job
        assert job["result"]["text"] == "fake transcription"
        assert job["usage"]["chunks"] == 3
        assert job["progress"]["audio_processed_ms"] == 180000
        assert job["progress"]["percent"] == 100
        assert "started_at" in job["timing"]
        assert "finished_at" in job["timing"]

        # a terminal job is deleted, not cancelled
        status, _ = request(port, "DELETE", f"/v1/asr/transcriptions/{job_id}")
        assert status == 204

        status, body = request(port, "GET", f"/v1/asr/transcriptions/{job_id}")
        assert status == 404
        assert body["error"]["code"] == "job_not_found"


def test_offline_job_validation():
    with run_server([], FAKE_ENV) as port:
        def post(fields: dict):
            body, ctype = multipart(fields)
            return request(port, "POST", "/v1/asr/transcriptions", body, {"Content-Type": ctype})

        status, body = post({"language": "Chinese"})
        assert status == 400
        assert body["error"]["code"] == "file_required"

        status, body = post({"file": ("a.wav", make_wav()), "language": "Klingon"})
        assert status == 400
        assert body["error"]["code"] == "unsupported_language"

        status, body = post({"file": ("a.wav", make_wav()), "timestamps": "true"})
        assert status == 400
        assert body["error"]["code"] == "timestamps_not_available"

        status, body = post({"file": ("a.wav", make_wav()), "context": "x" * 200000})
        assert status == 400
        assert body["error"]["code"] == "context_too_long"


def test_offline_job_cancel_running():
    with run_server([], {**FAKE_ENV, "LLAMA_SERVER_ASR_FAKE_STEP_MS": "1500"}) as port:
        body, ctype = multipart({"file": ("a.wav", make_wav())})
        status, job = request(port, "POST", "/v1/asr/transcriptions", body, {"Content-Type": ctype})
        assert status == 202, job

        job_id = job["id"]
        wait_job(port, job_id, ("running",))

        status, body = request(port, "DELETE", f"/v1/asr/transcriptions/{job_id}")
        assert status in (200, 202), body

        job = wait_job(port, job_id, ("cancelled",))
        assert job["status"] == "cancelled"

        status, _ = request(port, "DELETE", f"/v1/asr/transcriptions/{job_id}")
        assert status == 204
