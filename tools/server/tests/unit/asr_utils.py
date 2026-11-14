#!/usr/bin/env python3
"""Shared helpers for the gated Qwen3-ASR server tests.

The tests start a real llama-server with a converted ASR model. Set:
  QWEN3_ASR_MODEL, QWEN3_ASR_MMPROJ   converted GGUFs (required)
  QWEN3_ASR_AUDIO                     a 16 kHz mono PCM16 WAV (offline tests)
  QWEN3_ASR_DEVICE                    e.g. CUDA0, offloads the model (optional)
  N_GPU_LAYERS                        -ngl value, default 0
"""

import contextlib
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

MODEL  = os.environ.get("QWEN3_ASR_MODEL")
MMPROJ = os.environ.get("QWEN3_ASR_MMPROJ")
AUDIO  = os.environ.get("QWEN3_ASR_AUDIO")
DEVICE = os.environ.get("QWEN3_ASR_DEVICE")

ALIGNER_MODEL  = os.environ.get("QWEN3_ASR_ALIGNER_MODEL")
ALIGNER_MMPROJ = os.environ.get("QWEN3_ASR_ALIGNER_MMPROJ")


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
        with urllib.request.urlopen(req, timeout=300) as res:
            raw = res.read()
            return res.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return err.code, {}


def wait_ready(port: int, proc: subprocess.Popen, log_path: str) -> None:
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            with open(log_path, "r", encoding="utf-8", errors="replace") as log:
                raise RuntimeError(f"server exited early:\n{log.read()[-4000:]}")
        try:
            if request(port, "GET", "/health")[0] == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server did not become ready, log: {log_path}")


@contextlib.contextmanager
def run_server(extra_args: list[str] | None = None, env_overrides: dict | None = None):
    port = free_port()
    log_file = tempfile.NamedTemporaryFile(prefix="asr-", suffix=".log", delete=False)
    log_file.close()

    env = dict(os.environ)
    if "N_GPU_LAYERS" not in env and "QWEN3_ASR_DEVICE" not in env:
        env["CUDA_VISIBLE_DEVICES"] = ""  # no device requested: stay on the CPU
    env.update(env_overrides or {})

    ngl = env.get("N_GPU_LAYERS", "99" if "QWEN3_ASR_DEVICE" in env else "0")

    cmd = [
        server_binary(),
        "-m", MODEL,
        "--mmproj", MMPROJ,
        "-c", env.get("QWEN3_ASR_CTX", "8192"),
        "-np", "1",
        "-ngl", ngl,
        "--port", str(port),
        "--log-file", log_file.name,
    ]

    if DEVICE:
        cmd += ["--device", DEVICE, "--mmproj-device", DEVICE]

    cmd += extra_args or []

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


def submit(port: int, wav: bytes, **fields) -> tuple[int, dict]:
    body, ctype = multipart({"file": ("audio.wav", wav), **fields})
    return request(port, "POST", "/v1/asr/transcriptions", body, {"Content-Type": ctype})


def wait_job(port: int, job_id: str, timeout: float = 600.0) -> dict:
    deadline = time.time() + timeout
    job = {}
    while time.time() < deadline:
        status, job = request(port, "GET", f"/v1/asr/transcriptions/{job_id}")
        assert status == 200, job
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} stuck in {job.get('status')!r}")


def write_looped_wav(source: bytes, seconds: float, path: str) -> str:
    """writes a looped copy to disk, so a long upload is not held in memory"""
    import io
    import wave

    with wave.open(io.BytesIO(source), "rb") as src:
        assert src.getnchannels() == 1, "the fixture must be mono"
        assert src.getsampwidth() == 2, "the fixture must be PCM16"
        assert src.getframerate() == 16000, "the fixture must be 16 kHz"
        frames = src.readframes(src.getnframes())

    target = int(seconds * 16000) * 2

    with wave.open(path, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(16000)
        written = 0
        while written < target:
            block = frames[:min(len(frames), target - written)]
            dst.writeframes(block)
            written += len(block)

    return path


def submit_file(port: int, path: str, timeout: float = 1800.0, **fields) -> tuple[int, dict]:
    """streams a WAV from disk instead of building the whole body in memory"""
    boundary = "----asrtestboundaryfile"

    head = []
    for name, value in fields.items():
        head.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode())
    head.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode())
    head = b"".join(head)
    tail = f"\r\n--{boundary}--\r\n".encode()

    def body():
        yield head
        with open(path, "rb") as f:
            while True:
                block = f.read(1 << 20)
                if not block:
                    break
                yield block
        yield tail

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/asr/transcriptions",
        data=body(),
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(head) + os.path.getsize(path) + len(tail)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read()
            return res.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return err.code, {}


def loop_wav(source: bytes, seconds: float) -> bytes:
    """repeats a PCM16 mono 16 kHz WAV until it reaches the requested duration"""
    import io
    import wave

    with wave.open(io.BytesIO(source), "rb") as src:
        assert src.getnchannels() == 1, "the fixture must be mono"
        assert src.getsampwidth() == 2, "the fixture must be PCM16"
        assert src.getframerate() == 16000, "the fixture must be 16 kHz"
        frames = src.readframes(src.getnframes())

    target = int(seconds * 16000) * 2
    data = (frames * (target // len(frames) + 1))[:target]

    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(16000)
        dst.writeframes(data)
    return out.getvalue()
