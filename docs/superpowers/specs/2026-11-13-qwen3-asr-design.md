# Qwen3-ASR complete support design

Status: draft for review
Date: 2026-11-13

## 1. Summary

Add complete Qwen3-ASR support to llama.cpp server through two new, independent APIs:

- `POST /v1/asr/transcriptions`, async offline transcription with optional fine-grained timestamps.
- `WS /v1/asr/realtime`, live microphone/meeting transcription with append-only text.

The existing `/v1/audio/transcriptions` and `/v1/realtime` endpoints are not used by this feature and their behavior must not change.

This is a greenfield design. The abandoned Qwen3-ASR branch and its implementation must not be reused. Only the existing llama.cpp, mtmd, and server infrastructure may be reused.

## 2. Motivation

The current upstream Qwen3-ASR support is a basic audio-model integration:

- It can transcribe a single audio clip.
- It leaks the `language X<asr_text>` metadata prefix.
- It has no language forcing, context/hotword placement, repetition cleanup, or long-audio chunking.
- It has no realtime transcription.
- It has no timestamp output and cannot load Qwen3-ForcedAligner.

The target use is self-hosted meeting transcription on one RTX4060 8GB:

- Offline audio can be hours long.
- Realtime meetings can run for hours.
- Text must be emitted incrementally and never retracted.
- Offline jobs must yield to realtime immediately.
- Word/character timestamps are required for offline results.

## 3. Goals

- Offline async job API with correct long-audio chunking, language handling, and usage accounting.
- Optional offline timestamps at the finest granularity (`items` only, no `segments`).
- Realtime WebSocket API with append-only deltas and no final full re-decode.
- Bounded realtime memory and compute for unlimited meeting length.
- Realtime priority over offline work, with offline pause and automatic resume.
- Clear capability reporting and startup validation.
- Tests that cover API behavior, long audio, timestamps, realtime stability, and preemption.

## 4. Non-goals

- No changes to `/v1/audio/transcriptions`, `/v1/realtime`, or their behavior.
- No timestamps in realtime.
- No segment-level output. Clients post-process `items` themselves.
- No speaker diarization.
- No batch upload of multiple audio files in one job.
- No OpenAI-compatible shape for the new ASR endpoints in v1.
- No authentication, multi-tenancy, or user quotas. This is a self-use server.
- No persistence of jobs across server restart.
- No model router support for the aligner in v1. Single-model server only.
- No repetition cleanup in v1. The official `detect_and_fix_repetitions` behavior is explicitly deferred.

## 5. Anti-goals

- Do not silently truncate long audio or long output.
- Do not emit provisional realtime text that may later change.
- Do not do a full final re-decode of a realtime stream.
- Do not drop uncommitted realtime audio.
- Do not expose chunk lengths, context budgets, or holdback token counts to clients.
- Do not accept `timestamps=true` and silently ignore it when no aligner is loaded.
- Do not reuse code from the abandoned Qwen3-ASR branch.

## 6. Constraints and assumptions

- Server context size is fixed at startup by `-c`.
- Qwen3-ASR audio token rate is about 13 tokens per second of audio (100 mel frames -> 13 tokens).
- Qwen3-ASR text output rate is variable. Assume a conservative upper bound of about 7 tokens per second for context budgeting.
- Qwen3-ForcedAligner accepts at most 180 seconds of audio per alignment call.
- Qwen3-ASR official ASR input limit is 1200 seconds per model call. llama.cpp may use smaller chunks based on `-c`.
- mtmd audio encoding is serialized by an internal compute mutex. Multiple requests can be in flight, but audio encoding is one at a time.
- One realtime session at a time.
- One offline job running at a time. Additional offline jobs queue.
- RTX4060 8GB is the primary target. CPU execution must still work.

## 7. Public API

### 7.1 Common conventions

- JSON is UTF-8.
- Timestamps in payloads use milliseconds as integers. Field names use the `_ms` suffix.
- Wall-clock timestamps use RFC3339 UTC strings.
- `language` inputs accept a canonical Qwen language name or an ISO 639 code. The server normalizes them.
- Mixed-language output returns comma-separated canonical names and comma-separated codes.
- HTTP errors use this shape:

```json
{
  "error": {
    "code": "timestamps_not_available",
    "message": "timestamps requested but no aligner model is loaded",
    "param": "timestamps"
  }
}
```

- WebSocket errors use this shape:

```json
{"type":"error","code":"invalid_state","message":"...","fatal":true}
```

### 7.2 Offline transcription

#### 7.2.1 Endpoints

```text
POST   /v1/asr/transcriptions
GET    /v1/asr/transcriptions/{id}
DELETE /v1/asr/transcriptions/{id}
```

#### 7.2.2 Submit request

`POST` uses `multipart/form-data`.

| Field | Required | Type | Description |
|---|---|---|---|
| `file` | yes | file | Audio file. |
| `timestamps` | no | boolean string | Default `false`. |
| `language` | no | string | Language name or ISO code. |
| `context` | no | string | Context, hotwords, or meeting notes. Maximum 1024 tokens. |

The server writes the uploaded audio to a temporary file. It does not buffer the whole file in memory. See section 8.5.

The server decodes enough of the audio to determine duration before returning.

Success returns `202 Accepted`:

```json
{
  "id": "asr_01...",
  "object": "asr.transcription",
  "status": "queued",
  "timing": {
    "created_at": "2026-11-13T16:00:00Z"
  },
  "progress": {
    "audio_processed_ms": 0,
    "audio_total_ms": 180000,
    "percent": 0
  }
}
```

Response headers:

```text
Location: /v1/asr/transcriptions/asr_01...
Retry-After: 2
```

If `timestamps=true` and no aligner is loaded, return `400` with `timestamps_not_available` before creating a job.

If the offline queue is full, return `429` with `queue_full` and `Retry-After`.

If the disk cannot store the upload, return `507` with `storage_full`.

#### 7.2.3 Job states

```text
queued -> running -> succeeded
                  -> failed
                  -> cancelled
       -> paused -> running
```

`paused` is caused by realtime preemption. It resumes automatically. The response includes:

```json
"paused_reason": "realtime_priority"
```

#### 7.2.4 Query response

Queued or paused:

```json
{
  "id": "asr_01...",
  "object": "asr.transcription",
  "status": "queued",
  "timing": {
    "created_at": "2026-11-13T16:00:00Z"
  },
  "progress": {
    "audio_processed_ms": 0,
    "audio_total_ms": 180000,
    "percent": 0
  }
}
```

Paused:

```json
{
  "id": "asr_01...",
  "object": "asr.transcription",
  "status": "paused",
  "paused_reason": "realtime_priority",
  "timing": {
    "created_at": "2026-11-13T16:00:00Z",
    "started_at": "2026-11-13T16:00:03Z",
    "processing_ms": 12000,
    "paused_at": "2026-11-13T16:00:15Z"
  },
  "progress": {
    "audio_processed_ms": 65000,
    "audio_total_ms": 180000,
    "percent": 36.1
  }
}
```

Running:

```json
{
  "id": "asr_01...",
  "object": "asr.transcription",
  "status": "running",
  "timing": {
    "created_at": "2026-11-13T16:00:00Z",
    "started_at": "2026-11-13T16:00:03Z",
    "processing_ms": 12000,
    "eta_ms": 21000,
    "estimated_end_at": "2026-11-13T16:00:36Z"
  },
  "progress": {
    "audio_processed_ms": 65000,
    "audio_total_ms": 180000,
    "percent": 36.1
  },
  "usage": {
    "input_tokens": 1300,
    "output_tokens": 160,
    "total_tokens": 1460,
    "cached_tokens": 0,
    "audio_ms": 65000,
    "chunks": 1
  }
}
```

`eta_ms` and `estimated_end_at` are omitted until the first reliable estimate exists. They are also omitted while paused.

Succeeded:

```json
{
  "id": "asr_01...",
  "object": "asr.transcription",
  "status": "succeeded",
  "timing": {
    "created_at": "2026-11-13T16:00:00Z",
    "started_at": "2026-11-13T16:00:03Z",
    "finished_at": "2026-11-13T16:00:36Z",
    "processing_ms": 33000
  },
  "progress": {
    "audio_processed_ms": 180000,
    "audio_total_ms": 180000,
    "percent": 100
  },
  "usage": {
    "input_tokens": 3655,
    "output_tokens": 441,
    "total_tokens": 4096,
    "cached_tokens": 0,
    "audio_ms": 180000,
    "chunks": 2
  },
  "result": {
    "text": "Hello world.",
    "language": "English",
    "language_code": "en",
    "items": [
      {"text": "Hello", "start_ms": 120, "end_ms": 380},
      {"text": "world", "start_ms": 380, "end_ms": 700}
    ]
  }
}
```

`items` is present only when `timestamps=true`. `items` is sorted by `start_ms`, uses absolute input-audio time, and contains the finest granularity the aligner produces. No `segments` are returned. Punctuation and whitespace may not appear in `items`; `result.text` remains the punctuation-preserving transcript.

With timestamps, usage adds aligner counters:

```json
"usage": {
  "input_tokens": 3655,
  "output_tokens": 441,
  "total_tokens": 4096,
  "cached_tokens": 0,
  "audio_ms": 180000,
  "chunks": 2,
  "aligner": {
    "input_tokens": 700,
    "timestamp_tokens": 64,
    "total_tokens": 764,
    "chunks": 2
  }
}
```

Failed or cancelled jobs return the terminal timing shape and an `error` object. Partial `usage` is included if at least one chunk completed.

#### 7.2.5 Cancel and delete

`DELETE`:

- Queued job: remove from queue, return `200` with `status: "cancelled"`.
- Running job: ask the worker to stop at the next safe point, return `202` with `status: "cancelling"`.
- Paused job: mark cancelled and remove.
- Succeeded or failed job: delete the record, return `204`.

A cancelled job keeps its record with `status: "cancelled"` for 24 hours unless explicitly deleted.

### 7.3 Realtime transcription

#### 7.3.1 Endpoint

```text
WS /v1/asr/realtime
```

One WebSocket connection carries one session. No multiplexing. The server accepts at most one active realtime session. A second connection receives `error` with code `realtime_busy`, then the connection closes.

#### 7.3.2 Client messages

The first text frame must be `session.start`:

```json
{
  "type": "session.start",
  "language": "Chinese",
  "context": "",
  "stability": "balanced",
  "sample_rate": 16000
}
```

Fields:

- `language`: optional. Canonical name or ISO code.
- `context`: optional. Meeting context or hotwords. Maximum 1024 tokens.
- `stability`: optional. One of `low_latency`, `balanced`, `high_accuracy`. Default `balanced`.
- `sample_rate`: optional. Default `16000`. Range 8000 to 48000. The server resamples to the model rate.

After `session.started`, the client sends binary frames containing PCM16 little-endian, mono, at `sample_rate`. Frame size is arbitrary.

The client ends the session with:

```json
{"type":"session.finish"}
```

The client may cancel with:

```json
{"type":"session.cancel"}
```

No application-level keepalive is required. Standard WebSocket ping/pong may be used.

#### 7.3.3 Server messages

Session started:

```json
{
  "type": "session.started",
  "session_id": "asr_01...",
  "sample_rate": 16000,
  "stability": "balanced"
}
```

Append-only text:

```json
{"type":"transcript.delta","text":"Hello"}
```

`transcript.delta` is append-only. The server never revises, retracts, or resends text already sent.

Language change:

```json
{"type":"transcript.language","language":"English","language_code":"en"}
```

Sent only when the detected language changes. It is informational and does not affect the text stream.

Finished:

```json
{
  "type": "session.finished",
  "audio_ms": 60000,
  "processing_ms": 12345,
  "language": "English",
  "language_code": "en",
  "usage": {
    "audio_ms": 60000,
    "decode_steps": 60,
    "input_tokens": 123456,
    "output_tokens": 2345,
    "total_tokens": 125801,
    "committed_output_tokens": 2100
  }
}
```

`session.finish` flushes only the remaining tail. The server sends the final deltas, then `session.finished`, then closes the connection. It does not re-decode the whole stream.

Cancelled:

```json
{
  "type": "session.cancelled",
  "audio_ms": 60000,
  "processing_ms": 12345,
  "usage": {
    "audio_ms": 60000,
    "decode_steps": 60,
    "input_tokens": 123456,
    "output_tokens": 2345,
    "total_tokens": 125801,
    "committed_output_tokens": 2100
  }
}
```

Error:

```json
{"type":"error","code":"invalid_audio_format","message":"expected PCM16 mono","fatal":true}
```

#### 7.3.4 Stability presets

The preset controls how many tokens are held back before being emitted and how long a realtime segment may grow before a context reset.

| Preset | Holdback tokens | Segment max | Typical behavior |
|---|---:|---:|---|
| `low_latency` | 5 | 30 s | Earliest text, matches the official rollback count |
| `balanced` | 8 | 45 s | Default |
| `high_accuracy` | 12 | 60 s | Later text, more confirmation |

The same `holdback_tokens` value is used for both the official prompt rollback and the client commit boundary. This guarantees that committed text is always inside the fixed prompt prefix and cannot be revised by a later decode. Clients cannot set token counts directly.

### 7.4 Capabilities

`GET /props` adds an `asr` object when the loaded model is Qwen3-ASR:

```json
"asr": {
  "offline": true,
  "realtime": true,
  "timestamps": true,
  "audio_formats": ["wav", "mp3", "flac"],
  "languages": ["Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"]
}
```

`audio_formats` reports what this build can decode. v1 supports `wav`, `mp3`, and `flac` through the existing miniaudio decoder. M4A, MP4, AAC, and OGG require a new ffmpeg audio decode path and are out of scope for v1. `timestamps` is `true` only when both aligner model files loaded successfully at startup.

### 7.5 Startup flags

The ASR model continues to use the normal server model flags:

```text
-m, --model <asr-text.gguf>
--mmproj <asr-mmproj.gguf>
```

Add:

```text
--asr-aligner-model <aligner-text.gguf>
--asr-aligner-mmproj <aligner-mmproj.gguf>
--asr-aligner-ctx <tokens>          default 8192
```

Both aligner model flags must be provided together, or neither. If only one is provided, startup fails with a clear error. If either file fails to load, startup fails. `--asr-aligner-ctx` is accepted only when the aligner is loaded; setting it without an aligner is a startup error.

If the loaded model is not Qwen3-ASR, the `/v1/asr/*` routes return `501` with `asr_not_supported`. The rest of the server is unaffected.

## 8. Internal design

### 8.1 Components

New server-side components, isolated from the existing chat/completion paths:

- `asr_api`: HTTP and WebSocket route handlers for `/v1/asr/*`.
- `asr_engine`: borrows the server ASR model and shared `mtmd_context`, owns the dedicated ASR llama context, and owns the aligner model, aligner mtmd context, and aligner llama context.
- `asr_job_manager`: job queue, worker thread, job state, pause/resume, retention.
- `asr_offline_worker`: per-job chunk loop, ASR decode, aligner calls, result merge.
- `asr_realtime_session`: per-connection audio buffer, segment state, stable-text commit, context reset.
- `asr_aligner`: conversion metadata, tokenization, non-autoregressive aligner forward, timestamp post-processing.
- `asr_usage`: token and audio accounting.

The existing `server_context_impl` owns one `asr_engine` instance, created lazily. The existing `/v1/audio/transcriptions` route remains untouched.

### 8.2 Qwen3-ASR detection

A model is treated as Qwen3-ASR when all of these hold:

- Text architecture is `qwen3vl`.
- A mmproj is loaded.
- The mmproj audio projector type is `qwen3a`.
- No vision encoder is present.

If any check fails, `asr_not_supported`.

### 8.3 Dedicated ASR context

Create a dedicated `llama_context` from the loaded ASR model when the first `/v1/asr/*` request arrives:

- Sequence count: 1.
- Context size: server `-c`.
- Batch size: server `-b`.
- Same offload settings as the main model unless a dedicated device override is added later.

The main model weights are shared. The dedicated context isolates ASR KV cache from chat slots. If the main server context is already configured for ASR-only self-use, this still works; the extra KV allocation is acceptable on 8GB at the target context sizes.

The ASR context is protected by `asr_mutex`. Only one of offline worker or realtime session may use it at a time.

The ASR engine borrows the server's existing `mtmd_context` for the ASR mmproj. It must not load a second ASR audio encoder. The aligner loads its own `mtmd_context` because it has a different mmproj and text model. This keeps memory bounded and preserves the existing mtmd compute serialization.

### 8.4 Offline job manager

- One worker thread.
- FIFO queue of jobs.
- Default queue capacity: 8 jobs, including the running job. Configurable internally.
- At most one job in `running`.
- A global `realtime_active` flag.
- Worker checks `realtime_active` and a per-job cancel flag at these points:
  - Before starting a chunk.
  - Between audio encoder windows. Qwen3-ASR encodes audio in 8 second windows.
  - Between prompt prefill batches of at most `n_batch` tokens.
  - Between generated tokens.
  - Before the aligner call for a chunk.
- Prompt prefill must not be issued as one unbounded `llama_decode` call. Split it into `n_batch`-sized calls so preemption is checked between them.
- If preemption is requested during one bounded operation, finish that operation, then pause. The worst-case wait is one audio window or one text batch, not the whole chunk.
- If preemption is requested during autoregressive text generation, stop generating, discard the partial chunk, and pause.
- A chunk result is committed to the job only after ASR text and, when requested, aligner items both succeed.
- On pause:
  - Release `asr_mutex`.
  - Set `status = paused`, `paused_reason = realtime_priority`.
  - Keep completed chunks and committed text.
  - Drop the partial chunk result.
  - Free the ASR context KV cache (or reset the sequence) if needed for memory.
- On resume:
  - Re-run the current chunk from the beginning.
  - Keep all previously committed chunks.
- On realtime end or disconnect:
  - Clear `realtime_active`.
  - Notify the worker.
  - The worker resumes the paused job before any queued job.

Job records are kept in memory for 24 hours after reaching a terminal state. Temporary audio files are deleted as soon as the job reaches a terminal state. Records and temp files do not survive restart. On startup, stale temp files in the ASR temp directory are removed.

### 8.5 Streaming upload

The existing multipart parser buffers uploaded files in memory. This is not acceptable for hours-long offline audio.

For `POST /v1/asr/transcriptions`, add a streaming multipart path using cpp-httplib's `HandlerWithContentReader` API through `server_http_context`:

- Read the request body in chunks.
- Parse multipart boundaries incrementally.
- Write the `file` part directly to a generated temp file.
- Collect small text fields (`timestamps`, `language`, `context`) with size limits.
- Reject malformed multipart or multiple file parts.
- Enforce a configurable maximum field size, not a maximum audio size. Audio is limited only by disk space.
- Do not trust the client filename. Use a generated temp name.

If implementing a streaming multipart parser is too invasive, the fallback is a raw-body upload mode:

```text
POST /v1/asr/transcriptions?timestamps=true&language=Chinese
Content-Type: application/octet-stream
<audio bytes>
```

The primary contract remains multipart. The raw-body mode is an optional compatibility path and must not change the JSON responses.

### 8.6 Offline audio preprocessing

- Supported containers and codecs are exactly what the existing mtmd audio decoder supports: WAV, MP3, and FLAC. M4A, MP4, AAC, and OGG are out of scope for v1 and require a new ffmpeg audio decode path.
- An unsupported format returns `400 unsupported_audio_format`. Do not guess by file extension; detect from content.
- Decode the temp file to mono float32 at the ASR model sample rate (16000 Hz).
- Keep the decoded audio in memory only if it fits the configured offline audio memory budget. Otherwise decode chunk by chunk from the temp file.
- Compute total duration.
- Apply the official low-energy splitter to choose chunk boundaries:

```text
search_expand_sec = 5.0
min_window_ms = 100
min_chunk_sec = 0.5
```

At each target boundary, search a window of plus/minus 5 seconds, find the minimum-energy 100 ms window, and cut at the lowest-energy sample inside it. Concatenating chunks reproduces the original audio exactly.

### 8.7 Offline context budget

Let:

```text
audio_tokens_per_sec = 13
output_tokens_per_sec = 7
prompt_overhead = 64
context_tokens = token count of the normalized context field
safety_factor = 0.85
```

`context` is capped at 1024 tokens. A longer context is rejected with `context_too_long` before the job is created. The actual `context_tokens` is subtracted from every chunk budget.

The context-derived maximum chunk duration is:

```text
usable = n_ctx * safety_factor - prompt_overhead - context_tokens
max_chunk_sec = floor(usable / (audio_tokens_per_sec + output_tokens_per_sec))
```

When `timestamps=true`, the aligner context also limits the chunk duration:

```text
aligner_usable = aligner_ctx * safety_factor - prompt_overhead
aligner_max_chunk_sec = floor(aligner_usable / (audio_tokens_per_sec + 18))
max_chunk_sec = min(max_chunk_sec, aligner_max_chunk_sec)
```

The `18` is a conservative worst case of 6 text items per second, with two `<timestamp>` tokens per item plus one text token.

Then:

- Cap at 1200 seconds (official ASR limit).
- If `timestamps=true`, cap at 180 seconds (official aligner limit).
- If the result is below 1 second, reject the job with `context_too_small`.

The default aligner context is 8192 tokens. At 8192, the aligner cap is about 224 seconds, so the 180 second limit applies. At 4096, the cap is about 112 seconds, so timestamps still work with smaller chunks.

Examples:

| `n_ctx` | no timestamps | timestamps, aligner 4096 | timestamps, aligner 8192 |
|---:|---:|---:|---:|
| 4096 | 170 s | 112 s | 112 s |
| 8192 | 345 s | 112 s | 180 s |
| 16384 | 693 s | 112 s | 180 s |

Aligner context memory is about 0.94 GB at 8192 for the 0.6B model. If the ASR model is 1.7B Q8 and memory is tight, use a quantized aligner or lower `--asr-aligner-ctx` to 4096. The server must fail clearly on OOM, not crash.

Chunk text and language are merged in order. Text chunks are concatenated directly, matching the official implementation. If both sides of a boundary are Latin text and no whitespace exists, insert a single space. Document this as a deliberate deviation to avoid glued words. Repetition cleanup is not applied in v1.

### 8.8 Offline progress and ETA

- `progress.audio_processed_ms` advances after each completed chunk, including aligner time.
- `progress.percent` is `audio_processed_ms / audio_total_ms`, clamped to 0..100.
- `processing_ms` is active processing time. It excludes queue time and pause time.
- ETA uses the throughput of the most recent N completed chunks, not the whole job:

```text
rate = recent_processed_audio_ms / recent_processing_ms
eta_ms = max(0, audio_total_ms - audio_processed_ms) / rate
estimated_end_at = now + eta_ms
```

- Omit `eta_ms` and `estimated_end_at` until at least one chunk has completed and the rate is positive.
- Omit them while paused or queued.

### 8.8.1 Generation limits

Offline ASR chunks:

- `max_new_tokens = min(2048, usable - n_prompt_tokens)`.
- `usable = n_ctx * 0.85`.
- If generation reaches `max_new_tokens`, retry the chunk once with the full remaining context as the cap.
- If the retry also reaches the cap, fail the job with `output_limit_reached`. Do not silently truncate.
- The job result may include text and usage from completed chunks, but the failed chunk is not committed.

Realtime decode steps:

- `max_new_tokens = 32` per decode step, matching the official streaming demo.
- If a step reaches the cap, stop that step and continue with the next step. Do not emit beyond the cap.
- If 5 consecutive steps reach the cap, force a segment reset.
- If the next segment also stalls, send a fatal `decode_stalled` error and close the session.

### 8.9 Realtime algorithm baseline

The official Qwen3-ASR streaming algorithm is the baseline:

- Accumulate audio from the stream start.
- Every `chunk_size_sec` (1 second in the official demo), append new audio.
- Re-run the model on all accumulated audio.
- Build the prompt as `base_prompt + prefix_text`.
- For the first `unfixed_chunk_num` steps, `prefix_text` is empty.
- After that, encode the accumulated raw text, drop the last `unfixed_token_num` tokens, and decode the remainder as `prefix_text`.
- The model returns a full transcript of the accumulated audio, not a delta.

Official streaming has no context management and no final re-decode of a committed transcript. It is intended for a single utterance, not an unlimited meeting.

This design uses the official baseline for each segment but does not keep a separate `unfixed_chunk_num`. The stability preset's `holdback_tokens` is used as both the prompt rollback count and the client commit holdback.

### 8.10 Realtime stable commit

The server extends the official algorithm with an append-only commit pointer.

State:

- `committed_tokens`: global text tokens already sent to the client. Monotonic. Metadata tokens are never included.
- `segment_base_tokens`: suffix of `committed_tokens` carried into the current segment at a reset. Empty at session start.
- `segment_raw_tokens`: the latest decode of the current segment, as `prompt_prefix_tokens + generated text tokens`.
- `prev_segment_raw_tokens`: the previous decode of the current segment.
- `prompt_prefix_tokens`: the tokens used as the text prefix of the last prompt. This is the previous raw text minus `holdback_tokens`. At segment start it is `segment_base_tokens`.
- `emitted_in_segment`: number of tokens after `segment_base_tokens` already emitted.
- `holdback_tokens`: from the stability preset.

Metadata handling:

- A decode may start with `language X<asr_text>`.
- Parse and strip that metadata before token accounting.
- Only text tokens after `<asr_text>` enter `segment_raw_tokens`.
- If no `<asr_text>` has appeared yet and the output still looks like metadata, commit nothing.
- A forced language produces text only, so no metadata is expected.

After each decode:

1. Build `segment_raw_tokens = prompt_prefix_tokens + text_tokens`.
2. Compute the token-level longest common prefix (LCP) of `segment_raw_tokens` and `prev_segment_raw_tokens`.
3. Compute `safe = min(lcp, len(segment_raw_tokens) - holdback_tokens)`.
4. Let `emit_begin = len(segment_base_tokens) + emitted_in_segment`.
5. If `safe > emit_begin`, detokenize `segment_raw_tokens[emit_begin:safe]` and send it as one `transcript.delta`.
6. Set `emitted_in_segment = max(emitted_in_segment, safe - len(segment_base_tokens))`.
7. Never move `emitted_in_segment` backward.
8. Set `prompt_prefix_tokens = segment_raw_tokens[0 : len(segment_raw_tokens) - holdback_tokens]` for the next decode.
9. Set `prev_segment_raw_tokens = segment_raw_tokens`.

The same `holdback_tokens` value is used for the official prompt rollback. This is the critical invariant: every committed token is inside the fixed prompt prefix of the next decode, so the model cannot revise it. There is no separate smaller rollback count.

If detokenizing a span would produce a partial UTF-8 code point, hold the incomplete bytes until the next step. Do not send replacement characters.

At `session.finish`:

- If there is buffered tail audio, run one final decode with the current prefix.
- Commit all remaining tokens up to `len(raw_tokens)`.
- Send the final deltas.
- Send `session.finished`.
- Do not re-decode the full stream.

### 8.11 Realtime context management

The segment model is:

- A realtime session consists of one or more segments.
- A segment starts at an audio anchor and may carry a bounded text prefix from the previous segment.
- The segment grows until it reaches the stability preset's `segment_max` or the context budget.
- A reset is triggered when `segment_audio_ms >= segment_max` or the estimated token count exceeds `n_ctx * 0.85`.

The reset prefers a silence anchor. Silence is an energy-based pause of at least 300 ms below the recent speech energy, found in the last 5 seconds of the segment.

Case A: a silence anchor is found.

1. Append the silence audio to the current segment.
2. Run one final decode that includes the silence.
3. Flush the text for the speech before the silence: commit all text tokens returned by this decode, using `holdback_tokens = 0` for this one step. This is the only place where an unconfirmed holdback is committed. It is safe because the anchor is a silence and no new speech is being cut.
4. Set the new segment start to the end of the silence.
5. Carry the last 64 committed text tokens as `segment_base_tokens`.
6. Do not carry audio overlap. The new segment starts after the silence.
7. Reset the model KV cache for the new segment.
8. Set `prompt_prefix_tokens = segment_base_tokens`, `emitted_in_segment = 0`, and `prev_segment_raw_tokens = segment_base_tokens`.

Case B: no silence anchor is found before the hard limit.

1. Choose the lowest-energy 100 ms window near the end of the segment.
2. Commit stable text using the normal `holdback_tokens` rule.
3. Drop audio before `cut - carry_audio_ms`.
4. Keep `carry_audio_ms` of audio before the cut as overlap.
5. Discard uncommitted text after the cut. Its audio is inside the overlap or after the cut and will be decoded again.
6. Carry the last 64 committed text tokens as `segment_base_tokens`.
7. Reset the model KV cache and start the new segment at `cut - carry_audio_ms`.
8. Set `prompt_prefix_tokens = segment_base_tokens`, `emitted_in_segment = 0`, and `prev_segment_raw_tokens = segment_base_tokens`.
9. On the first decode of the new segment, run boundary deduplication: find the longest suffix of the committed text that is also a prefix of the newly generated continuation, and drop that duplicated prefix before committing.
9. Mark the segment boundary as `forced_cut` in diagnostics. Tests must measure word loss and duplication at forced cuts.

Defaults:

```text
carry_audio_ms = 1000
carry_text_tokens = 64
silence_min_ms = 300
```

This is not a blind sliding window:

- The window does not slide every step.
- A reset happens only at a silence or, when no silence exists, at the lowest-energy point.
- Audio is dropped only after its text is committed.
- Uncommitted audio is kept in the overlap.
- A bounded text prefix preserves cross-segment context.
- Committed text is never re-emitted. The prefix only conditions the model.

The reset must also respect the context budget:

```text
estimated_tokens = audio_ms/1000 * 13 + carry_text_tokens + context_tokens + segment_max_sec * 7
```

If `estimated_tokens > n_ctx * 0.85`, shorten the segment or reduce `carry_text_tokens` before starting the segment. If the server context is too small for even the minimum segment, reject the realtime session with `context_too_small`.

The `context` field is capped at 1024 tokens for realtime as well. Its actual token count is included in the budget.

Language and context:

- The initial `language` hint and `context` are placed in the official prompt position.
- The prompt template is the Qwen3-ASR ChatML template.
- Context is a system message.
- A forced language is appended as `language X<asr_text>` after the assistant generation prompt.
- Without a forced language, the model emits `language X<asr_text>...`; the server parses and strips it.
- `transcript.language` is emitted when the parsed language changes.

### 8.12 Realtime priority and preemption

- `realtime_active` is set when a realtime session starts and cleared when it ends, finishes, cancels, or disconnects.
- On a new realtime session:
  1. Set `realtime_active`.
  2. Signal the offline worker to pause.
  3. Wait for `asr_mutex`.
  4. Acquire `asr_mutex`.
  5. Start the session.
- The offline worker releases `asr_mutex` at a safe point.
- Preemption is checked between 8 second audio encoder windows, between `n_batch`-sized prompt prefill batches, and between generated tokens. The realtime start waits for at most one bounded operation. Target worst-case wait is 3 seconds.
- On realtime end, release `asr_mutex`, clear `realtime_active`, notify the worker, and resume the paused offline job.
- A second realtime connection gets `realtime_busy` and is closed; it does not wait or queue.

### 8.13 ForcedAligner model support

The aligner is architecturally Qwen3VL with a classifier head. Extend the existing Qwen3VL model path instead of adding a new architecture.

Conversion:

- `Qwen3ASRTextModel` detects `thinker_config.classify_num` greater than 0.
- Map `thinker.lm_head.weight` to `cls_out.weight`, not `output.weight`.
- Write new GGUF keys:

```text
qwen3vl.classifier.out_count          = classify_num
qwen3vl.asr.timestamp_token_id          = timestamp_token_id
qwen3vl.asr.timestamp_segment_time_ms   = timestamp_segment_time
```

- The tokenizer and token embeddings remain the full 152064 vocabulary.
- Do not write `output.weight` for an aligner.

Loader:

- Read `qwen3vl.classifier.out_count` into `hparams.n_cls_out`.
- Read the timestamp keys into new `llama_hparams` fields.
- Add `bool is_asr_aligner` and set it when `n_cls_out > 1` and `timestamp_token_id != 0`.
- In `llama_model_qwen3vl::load_arch_tensors`, create `cls_out` with shape `{n_embd, n_cls_out}` when `is_asr_aligner`.
- When `is_asr_aligner` is true, do not create or duplicate `output.weight`.
- When `is_asr_aligner` is true, set `hparams.n_embd_out_impl = n_cls_out`.
- Expose `timestamp_token_id` and `timestamp_segment_time_ms` through a small model query API.

Graph:

- At the final projection, use `cls_out` only when `hparams.is_asr_aligner` is true; otherwise use the normal output head. Do not affect normal Qwen3VL reranker or TTS paths.
- All text layers, norms, and M-RoPE handling stay identical to Qwen3VL.
- Do not expose the classifier as vocabulary-wide logits. `llama_get_logits_ith()` uses a fixed `n_vocab` stride and the logits buffer is sized by vocabulary, so a 5000-wide logits tensor would corrupt output indexing.
- Instead, expose the classifier output as per-token embeddings:
  - Set `hparams.n_embd_out_impl = hparams.n_cls_out` when `n_cls_out > 1`.
  - Set `res->t_embd = cls_out` and `res->t_logits = nullptr`.
  - Create the aligner context with `embeddings = true` and pooling type `NONE`.
  - Read the per-position classifier row with `llama_get_embeddings_ith()`.
- This reuses the existing per-token embedding output path (`llama_context` uses `n_embd_out()` as the stride) and avoids changing the logits buffer layout.

Align input construction:

- Build the token sequence directly, not through the chat template:

```text
<|audio_start|><|audio_pad|>...<|audio_end|>
<timestamp><timestamp>word1<timestamp><timestamp>word2...<timestamp><timestamp>
```

- The audio pad count must equal the number of audio embeddings produced by the aligner mmproj.
- Tokenize with the aligner tokenizer, keeping `<timestamp>` as a single special token.
- Text segmentation:
  - CJK characters: one item per character.
  - Latin runs: split on whitespace and punctuation, keep apostrophes.
  - Japanese: CJK and kana characters as individual items, Latin runs as words. This is an approximation of the official nagisa segmentation.
  - Korean: Hangul syllables as individual items, Latin runs as words. This is an approximation of the official soynlp segmentation.
  - Other space-delimited languages: split on whitespace and punctuation.
- Strip punctuation from item text, matching the official aligner processor.

Align forward:

1. Encode the audio with the aligner mmproj and qwen3a graph.
2. Build a single `llama_batch_ext` containing text tokens as pure token entries and audio embeddings as pure embedding entries at the audio pad positions.
3. Positions are sequential from 0. Audio embedding entries need one position per M-RoPE section.
4. Call `llama_batch_ext_set_output_embd()` on every `<timestamp>` position.
5. Run one `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)`.
6. For each `<timestamp>` position, call `llama_get_embeddings_ith()` and take argmax over the 5000 classifier values.
7. Multiply the class index by `timestamp_segment_time_ms` (80 ms).
8. Apply the official monotonic fix (longest increasing subsequence plus interpolation for anomalies).
9. Pair consecutive timestamp values into start/end for each word/character item.
10. Clamp each item to `[0, chunk_audio_ms]`.
11. Ensure `end_ms >= start_ms`. If not, set `end_ms = start_ms`.
12. Ensure item start and end times are non-decreasing across the chunk. If a later item starts before the previous item, clamp it to the previous start.
13. Add the chunk's audio offset to every item.

The clamping in steps 10 to 12 is a deliberate deviation from the official aligner, which can emit zero-duration or out-of-range spans. The API promises sorted items inside the audio duration.

If the text is empty, return an empty item list without a forward pass.

If the aligner is asked to align more than 180 seconds, return `aligner_audio_too_long`. The offline chunker must prevent this by capping chunks at 180 seconds when timestamps are requested.

Aligner context:

- `--asr-aligner-ctx` sets the aligner context size. Default 8192.
- The aligner context must hold the audio placeholders, text items, two timestamp tokens per item, and the template overhead.
- The timestamp chunk cap is computed in section 8.7 from both the ASR context and the aligner context.
- At 8192, the 180 second limit is safe for the conservative worst case.
- At 4096, the cap falls to about 112 seconds.
- If the configured context is too small for even a 1 second chunk, startup fails with `context_too_small`.

Aligner usage:

- `input_tokens`: number of prompt tokens, including audio placeholders.
- `timestamp_tokens`: number of timestamp positions classified.
- `total_tokens`: sum of the two.
- `chunks`: number of aligner calls.

### 8.14 Usage accounting

Offline:

- `input_tokens`: sum of prompt tokens for all ASR chunks.
- `output_tokens`: sum of generated tokens for all ASR chunks.
- `total_tokens`: input plus output.
- `cached_tokens`: sum of prompt tokens served from KV cache, if the backend reports it; otherwise 0.
- `audio_ms`: audio processed so far.
- `chunks`: completed ASR chunks.
- `aligner`: present only when timestamps were requested and at least one aligner call completed.

Realtime:

- `input_tokens`: sum of prompt tokens over all decode steps. This is cumulative compute, not final text length.
- `output_tokens`: sum of generated tokens over all decode steps, including provisional tokens that were not committed.
- `total_tokens`: input plus output.
- `committed_output_tokens`: tokens actually sent to the client.
- `decode_steps`: number of model decode steps.
- `audio_ms`: audio received.

### 8.15 Error codes

HTTP:

| Code | Status | Meaning |
|---|---:|---|
| `asr_not_supported` | 501 | Loaded model is not Qwen3-ASR. |
| `timestamps_not_available` | 400 | `timestamps=true` but no aligner loaded. |
| `unsupported_language` | 400 | Language is not in the model's supported list. |
| `unsupported_audio_format` | 400 | Audio container or codec is not supported by this build. |
| `invalid_audio` | 400 | File cannot be decoded. |
| `invalid_timestamps` | 400 | `timestamps` is not a boolean. |
| `context_too_long` | 400 | `context` exceeds the 1024 token cap. |
| `queue_full` | 429 | Offline queue is full. |
| `storage_full` | 507 | Temp storage cannot hold the upload. |
| `job_not_found` | 404 | Unknown job id. |
| `job_not_cancellable` | 409 | Job already terminal. |
| `context_too_small` | 400 | `-c` is too small for the requested mode. |
| `output_limit_reached` | 422 | A chunk hit the generation cap twice. |
| `aligner_audio_too_long` | 500 | Internal chunking error: aligner input exceeded 180 s. |
| `internal_error` | 500 | Unexpected failure. |

WebSocket:

| Code | Fatal | Meaning |
|---|---|---|
| `invalid_state` | yes | Audio before `session.start`, or a second start. |
| `invalid_audio_format` | yes | Non-PCM16 audio or unsupported sample rate. |
| `realtime_busy` | yes | Another realtime session is active. |
| `unsupported_language` | yes | Language hint is invalid. |
| `context_too_long` | yes | `context` exceeds the 1024 token cap. |
| `context_too_small` | yes | `-c` cannot hold a realtime segment. |
| `decode_stalled` | yes | Repeated decode steps hit the generation cap. |
| `internal_error` | yes | Unexpected failure. |

HTTP ASR endpoint errors use the `{"error":{...}}` shape. WebSocket errors use the `{"type":"error",...}` shape.

The status column applies when the error is returned synchronously from `POST` or `GET`. When an async job fails during processing, `GET` still returns HTTP `200` with `status: "failed"` and the error code inside the job's `error` object.

### 8.16 Threading and locking

- `asr_mutex`: serializes offline and realtime use of the ASR context.
- `asr_state_mutex`: protects job state, queue, and session registry. Held only for short state updates, never across model compute.
- `realtime_active`: atomic flag.
- Offline worker: one thread.
- Realtime session: runs on the WebSocket connection thread.
- Aligner calls use a separate `aligner_mutex` and aligner context. They happen inside the offline worker while it holds `asr_mutex`, so only one at a time.
- The mtmd context has its own compute mutex. Do not hold `asr_state_mutex` while calling mtmd.
- On shutdown:
  - Stop accepting new jobs.
  - Cancel queued and running jobs.
  - Close realtime sessions.
  - Join the worker thread.
  - Free contexts and models.
  - Delete temp files.

### 8.17 Memory and retention

- ASR model and mmproj are shared with the server model loader.
- Dedicated ASR context KV cache is about 1 GB at `n_ctx=8192` for the 1.7B model.
- Aligner model plus mmproj adds about 1.8 GB when loaded at BF16.
- Aligner KV cache at 8192 adds about 0.94 GB.
- On 8GB, 1.7B Q8 ASR plus 0.6B BF16 aligner with both contexts at 8192 is very tight and may OOM. Use a quantized aligner or `--asr-aligner-ctx 4096` if needed.
- The server must report an OOM error at startup or first use, not crash.
- Finished job records are retained in memory for 24 hours.
- Temporary audio files are deleted at terminal state.
- Completed job results (text and items) are kept in memory. A 10-hour meeting is expected to produce a few MB of JSON.
- No job persistence across restart.

## 9. Testing

### 9.1 Unit tests

- Language normalization and validation.
- `language X<asr_text>` parsing, including `language None`.
- Low-energy chunk boundary selection.
- Context budget formula.
- Stable commit LCP and holdback.
- UTF-8-safe detokenization.
- ETA calculation.
- Usage summation.
- Aligner text segmentation for Chinese and English.
- Aligner monotonic timestamp fix.

### 9.2 Offline integration tests

- 30 s Chinese and English audio: text and language are correct, metadata prefix is stripped.
- 70 s, 280 s, 500 s, and one 2-hour synthetic or looped audio: no truncation, no context overflow.
- `timestamps=true` with aligner loaded: items are sorted, within duration, monotonic, and cover the text tokens.
- `timestamps=true` without aligner: `400 timestamps_not_available`.
- Language hint and context/hotword effect.
- Context longer than 1024 tokens: `400 context_too_long`.
- Generation cap hit: explicit `output_limit_reached`, not silent truncation.
- Unsupported audio format: `400 unsupported_audio_format`.
- Cancel queued, running, and paused jobs.
- Job record retention and deletion.
- Usage fields are consistent with chunk sums.

Aligner integration tests require a local aligner GGUF. They are gated by environment variables and are not part of the default CI run:

```text
QWEN3_ASR_ALIGNER_MODEL
QWEN3_ASR_ALIGNER_MMPROJ
```

The work includes the conversion changes and a local test script that converts the user-provided HF aligner to GGUF. Publishing converted models to Hugging Face is out of scope.

### 9.3 Realtime integration tests

- 0.6B and 1.7B models.
- 1-minute and 10-minute streams: deltas are append-only, no retractions.
- Joined deltas equal the final committed text.
- Stability presets: low_latency emits earlier than high_accuracy.
- Silence-anchored reset: no missing or duplicated words at the boundary.
- Forced low-energy reset: boundary deduplication prevents duplicate words; measure any loss.
- `session.finish` flushes the tail without a full re-decode.
- `session.cancel` stops cleanly.
- Second session gets `realtime_busy`.
- Disconnect without finish releases the session and resumes offline.
- Realtime preempts a running offline job; offline resumes and completes with correct text.
- Usage cumulative counters are present and sane.

### 9.4 Regression tests

- Existing `/v1/audio/transcriptions` behavior is unchanged for other audio models.
- Existing chat/completion endpoints are unchanged.
- Server starts without aligner flags and reports `timestamps=false`.
- Server starts with aligner flags and reports `timestamps=true`.

## 10. Implementation phases

1. API skeleton and capability detection.
   - Routes, error shape, `/props` `asr` object, startup flags.
   - ASR context and engine initialization.
2. Offline async jobs without timestamps.
   - Streaming upload, job manager, chunking, ASR decode, text merge, usage, ETA, cancel, retention.
3. Realtime WebSocket.
   - Protocol, PCM input, official re-decode baseline, stable commit, context reset, priority/preemption.
4. ForcedAligner.
   - Conversion metadata, classifier head loader/graph, aligner runtime, timestamp items, usage.
5. Polish.
   - Language handling, error cases, docs, tests, performance tuning.

## 11. Decisions made by the designer

- New independent API paths `/v1/asr/transcriptions` and `/v1/asr/realtime`.
- Offline is asynchronous with polling `GET`.
- Offline timestamps use `items` only, in milliseconds, finest granularity.
- Realtime uses binary PCM16 frames plus JSON text frames.
- Realtime text is append-only, with a stable commit pointer and holdback.
- The same holdback count is used for prompt rollback and client commit, so committed text cannot diverge.
- Realtime context management uses silence-anchored segment resets first, with low-energy forced cuts plus audio overlap as a fallback. It is not a blind sliding window.
- Realtime has priority and pauses offline jobs; offline resumes automatically.
- One realtime session at a time; one offline job running at a time.
- The aligner is implemented as a Qwen3VL classifier variant, not a new architecture.
- The aligner context defaults to 8192 tokens; the timestamp chunk cap is derived from both ASR and aligner contexts.
- `context` is capped at 1024 tokens and included in every chunk/segment budget.
- The ASR engine reuses the server's existing ASR mtmd context; the aligner has its own.
- Job records and temp files are not persistent across restart.
- No `segments` output; clients post-process `items`.
- Repetition cleanup is out of scope for v1.

## 12. Open tuning parameters

These are implementation defaults to validate by measurement, not API changes:

- Exact `carry_audio_ms` and `carry_text_tokens` for segment reset.
- Exact stability preset holdback counts and segment maxima.
- Exact context budget constants (`audio_tokens_per_sec`, `output_tokens_per_sec`, safety factor).
- ETA smoothing window.
- Realtime decode step interval.
- Aligner tokenization quality for Japanese and Korean. If the per-character approximation is not accurate enough, port the official nagisa/soynlp segmentation in a later change.

### 12.1 Deferred

- Repetition cleanup. The official `detect_and_fix_repetitions` is not implemented in v1, offline or realtime. Loops are bounded by generation caps and segment resets, not rewritten.

## 13. Success criteria

- Offline: a 2-hour file completes on the RTX4060 without context overflow or silent truncation.
- Offline timestamps: items are usable for subtitles and audio indexing.
- Realtime: a 1-hour meeting streams append-only text with no retractions and bounded memory.
- Realtime balanced preset on RTX4060 with 1.7B Q8: median speech-to-delta latency <= 1.5 s, p95 <= 3 s.
- Realtime low_latency preset: p95 <= 2 s.
- Realtime start latency after preemption is under 3 seconds in the default configuration.
- Existing server endpoints and audio models are unaffected.
