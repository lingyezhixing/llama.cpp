# Final fix report: whole-branch review of T32 stage 0+1 (FIX_BASE dfd8a7035)

## Status

DONE. One commit on `t32-stage1`: `c5a91c15e` ("t32 : archive correctness runs, guard mirrored range read",
trailer `Assisted-by: opencode`). Not pushed. All three covering runs PASS, exit 0.

## What changed

### Important 1 - stage-1 equivalence runs archived (external channel)

The three plan Task 2 Step 2 commands were re-run on the final binary (same args) and archived with
command + full output to `D:\LLM\Backend\v100-collab\artifacts\t32-stage1-correctness.txt`
(2B hybrid, 2B `-kvu`, 3B pure attention; 23/23/22 checks PASS, 0 FAIL, exit 0 each).
One reference line was appended to `## Result (stage 1, ...)` of
`D:\LLM\Backend\v100-collab\TASKS\T32-agent-session-reuse.md`.

### Important 2 - mirrored-cache guard on range read

`src/llama-kv-cache.cpp`: `state_read_range` now throws for `other` (mirrored cache) before the
`seq_id < 0` check, exactly parallel to `state_write_range`. Previously `state_read_sinfo` returned
silently and the public setter reported success with nothing restored.

### Folded small fixes

1. `tests/test-t32-range.cpp`: continuation check hardened to `!gen1b.empty() && gen1b == gen2b`.
2. `tests/test-t32-range.cpp`: contract checks appended at the end of `run_correctness`
   (`p1 <= p0`, `p0 < 0`, `flags != 0` -> size 0).
3. `tests/test-t32-range.cpp`: post-allocation append-failure test added after the overlap block
   (truncated blob rejected; clean re-append succeeds; destination still generates like seq 1).
4. `include/llama.h`: `// Returns 0 on failure (unsupported memory, invalid range).` above
   `llama_state_seq_get_data_range_ext`.
5. Spec `t32-tree-storage-design.md`: section 2.2.3 read-side sentence corrected (`find_slot` never
   reuses the destination cell; append rejects any position already present); appendix 100K line now
   marks ~2.0s/~1.8s as full-state bandwidth and adds range-throughput equivalents at chunk 512
   (restore ~2.5s at 1896 MiB/s, save ~4.0s at 1198 MiB/s for 4.68 GiB); section 8 notes iswa /
   hybrid-iswa return 0 for the range API; section 9 risk line suggests a distinct blob magic for
   range payloads before stage 2/3.

### Extra library fix required by the new test: `state_clear_append` head reset

`src/llama-kv-cache.cpp`: a failed append restore now moves `head` back to the freed cells
(min freed index, only if smaller than `head`), same pattern as `llama_kv_cache::seq_rm`.
Rationale and evidence below under "Deviations".

## Deviations from the reviewer's exact test snippet (2 items, both needed to keep the check valid)

1. **Hybrid recurrent restore before the final comparison.**
   The range API is attention-only by design (spec 2.1/2.2.1), and `llama_memory_seq_rm(memory, 2, -1, -1)`
   also clears seq 2's recurrent state. Without restoring it, gen2c != gen1 on 2B hybrid
   (run observed FAIL). The test already restores the recurrent blob into seq 1 the same way, so seq 2
   gets the same precondition before the comparison:

   ```cpp
   if (hybrid) {
       check(llama_state_seq_set_data_ext(ctx, data_part.data(), data_part.size(), 2, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size_part, "restore recurrent state into seq 2 after the failed append");
   }
   ```

   This adds a check; it does not weaken any check.

2. **`state_clear_append` head reset (library fix, outside the findings list).**
   With the reviewer's test and no library change, 3B FAILs "destination still generates like seq 1
   after the failed append": generations match for 6 tokens and diverge at the 7th (pos 70).
   Diagnosis (all steps reproduced):
   - K/V data of seq 2 after the failed + clean append is byte-identical to seq 1's known-good range
     payload: a temporary parser compared the payload data sections and printed
     `data sections equal: 1` (both 2360168 bytes).
   - Cell positions/seq maps are correct at every step (`LLAMA_KV_CACHE_DEBUG=3` dump parsed:
     chunk 0 on cells 0..31, chunk 1 on cells 64..95, tokens on cells 32..39).
   - Cause: the failed append's `apply_ubatch()` advanced `head` to 64; `state_clear_append` freed the
     cells but left `head` there, so the clean re-append allocated cells 64..95 instead of reusing
     32..63. The different physical placement changes FA reduction order; the greedy argmax at
     position 70 then flips on a near tie (deterministic, layout-dependent numerics).
   - Fix: reset `head` to the smallest freed cell in `state_clear_append`, exactly like
     `llama_kv_cache::seq_rm` does ("if we freed up a slot, set head to it"). With it, the failed
     append leaves the cache as if it never happened, the clean re-append reuses 32..63, and
     gen2c == gen1 bit-for-bit on both models.
   This is a behavior change beyond the letter of the findings; flagging it for the controller.
   It only affects the failure cleanup path of append restores.

## Covering runs (final binary, post-fix)

Build (tests already configured ON):

```
<TEMP>\v100\build_test_t32.cmd test-t32-range
-> [1/2] Building ... test-t32-range.cpp.obj; [2/2] Linking ... test-t32-range.exe (no errors)
```

All runs: `$env:CUDA_VISIBLE_DEVICES=1`, `-ngl 99 -fa on`; full outputs archived in
`D:\LLM\Backend\v100-collab\artifacts\t32-stage1-correctness.txt`.

### Run 1: 2B hybrid, non-unified (EXIT=0, 23 PASS / 0 FAIL)

```
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```

```
[t32] hybrid=1 full=20990228 partial=20202092 range=788144
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     PASS
[t32] hybrid: range(0,L) size == full - partial + 8                            PASS
[t32] range(0,L/2) size is between header and full range                       PASS
[t32] read chunk 0                                                             PASS
[t32] read chunk 1                                                             PASS
[t32] read recurrent state                                                     PASS
[t32] restore recurrent state into seq 1                                       PASS
[t32] restore chunk 0 (append=false)                                           PASS
[t32] restore chunk 1 (append=true)                                            PASS
[t32] read full state                                                          PASS
[t32] restore full state into seq 2                                            PASS
[t32] chunked restore generates the same tokens as a full restore              PASS
[t32] append with overlapping positions is rejected                            PASS
[t32] rejected append left the existing state intact                           PASS
[t32] rebuild chunk 0 on seq 2                                                 PASS
[t32] truncated append blob is rejected                                        PASS
[t32] destination is clean after the failed append                             PASS
[t32] restore recurrent state into seq 2 after the failed append               PASS
[t32] destination still generates like seq 1 after the failed append           PASS
[t32] range with p1 <= p0 is rejected                                          PASS
[t32] range with p0 < 0 is rejected                                            PASS
[t32] range with unsupported flags is rejected                                 PASS
```

### Run 2: 2B hybrid, `-kvu` (EXIT=0, 23 PASS / 0 FAIL)

```
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 -kvu --mode correctness
```

```
[t32] hybrid=1 full=20990220 partial=20202092 range=788136
... same 23 checks as run 1, all PASS ...
```

### Run 3: 3B pure attention (EXIT=0, 22 PASS / 0 FAIL)

```
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen2.5-Coder-3B-IQ4_XS.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```

```
[t32] hybrid=0 full=2360960 partial=2360960 range=2360960
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     PASS
[t32] attn: range(0,L) size == full size                                       PASS
[t32] payload reads returned the expected sizes                                PASS
[t32] attn: range(0,L) payload == full payload                                 PASS
[t32] range(0,L/2) size is between header and full range                       PASS
[t32] read chunk 0                                                             PASS
[t32] read chunk 1                                                             PASS
[t32] restore chunk 0 (append=false)                                           PASS
[t32] restore chunk 1 (append=true)                                            PASS
[t32] read full state                                                          PASS
[t32] restore full state into seq 2                                            PASS
[t32] chunked restore generates the same tokens as a full restore              PASS
[t32] append with overlapping positions is rejected                            PASS
[t32] rejected append left the existing state intact                           PASS
[t32] rebuild chunk 0 on seq 2                                                 PASS
[t32] truncated append blob is rejected                                        PASS
[t32] destination is clean after the failed append                             PASS
[t32] destination still generates like seq 1 after the failed append           PASS
[t32] range with p1 <= p0 is rejected                                          PASS
[t32] range with p0 < 0 is rejected                                            PASS
[t32] range with unsupported flags is rejected                                 PASS
```

The E lines in the archived output (`state_read_meta: position 0 is already present in seq 1`,
`state_seq_set_data_range: error loading range state`, `state_seq_get_size_range: ... invalid range /
flags`) are the expected negative-path logs behind the PASS checks.

## Files changed

Repo (commit `c5a91c15e`, 3 files, +43/-2):

```
include/llama.h          |  1 +
src/llama-kv-cache.cpp   | 18 +++++++++++++++++-
tests/test-t32-range.cpp | 26 +++++++++++++++++++++++++-
```

External channel (not in git):

- `D:\LLM\Backend\v100-collab\artifacts\t32-stage1-correctness.txt` (new archive)
- `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md` (2.2.3, appendix, 8, 9)
- `D:\LLM\Backend\v100-collab\TASKS\T32-agent-session-reuse.md` (stage-1 result line)

## Commit

```
c5a91c15ed84a7a934ea7ab9d2162ed03da54618
t32 : archive correctness runs, guard mirrored range read

Assisted-by: opencode
```

Working tree clean after the commit. No push.

## Concerns for the controller

1. The `state_clear_append` head reset is an extra library change not in the findings list (see
   deviation 2). It is small, mirrors `seq_rm`, and only touches the append-failure cleanup path.
   Without it the reviewer's generation check fails on 3B for a layout-dependent numerical reason.
2. The generation comparisons (`gen1b == gen2b`, `gen2c == gen1`) assume greedy determinism and, in
   the 3B case, identical cell placement. The head reset makes the placement deterministic after a
   failed append; the checks stay all-PASS on all three configurations.
3. `state_zero_data`'s comment ("the attention can still read the data of free cells") suggests the
   zeroing is a safety net for kernels that ignore masks; it was not further investigated and no
   change was made there.
