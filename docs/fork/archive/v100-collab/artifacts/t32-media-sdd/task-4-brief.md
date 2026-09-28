### Task 4: end-to-end media verification on the 0.8B + mmproj

**Files:**
- Create: `D:\LLM\Backend\v100-collab\artifacts\t32-media-e2e.ps1`
- Create evidence: `D:\LLM\Backend\v100-collab\artifacts\t32-media-e2e-<timestamp>.txt`
- Append: `D:\LLM\Backend\v100-collab\RESULTS.md`, `STATUS.md`, `TASKS\T32-agent-session-reuse.md`

**Steps:**

- [ ] **Step 1: generate two small test images**

Use PowerShell + System.Drawing to write `t32-img-a.png` (e.g. 320x240 red/blue pattern) and `t32-img-b.png` (different pattern) under `<TEMP>\v100\`. Base64 them in the script.

- [ ] **Step 2: script**

Start `llama-server` (cuda0, port 18080) with:
`-m <models>\Qwen3.5-0.8B-MTP\Qwen3.5-0.8B-UD-Q4_K_XL.gguf --mmproj ...mmproj-F16.gguf -ngl 99 -fa on --image-min-tokens 1024 -np 1 -c 32768 -b 2048 -ub 512 --temp 0 --seed 42 --ctx-checkpoints 4 --kv-tree --tree-ram 2048 --tree-disk <TEMP>\v100\t32-media-tree --tree-checkpoint-anchor-step 8192 --tree-checkpoint-fork-step 4096 -a media-e2e`
Then:
1. request 1: chat completion with image A + a fixed question (greedy, `max_tokens` 32) -> record output + `timings.prompt_n`.
2. request 2: the same conversation plus one more user turn (image A repeated, same prefix) -> expect `prompt_n` much smaller than the full length and the log to show `kv tree: restored ... tokens`.
3. request 3: same positions but image B -> must NOT restore from A's chain (prompt_n full or a restore that stops before the image), and the log must not report a restore whose C is past the image start.
4. run the same three requests against a second server started WITHOUT `--kv-tree` and compare outputs token by token (greedy). If a near-tie shows up (logit gap < 0.05), record it as `cmp_tie` instead of a mismatch (see `artifacts\t32-stage3-ab.ps1` for the tie-aware comparison).
5. assert the tree log contains `parked` for a media prompt and `restored` for request 2.

- [ ] **Step 3: run, collect, archive**

Run the script with `CUDA_VISIBLE_DEVICES=0`; save the full log to the evidence file; include the tree stats lines (`kv tree stats:`), prompt_n values, and the comparison verdict.

- [ ] **Step 4: update the channel docs**

Append a section to `RESULTS.md`, `STATUS.md` and the T32 task file: what was built, the evidence file, the commands, the limitations, and the ruling list from the SDD ledger.

- [ ] **Step 5: commit** (repo changes only; the channel docs live outside the repo and are not committed)

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tools/server/server-common.h tools/server/server-common.cpp tools/server/server-context.cpp tests/test-t32-tree.cpp
git commit -m "kv tree : media reuse end to end"
```
(only if anything is still uncommitted; the E2E script and evidence are artifacts, not repo files)
