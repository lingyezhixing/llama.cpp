# Fork documentation

This folder records the private V100/27B development effort of this fork (2026-09-22 to
2026-09-28) so that the work, the measurements and the mistakes stay searchable after the
working directories were deleted.

Quick reference for what the fork changes: [FORK-NOTES.md](../../FORK-NOTES.md) (repo root).

## Documents

| file | content |
|---|---|
| `history.md` | development timeline, task ledger T01-T32, final commit series, rejected work |
| `benchmarks.md` | measured numbers (V100 ggml-cuda, MTP, ReplaySSM, kv tree, A/B summaries) |
| `pitfalls.md` | failure lessons: production incident RCA, logging/env/deploy traps, do-not-retry list |
| `kv-tree.md` | kv tree design, invariants, options, integration points, limits |
| `replayssm.md` | ReplaySSM record/fold design, the conv-plane restore bug and fix, limits |
| `archive/v100-collab/` | raw channel log: protocol, BOARD/STATUS/RESULTS/ARCHIVE, task cards, artifacts (patches, specs, benchmark data) |
| `archive/temp-v100/` | working scratch records from `TEMP\v100` (block notes, repro scripts, measurement outputs) |
| `archive/mtp-sealed-20260923/` | the sealed MTP investigation material (T20 trajectory consistency, T24 notes) |

Paths inside these documents are relative to this folder for `archive/...`, and relative
to the repo root for source files; the raw log printer files were at
`D:\LLM\Backend\v100-collab\` and `<TEMP>\v100\` before archiving.

## Provenance

- The documents were distilled from the channel log and the task cards; `benchmarks.md`
  cites the exact artifact file for every table. The archive keeps the raw material, so
  any distilled statement can be re-checked.
- Commit hashes of the original 21-commit series are gone after the history rebuild; the
  `history.md` mapping table uses commit subjects instead. `replayssm.md` refers to the
  fix by subject as well.
- Where sources conflicted, `benchmarks.md` keeps both values in a "Conflicting or
  corrected figures" table instead of silently picking one.

## Rebuild and deploy (short form)

The fork-specific additions are listed in `FORK-NOTES.md`. The Windows build used:

```sh
cmake --build build --config Release -j --target llama-server
cmake --build build --config Release -j --target test-t32-tree test-t32-range test-backend-ops
```

with `CMAKE_CUDA_ARCHITECTURES=70-real;89-real` (V100 sm70 + RTX 4060 sm89) and MSVC 2022.

Verification commands and the deployment procedure are in `FORK-NOTES.md`
("Standardized rebase workflow", step 5 and step 6).
