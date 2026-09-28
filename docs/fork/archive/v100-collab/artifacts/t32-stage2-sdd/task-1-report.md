# Task 1 report - module skeleton + hash chain + match + park (pure logic, fake IO)

Status: DONE
Commit: 15588ab6c "server : add kv tree module skeleton and block matching"

## Context

This run finished an aborted attempt. The first implementer was stopped mid-flight
after writing all five files but before committing. On this run the files were
present but untracked/modified, and no commit existed. I read the brief, compared
every file against it line by line, rebuilt, re-ran the tests, committed, and
self-reviewed the diff.

## What was verified / done

1. Compared `tools/server/server-kv-tree.h` (173 lines), `tools/server/server-kv-tree.cpp`
   (218 lines), and `tests/test-t32-tree.cpp` (162 lines) against the brief's Step 1-3
   code blocks. All three match the brief verbatim: types, signatures, logic,
   comments, and formatting. No deviation or omission was found.
2. Verified both CMake diffs match the brief's Step 4 exactly:
   - `tools/server/CMakeLists.txt`: added `server-kv-tree.cpp` / `server-kv-tree.h`
     in alphabetical order; added
     `target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/vendor)`.
   - `tests/CMakeLists.txt`: added the `llama_build(test-t32-tree.cpp ...)` and
     `target_include_directories(test-t32-tree PRIVATE ...)` lines right after
     `llama_build(test-t32-range.cpp)`.
3. ASCII-only check: 0 non-ASCII bytes in all three new files.
4. No files were added beyond the brief; `src/` untouched.

## Commands run and results

Build:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```
Result: `ninja: no work to do.` EXIT=0 (build tree already up to date with the
current sources).

Logic test:
```powershell
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
```
Result: EXIT=0, 8/8 PASS:
```
[t32-tree] mode logic
[t32-tree] park A (3 chunks)                                                        PASS
[t32-tree] blocks after A                                                           PASS (got 3, want 3)
[t32-tree] park B2 (shares 2 chunks)                                                PASS
[t32-tree] blocks after B2 (dedup)                                                  PASS (got 4, want 4)
[t32-tree] park A again (identical)                                                 PASS
[t32-tree] blocks after A re-park (no duplicate)                                    PASS (got 4, want 4)
[t32-tree] park C (partial tail block)                                              PASS
[t32-tree] blocks after C (tail is a new block)                                     PASS (got 5, want 5)
```
Dump shows the expected 5 blocks (two shared [0,512)/[512,1024), divergent tails
at [1024,1536), and the partial [1024,1124) block) and 3 sequence tips.

Server compile check:
```powershell
& '<TEMP>\v100\build_server.cmd'
```
Result: EXIT=0. Only a warning that UI assets could not be downloaded
(disconnected environment); the build then reported "UI: assets unchanged,
skipping embedding". No stale-artifact failure, so no dll deletion was needed.
Confirmed `build\tools\server\CMakeFiles\server-context.dir\Release\server-kv-tree.cpp.obj`
exists and is newer than the source, so the module is compiled into `server-context`.

Commit:
```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tools/server/CMakeLists.txt tests/test-t32-tree.cpp tests/CMakeLists.txt
git commit -m "server : add kv tree module skeleton and block matching" -m "Assisted-by: opencode"
```
Result: `[t32-stage2 15588ab6c] server : add kv tree module skeleton and block matching`,
5 files changed, 559 insertions(+). Working tree clean after the commit.

## Self-review

- Completeness vs brief: all five files present, all three code blocks verbatim,
  both CMake edits verbatim. No missing element.
- Beyond-brief scan: no extra functions, fields, tests, or files. The unused
  `#include <cstring>` in the harness is in the brief, so it was kept.
- Names: `kv_tree*` types and `park`/`match`/`make_room`/`stats`/`dump` all match
  the brief's Produces list.
- YAGNI: `io_dft` and `checkpoints` are deliberately unused in Task 1 (cast to
  void), as the brief specifies; they are consumed in later tasks.
- ASCII-only: verified by byte scan.
- Comments: minimal, match the brief.

Deviations found and fixed: none.

## Concerns

- `park` increments stats (`park_calls`, `park_ok`, `refcount`, `heat`) but does
  not release/double-count anything yet; this is expected for Task 1 and later
  tasks redesign these paths (per the plan's pair analysis).
- The Git LF->CRLF warning on commit is a normal checkout/autocrlf notice for new
  files; the blobs in the commit contain LF as authored.
