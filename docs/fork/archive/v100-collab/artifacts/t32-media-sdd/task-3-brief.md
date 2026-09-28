### Task 3: server integration

**Files:**
- Modify: `tools/server/server-common.h` / `server-common.cpp` (`pos_last`, media map accessor)
- Modify: `tools/server/server-context.cpp` (spans helper, park/restore/erase/heal wiring)

**Interfaces:**
- `llama_pos server_tokens::pos_last() const;`
- `const std::map<size_t, mtmd::input_chunk_ptr> & server_tokens::media_map() const;`
- static `std::vector<kv_tree_media> tree_media_spans(const server_tokens & tokens);` in server-context.cpp (FNV-1a 64 over `mtmd_input_chunk_get_id`).

**Steps:**

- [ ] **Step 1: server_tokens helpers**

```cpp
llama_pos server_tokens::pos_last() const {
    if (tokens.empty()) {
        return -1;
    }
    if (has_mtmd && tokens.back() == LLAMA_TOKEN_NULL) {
        for (auto it = map_idx_to_media.rbegin(); it != map_idx_to_media.rend(); ++it) {
            const auto & chunk = it->second;
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk.get());
            if (it->first + n_tok == tokens.size()) {
                return pos_next((int64_t) it->first);
            }
        }
    }
    return pos_next() - 1;
}
```

- [ ] **Step 2: spans helper in server-context.cpp**

```cpp
static uint64_t fnv1a64(const char * s) {
    uint64_t h = 1469598103934665603ull;
    for (; s != nullptr && *s != '\0'; ++s) {
        h ^= (uint8_t) *s;
        h *= 1099511628211ull;
    }
    return h;
}

static std::vector<kv_tree_media> tree_media_spans(const server_tokens & tokens) {
    std::vector<kv_tree_media> out;
    for (const auto & e : tokens.media_map()) {
        const mtmd_input_chunk * chunk = e.second.get();
        kv_tree_media m;
        m.idx   = (int64_t) e.first;
        m.id    = fnv1a64(mtmd_input_chunk_get_id(chunk));
        m.n_tok = (int32_t) mtmd_input_chunk_get_n_tokens(chunk);
        m.n_pos = (int32_t) mtmd_input_chunk_get_n_pos(chunk);
        out.push_back(m);
    }
    return out;
}
```

- [ ] **Step 3: wire the four call sites**

- `prompt_park` (`:303`): drop `prompt.tokens.has_mtmd` from the guard; guard `p_max != prompt.tokens.pos_last()`; `tree.park(io_tgt, io_dft_ptr, prompt.tokens.get_tokens(), tree_media_spans(prompt.tokens), cks)`; `in.tok = c.n_tokens`.
- `prompt_restore_tree` (`:347`): drop `tokens.has_mtmd` from the guard; `tree.restore(io_tgt, io_dft_ptr, tokens.get_tokens(), tree_media_spans(tokens))`; on hit:
  ```cpp
  prompt.tokens = tokens.clone();
  prompt.tokens.keep_first(res.C);
  prompt.checkpoints.clear();
  if (n_ckpt_max > 0 && !prompt.tokens.has_media()) { ... existing rebuild ... }
  ```
  Add `bool server_tokens::has_media() const { return !map_idx_to_media.empty(); }` (private map access is fine inside the class).
- `SLOT_ERASE` (`:2866`): drop the `!slot->prompt.tokens.has_mtmd` guard; `tree->drop_seq(slot->prompt.tokens.get_tokens(), tree_media_spans(slot->prompt.tokens))`.
- heal capture (`:3635`): `tree->capture_anchor(io_h_tgt, io_h_dft_ptr, slot.prompt.tokens.get_tokens(), tree_media_spans(slot.prompt.tokens), hp)`.

- [ ] **Step 4: build the server and run the text-only regressions**

Run: delete `build\bin\Release\llama-server-impl.dll`; `build_server.cmd`; expect a clean link.
Run: logic + model harness again (0 failures).
Run a quick text-only server check on the 2B with the tree enabled (no mmproj), one 2-request conversation, expect `parked` / `restored` in the log and no warnings. Use the existing script `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` mode `soak` for a short run if convenient, or a direct server start + two requests.

- [ ] **Step 5: commit**

```
git add tools/server/server-common.h tools/server/server-common.cpp tools/server/server-context.cpp
git commit -m "server : wire media prompts into the kv tree"
```

---

