### Task 2: D13 启动清理

**Files:**
- Modify: `tools/server/server-kv-tree.cpp:182-187` (ctor)
- Test: `tests/test-t32-tree.cpp` (新 `run_logic_wipe`, 注册到 `run_logic`)

**Interfaces:**
- Consumes: `kv_tree_config.disk_dir`, `check`.
- Produces: ctor 清理语义 + 日志 `[kv-tree] cleared %zu stale files from %s` (Task 4 restart 断言 grep `cleared \d+ stale files`).

- [ ] **Step 1: 写失败测试**

`run_logic_fixes` 之后插入:

```cpp
static void run_logic_wipe() {
    fprintf(stderr, "[t32-tree] logic: startup wipe\n");

    const auto dir = std::filesystem::temp_directory_path() / "t32-tree-wipe";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir / "blocks", ec);
    std::filesystem::create_directories(dir / "anchors", ec);

    auto touch = [](const std::filesystem::path & p) {
        std::FILE * f = fopen(p.string().c_str(), "wb");
        if (f != nullptr) {
            fputc('x', f);
            fclose(f);
        }
    };

    touch(dir / "blocks" / "aa.bin");
    touch(dir / "blocks" / "bb.bin.tmp");
    touch(dir / "anchors" / "cc.bin");

    {
        kv_tree_config cfg;
        cfg.disk_dir = dir.string();

        kv_tree tree(cfg);
        check(true, "wipe: constructor with a stale disk dir");
    }

    check(!std::filesystem::exists(dir / "blocks" / "aa.bin"), "wipe: block file removed");
    check(!std::filesystem::exists(dir / "blocks" / "bb.bin.tmp"), "wipe: tmp file removed");
    check(!std::filesystem::exists(dir / "anchors" / "cc.bin"), "wipe: anchor file removed");

    {
        kv_tree_config cfg;
        kv_tree tree(cfg);
        check(true, "wipe: constructor without a disk dir");
    }

    std::filesystem::remove_all(dir, ec);
}
```

`run_logic()` 里追加 `run_logic_wipe();`.

- [ ] **Step 2: 运行确认失败**

Run: 同上 build + `--mode logic`
Expected: FAIL `wipe: block file removed` 等 (文件仍在).

- [ ] **Step 3: 实现 ctor 清理**

`server-kv-tree.cpp:182-187` 替换为:

```cpp
kv_tree::kv_tree(const kv_tree_config & cfg) : cfg(cfg) {
    if (cfg.chunk <= 0 || cfg.anchor_step < 0) {
        fprintf(stderr, "[kv-tree] invalid config: chunk = %d, anchor_step = %d\n", cfg.chunk, cfg.anchor_step);
        GGML_ABORT("invalid kv tree config");
    }

    if (!cfg.disk_dir.empty()) {
        // the tree is not persistent: drop the leftovers of a previous run
        size_t n_stale = 0;

        std::error_code ec;
        for (const auto & entry : std::filesystem::recursive_directory_iterator(cfg.disk_dir, ec)) {
            if (entry.is_regular_file(ec)) {
                n_stale++;
            }
        }

        std::filesystem::remove_all(cfg.disk_dir + "/blocks", ec);
        std::filesystem::remove_all(cfg.disk_dir + "/anchors", ec);

        fprintf(stderr, "[kv-tree] cleared %zu stale files from %s\n", n_stale, cfg.disk_dir.c_str());
    }
}
```

- [ ] **Step 4: 运行确认通过 (logic + server 冒烟)**

Run (logic): build + `--mode logic` → 0 FAIL (61 checks).

Run (server 冒烟, cuda1):

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$repo='D:\LLM\Backend\src\llama.cpp-my'
Remove-Item -Force "$repo\build\bin\Release\llama-server-impl.dll" -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
$d='<TEMP>\v100\t32-stage4-wipe'; Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$d\blocks","$d\anchors" | Out-Null
Set-Content -Path "$d\blocks\aa.bin" -Value 'x'; Set-Content -Path "$d\anchors\cc.bin" -Value 'x'
$p = Start-Process -FilePath "$repo\build\bin\Release\llama-server.exe" -ArgumentList @('-m','<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf','-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','4096','--port','8937','--host','127.0.0.1','--kv-tree','--tree-disk',$d,'--tree-ram','64') -NoNewWindow -PassThru -RedirectStandardError "$d\srv.txt"
for ($i=0; $i -lt 120; $i++) { Start-Sleep -Milliseconds 1000; try { Invoke-RestMethod 'http://127.0.0.1:8937/health' -TimeoutSec 3 | Out-Null; break } catch {} }
Stop-Process -Id $p.Id -Force; $p.WaitForExit(10000) | Out-Null
$cleared = (Select-String -Path "$d\srv.txt" -Pattern 'cleared \d+ stale files').Count
$left = (Get-ChildItem $d -Recurse -File -Exclude 'srv.txt').Count
Write-Output "cleared_lines=$cleared files_left=$left"
```

Expected: `cleared_lines=1 files_left=0`.

- [ ] **Step 5: 提交**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : clear stale kv tree files on startup" -m "Assisted-by: opencode"
```

---

