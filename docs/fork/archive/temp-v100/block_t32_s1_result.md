
## S1 结果 (implementer, 2026-09-27): A/B/D 已实现, T32-1 与 T32-2 均实测验证

**实现** (工作区, 未提交; 补丁 `artifacts/t32-s1-worktree.patch` (32.7KB), 基于 `3eae5cdae`)
- A. pin 锚点: 聊天路径新增 `last_msg_is_user` (server-common.cpp -> task.params, 由消息数组判定, tool 消息不算);
  轮次首请求的 prompt_end-4 检查点置 `pinned`; 淘汰循环 (min-step 压缩 / 32 上限) 跳过 pinned; 只保留最新 pin;
  **同位置 supersede 继承 pin** (实测发现的最初版本会把 pin 冲掉, 已修)
- B. L2 选择: `load` 改 max-LCP (主序), 去掉 0.25 f_keep 门槛
- D. 轮换: `alloc` 保护"与来任务 LCP 最长"的条目不被 park 逐出; blob 只存 {最老 + pinned + 最新 2} 检查点 (size 计费同步)
- 诊断: 检查点创建打 pin 判定 (TRC), 加载打 lcp

**实测** (`t32_repro.ps1`: 场景 A = 32K 前置 + 36 步 agent 回合 (assistant/tool 真角色) + 下一轮助手内容替换 (思考剔除等价);
场景 B = 两会话 (各 ~5.5-5.9K) 轮换; 均 `-lv 4` + `LLAMA_SERVER_SLOTS_DEBUG=1`, MTP off)
- **T32-1 (A36)**: 基线 D 锚点 (n_tokens=32350) 被 "too close" 压缩删除 (日志 L1295), 最终恢复退到 31838 (D-516),
  prompt_n=3274 / 5.68s; 修复版锚点存活, 恢复 32350 (**D-4**), prompt_n=2762 / 5.10s (-512 token, -10%)
- **T32-2 (B, cache-ram 1100)**: 基线每次切换 5.4-5.7K token / 6.7-7.3s, 6x `making room ... removing oldest entry (1022-1029 MiB)`
  = park 逐出目标 + 恢复落回会话起点 (305); 修复版除前两次外每次切换 **71 token / 0.6s** (最终 4 token / 0.23s),
  7x `found better prompt lcp = 5.7-5.9K, f_keep = 1.0`, 恢复在 blob 末尾 end-4 -> **~12x TTFT**
- **规模效应**: 未裁剪 blob = state 424MB + ~10 检查点 x 149.6MB ~= 1.9GB (超限 -> skip/thrash); 裁剪后 ~1.02GB (可保存)
- **正确性**: A 22 个 + B 8 个请求在基线与修复版输出 (content + reasoning_content) **逐位一致**
- 检查点结构复核: 无 MTP = 149.6 MiB 常数 (GDN recurrent); MTP on = +4.1KiB/token (draft KV) -> S1.5 瘦身依据

**产物**: `artifacts/t32-s1-worktree.patch`, `t32_repro.ps1`, `t32_log.py`, 日志 `t32_{A36base,A36fix3,B1100base,B2fix}.err`,
jsonl `t32_{A20base2,A20fix2,B2base2,B2fix2}.jsonl` (含逐请求 choice 便于回归对比)

**下一步 (待用户)**: (1) 提交 S1 到 fork (需用户批准); (2) S1.5 检查点瘦身 (drop data_dft, restore 走 seq_rm) + 网格/密度评估;
(3) S2a 冒烟 (seq_cp 4 项); (4) B 的后缀复算: 目前 71 token/切换 = 4 rewind + ~67 增量, 已近最优
