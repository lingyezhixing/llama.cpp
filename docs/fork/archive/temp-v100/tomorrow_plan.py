import io

t = """
## 明日任务清单 (2026-09-23 夜整理, 待用户叫起; 顺序固定)

### 0. 开工检查 (5 min)
- 重建并部署: `cmd /c %TEMP%\\v100\\build_ggml_cuda.cmd` (build 输出缺失, 会重链) ->
  `Copy-Item build\\bin\\Release\\ggml-cuda.dll D:\\LLM\\Backend\\llama.cpp-my\\ggml-cuda.dll -Force`
- **核对部署 SHA** (必须): BASE = `102BF84488F2FD43`; T12+T16 构建应为新 SHA (记下来);
  若构建进程/部署异常, 从 `%TEMP%\\v100\\ggml-cuda-T12-BASE.dll` 恢复交付版
- 验证工作区 = 5 文件 (交付 4 + fattn-common.cuh), `fattn-mma-f16.cuh` 干净

### 1. T12+T16 合并验收 (analyst 门槛; A=BASE, B=T12+T16, 同 session 交替 >=3 轮)
1. **短点先行**: `-p 512,4096,8192 -n 128 -r 3` (>=2 轮交替) + tg128 + `-ub 2048 -p 4096` 抽查
   - 一致性回退 >0.3% -> 加条件 `ntiles_KV >= 64` (KV >= 4096 才用 PB=2); 无回退 -> 保持简单 (不加条件)
2. **长点** (r=2, 交替): depth32k (`-p 4096 -n 0 -d 32768`) >= +1.5%; pp32768 (`-p 32768 -n 0`) >= +0.8%;
   **pp8192@depth128k (`-p 8192 -n 0 -d 131072`) >= +4% = 主判据** (先建同 session 基线)
   - 若长点落 +2~4% 且其余全过 -> 报 analyst 复核 (不自动拒); < +2% -> 拒
3. PPL (`-c 512 --chunks 8 --seed 42`, 期望 4.3568; 门槛 4.3572±0.013) + 200 token 生成检查
4. 通过 -> patch 归档 (T12 去 REJECTED 名, T16 新建) + TASKS/T12+T16 写 Result + BOARD/ENVIRONMENT 更新

### 2. T17: Q8_0 反量化补测 (GPU, 轻 ~1 min)
- 微基准: Q8_0 反量化吞吐 vs Q6_K/Q5_K 参考 707-825 GB/s; nsys: pp512 中 Q8_0 路径占比
- go 条件: 明显偏慢 (<~700 GB/s) 且向量化可省 >=2ms/ubatch (>=+0.4% pp512) -> 实施 (逐位相同) + 验收
  (PPL/pp512-8192/pp32768/depth32k/**pp8192@depth128k 不回退**); 否则记录数字关闭

### 3. T18 Stage 0 (静音可做; 纯 CPU/代码, 不动工作区)
- 抽独立 harness: 真实形状 DKQ=DV=256, ncols=64, l=32768 (取 T10/T11 的 nsys 口径),
  **基线 = 生产配置 (T12+T16, PB=2/grid=384, ≈2027us/launch @depth8k)**, 不是 stock (2375us)
- 先复刻基线 -> 再进 Stage 1 方向① (K/V 寄存器软件流水线); 1.20x 才集成, 1.5 天 checkpoint <1.10x 停
- 方向顺序 (analyst): ①K/V 寄存器流水 -> ④ncols=32 探针 (0.5 天) -> ②消 67584B combine smem (2 CTA/SM) -> ③warp tile 扩展

### 环境提醒
- 交付版 DLL 备份: `%TEMP%\\v100\\ggml-cuda-T12-BASE.dll` (102BF844); 实验中 DLL 9 个在 `%TEMP%\\v100\\`
- nsys 图内 kernel 必须 `--cuda-graph-trace=node`; 长点比较必须同 session 交替 A/B
- 用户夜间静音: GPU 重负载/长时间构建都先要时段
"""
io.open(r'D:\LLM\Backend\v100-collab\ENVIRONMENT.md', 'a', encoding='utf-8', newline='').write(t)
print('ENVIRONMENT appended (明日任务清单)')
