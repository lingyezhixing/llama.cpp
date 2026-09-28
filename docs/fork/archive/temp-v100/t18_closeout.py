import io

def read(p):
    s = io.open(p, encoding='utf-8', newline='').read()
    nl = '\r\n' if '\r\n' in s else '\n'
    return s, nl

def write(p, s):
    io.open(p, 'w', encoding='utf-8', newline='').write(s)

# --- BOARD updates ---
p = r'D:\LLM\Backend\v100-collab\BOARD.md'
s, nl = read(p)
reps = [
    ('> 状态: **T18 恢复 RUNNING** (用户放宽 Q15: +2 天探索; 不突破则集成最好变体; **新门槛 = e2e >2% 即合并**);',
     '> 状态: **T18 已关闭 (用户 Q16: 长文点仅 +0.1%, 1% 级收益不值得动 attention 改动)**;'),
    ('T04 (用户口径) -> **最终统一回顾 (用户 Q7)**',
     '~~T18~~ **CLOSED (用户 Q16)** -> T04 (用户口径) -> **最终统一回顾 (用户 Q7)**'),
    ('| T18 | Volta FA mma 内核效率重写 (长文) | **RUNNING (用户放宽 Q15)** | 已得 1.12x (ncols=32+2 CTA/SM; 短 1.22x/长 1.10x); 继续探索结构性路线 (+2 天, 1 天 checkpoint); **保底 = 集成 1.12x; 新门槛 e2e 128k 点 > +2% 即合并** |',
     '| T18 | Volta FA mma 内核效率重写 (长文) | **CLOSED (用户 Q16)** | 生产实测: 128k 点 **+0.11%** (门槛 +2% 未达) / depth32k +1.50% / pp32768 +1.17%; nsys 逐 launch 证明长 l 下 ncols=32 反慢 0.3-0.6% (harness 长 l 保真缺口); 已回退, patch `artifacts/t18-ncols32-REJECTED.patch` |'),
    ('| 4 | **T18 Volta FA mma 重写 (恢复)** | 保底 1.12x (预估 128k +6.3%); 探索目标更高 | +2 天探索 + 集成验收 | **RUNNING (用户放宽 Q15; e2e >2% 即合并)** |',
     '| 4 | ~~T18 Volta FA mma 重写~~ | 实测 128k +0.11% (harness 长 l 保真缺口) | - | **CLOSED (用户 Q16: 1% 级不值得)** |'),
]
for old, new in reps:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
write(p, s)
print('BOARD updated')

# --- RESULTS append ---
t = """
---

# T18 最终处置 (2026-09-23, 用户 Q16): **撤回归档, 不合并**

- 用户裁决: "1% 就不要了, 还牵扯了注意力改动, 划不来" -> T18 全部改动撤销
- 已执行: `git checkout -- ggml/src/ggml-cuda/fattn.cu ggml/src/ggml-cuda/fattn-mma-f16.cuh`;
  工作区回到 5 文件交付态 (convert.cu/dequantize.cuh/fattn-common.cuh/gated_delta_net.cu/unary.cu);
  部署 DLL 保持 `453E2911` (T12+T16, 门槛通过版, 无需重建)
- 留档: `artifacts/t18-ncols32-REJECTED.patch` (1783 B, ncols=32 配置行 + Volta dispatch 覆盖)
  + harness/变体 exe/ncu 2/nsys 2/逐 launch 数据 (artifacts/ 与 %TEMP%/v100)
- 归档结论: 长文 (l>=100k) FA 对 tile 配置不敏感, 瓶颈是 K/V 流式/预取;
  后续任何 FA harness 必须先在**目标 l** 做生产保真对照
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

q = """
---

## 2026-09-23 | T18 裁决结果: 用户选 A (撤回归档) | implementer

用户: "1% 就不要了, 还牵扯了注意力改动, 划不来" -> 选 A。
- 两处源码改动已 `git checkout` 撤销; 工作区 = 5 文件交付态; 部署 DLL = `453E2911`
- patch 留档 `artifacts/t18-ncols32-REJECTED.patch`; 无需你进一步动作 (归档结论已在 RESULTS/TASKS)
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(q)
print('QUESTIONS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T18-fa-mma-rewrite.md'
s, nl = read(p)
s += nl + nl + """## 用户裁决 (Q16): 撤回归档 (2026-09-23)

- 理由: "1% 就不要了, 还牵扯了注意力改动, 划不来"
- 已撤销 `fattn.cu` (dispatch 覆盖) + `fattn-mma-f16.cuh` (ncols=32 配置行); 部署 DLL 保持 `453E2911`
- 留档 `artifacts/t18-ncols32-REJECTED.patch`; 状态 -> **CLOSED**
"""
write(p, s)
print('TASKS/T18 updated')

# --- ENVIRONMENT append ---
p = r'D:\LLM\Backend\v100-collab\ENVIRONMENT.md'
s, nl = read(p)
s += nl + nl + """## T18 归档后状态 (2026-09-23, implementer)

| 项 | 值 |
|---|---|
| 工作区 | `D:\\LLM\\Backend\\src\\llama.cpp-my` **5 个修改文件** (原 4 + `fattn-common.cuh` T12+T16); T18 两文件已 `git checkout` 撤销 |
| patch | 4 个 (dequant-vec / gdn-vec4 / t07-silu-vec4 / **v100-t12t16-fattn-split**); T18 被拒 patch: `artifacts/t18-ncols32-REJECTED.patch` |
| 部署 DLL | `453E2911...` (T12+T16; 每次构建后核对, 防硬链接意外) |
| 验收 (T12+T16) | pp8192@depth128k 375.5 / depth32k 682.2 / pp32768 791.1 / pp512 ~950 / tg128 ~26.6 / PPL 4.3562 |
| T18 产物 | harness `artifacts/t18_fa_harness.cu` + 变体 exe 5 个 + ncu 日志 2 + nsys rep 2 (`%TEMP%/v100/t18_d128k_{A,B}.nsys-rep`) |
| 遗留经验 | FA 类 harness 必须在**目标 l** 做生产保真对照 (T18 在 l=35k 对过, 长 l 未对 -> 误判 1.10x) |
"""
write(p, s)
print('ENVIRONMENT updated')
