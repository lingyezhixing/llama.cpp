import numpy as np
M, K = 17408, 5120
raw = np.fromfile(r"<TEMP>\v100\q6k_gate.raw", dtype=np.uint8).reshape(M, K//256, 210)
nb = K//128                      # 40 half-super-blocks
packed = np.zeros((nb, M, 128), dtype=np.uint8)
for sb in range(K//256):
    for H in range(2):
        idx = sb*2 + H
        packed[idx, :,  0: 64] = raw[:, sb,  0+64*H :  64+64*H]   # ql half
        packed[idx, :, 64: 96] = raw[:, sb, 128+32*H : 160+32*H]   # qh half
        packed[idx, :, 96:104] = raw[:, sb, 192+ 8*H : 200+ 8*H]   # scales (8)
        packed[idx, :,104:106] = raw[:, sb, 208:210]               # d
print("packed", packed.shape, "stride", packed.strides)
packed.tofile(r"<TEMP>\v100\q6k_gate_packed.raw")
print("bytes", packed.nbytes, "  per-row 128B =", 128*M*nb)
# self-check: unpack one group with the kernel formula and compare against a numpy reference dequant
def deq_ref(row, l0):
    # reference: full 256-element dequant of the super-block, take l0 of the half
    pass
