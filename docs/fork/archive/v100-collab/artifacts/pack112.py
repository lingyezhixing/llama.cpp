import numpy as np
M, K = 17408, 5120
raw = np.fromfile(r"<TEMP>\v100\q6k_gate.raw", dtype=np.uint8).reshape(M, K//256, 210)
nb = K//128
packed = np.zeros((nb, M, 112), dtype=np.uint8)
for sb in range(K//256):
    for H in range(2):
        idx = sb*2 + H
        packed[idx, :,  0: 64] = raw[:, sb,  0+64*H :  64+64*H]
        packed[idx, :, 64: 96] = raw[:, sb, 128+32*H : 160+32*H]
        packed[idx, :, 96:104] = raw[:, sb, 192+ 8*H : 200+ 8*H]
        packed[idx, :,104:106] = raw[:, sb, 208:210]
packed.tofile(r"<TEMP>\v100\q6k_gate_packed112.raw")
print("packed112", packed.shape, packed.nbytes)
