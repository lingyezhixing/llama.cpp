import numpy as np, gguf, sys
path = r"<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
r = gguf.GGUFReader(path)
print("tensors:", len(r.tensors))
want = None
for tt in r.tensors:
    n = tt.name
    if "ffn_gate.weight" in n:
        print(f"{n}: type={tt.tensor_type.name} shape={tt.shape}")
        if tt.tensor_type.name == "Q6_K" and want is None:
            want = tt
if want is None:
    print("no Q6_K ffn_gate found")
    sys.exit(1)
name = want.name
info = gguf.gguf_reader.ReaderTensor
print("using", name, want.shape)
# extract raw bytes
data = np.asarray(want.data)
print("data shape", data.shape, data.dtype, "nbytes", data.nbytes)
data.tofile(r"<TEMP>\v100\q6k_gate.raw")
# also extract the matching activations? not needed
# dequantize the first super-block on the host for a sanity check
with open(r"<TEMP>\v100\q6k_gate.raw","rb") as f:
    b = f.read(210*3)
print("first 210 bytes:", b[:16].hex())
# save the shape
with open(r"<TEMP>\v100\q6k_gate.txt","w") as f:
    f.write(f"{name} {want.shape} ne0={want.shape[0]} ne1={want.shape[1]}\n")
