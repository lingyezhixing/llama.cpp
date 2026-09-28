import re, os, collections
p = os.path.expandvars(r'%TEMP%\v100\cublas_log.txt')
txt = open(p, encoding='utf-8', errors='replace').read()
# split into call blocks
blocks = re.split(r'(?=I! cuBLAS \(v[\d.]+\) function )', txt)
calls = collections.Counter()
detail = {}
for b in blocks:
    mfn = re.search(r'function cublasStatus_t __cdecl (cublas\w+)\(', b)
    if not mfn: continue
    fn = mfn.group(1)
    if fn not in ('cublasGemmEx','cublasGemmStridedBatchedEx','cublasGemmBatchedEx','cublasSgemm_v2','cublasSgemm'): continue
    def g(key):
        mm = re.search(r'\b' + key + r': type=[^;]+; val=([^\r\n]+)', b)
        return mm.group(1).strip() if mm else None
    m = g('m'); n = g('n'); k = g('k'); lda = g('lda'); ldb = g('ldb'); ldc = g('ldc')
    ta = g('transa'); tb = g('transb')
    at = g('Atype'); bt = g('Btype'); ct = g('Ctype') or g('Ctype')
    ctype = re.search(r'C: type=void; val=[^\r\n]+[\r\n]+i!\s+Ctype: type=cudaDataType_t; val=([^\r\n]+)', b)
    ctype = ctype.group(1).strip() if ctype else None
    comp = re.search(r'computeType: type=cublasComputeType_t; val=([^\r\n]+)', b)
    algo = re.search(r'algo: type=cublasGemmAlgo_t; val=([^\r\n]+)', b)
    batch = re.search(r'batchCount: type=int; val=([^\r\n]+)', b)
    key = (fn, m, n, k, lda, at, ct, ctype, comp.group(1).strip() if comp else None, algo.group(1).strip() if algo else None, batch.group(1).strip() if batch else None)
    calls[key] += 1
    detail[key] = (ta, tb, at, bt, ctype)
print(f'{"count":>5}  {"fn":30} {"m":>7} {"n":>6} {"k":>6} {"lda":>6} {"batch":>5}  compute/algo/Ctype')
for key, c in calls.most_common(40):
    fn, m, n, k, lda, at, ct, ctype, comp, algo, batch = key
    print(f'{c:5d}  {fn:30} {m:>7} {n:>6} {k:>6} {lda:>6} {str(batch):>5}  {comp}/{algo}/{ctype}')
