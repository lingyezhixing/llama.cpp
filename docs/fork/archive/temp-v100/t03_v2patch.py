import io, re

p = r'<TEMP>\v100\t03_chunked.cu'
s = io.open(p, encoding='utf-8').read()

# ---- (a) DP define ----
s = s.replace('#define WP (L+1)                // padded row length for the transposed solve buffers',
              '#define WP (L+1)                // padded row length for the transposed solve buffers\n#define DP (D+1)                // padded state/key row length (bank-conflict free)')

# ---- (b) buffer offsets / sizes: sK and sM padded, add sC ----
s = s.replace('    float * sK   = smem;                    // [L][D]\n    float * sKKT = sK   + L*D;              // [L][L]',
              '    float * sK   = smem;                    // [L][DP]\n    float * sKKT = sK   + L*DP;             // [L][L]')
s = s.replace('    float * sOS  = sQK  + L*L;              // [L][COLS] OS = sum_s QK*T1  (then unused)',
              '    float * sOS  = sQK  + L*L;              // [L][COLS] OS = sum_s QK*T1\n    float * sC   = sOS  + L*COLS;            // [L][L]   C (separate buffer)')
s = s.replace('    float * sAc  = sOS  + L*COLS;           // [L]',
              '    float * sAc  = sC   + L*L;              // [L]')

# ---- (c) sK accesses -> DP ----
s = s.replace('''        for (int idx = tid; idx < L*D; idx += blockDim.x) {
            const int t = idx / D, i = idx % D;
            sK[idx] = (t < len) ? k[(size_t)((t0+t)*H + h)*D + i] : 0.f;
        }''',
'''        for (int idx = tid; idx < L*D; idx += blockDim.x) {
            const int t = idx / D, i = idx % D;
            sK[t*DP + i] = (t < len) ? k[(size_t)((t0+t)*H + h)*D + i] : 0.f;
        }''')
s = s.replace('if (t >= s && s < len) { for (int i = 0; i < D; i++) acc += sK[t*D+i]*sK[s*D+i]; }',
              'if (t >= s && s < len) { for (int i = 0; i < D; i++) acc += sK[t*DP+i]*sK[s*DP+i]; }')
s = s.replace('if (kind == 0) rhs = sK[t*D + col];', 'if (kind == 0) rhs = sK[t*DP + col];')
s = s.replace('for (int i = 0; i < D; i++) acc += q[(size_t)((t0+t)*H + h)*D + i]*sK[s*D + i];',
              'for (int i = 0; i < D; i++) acc += q[(size_t)((t0+t)*H + h)*D + i]*sK[s*DP + i];')
s = s.replace('acc += sK[t*D + i]*(sUT[jj*WP + t] - sT1[t*COLS + jj]);',
              'acc += sK[t*DP + i]*(sUT[jj*WP + t] - sT1[t*COLS + jj]);')

# ---- (d) sM accesses -> DP ----
s = s.replace('        sM[idx] = s_in[(size_t)h*D*D + (j0+jj)*D + ii];',
              '        sM[jj*DP + ii] = s_in[(size_t)h*D*D + (j0+jj)*D + ii];')
s = s.replace('                    mv = sM[jj*D + m];', '                    mv = sM[jj*DP + m];')
s = s.replace('''            for (int t = 0; t < len; t++) {
                acc += sK[t*DP + i]*(sUT[jj*WP + t] - sT1[t*COLS + jj]);
            }
            sM[idx] += acc;''',
'''            for (int t = 0; t < len; t++) {
                acc += sK[t*DP + i]*(sUT[jj*WP + t] - sT1[t*COLS + jj]);
            }
            sM[jj*DP + i] += acc;''')
s = s.replace('for (int idx = tid; idx < COLS*D; idx += blockDim.x) sM[idx] *= acal;',
              'for (int idx = tid; idx < COLS*D; idx += blockDim.x) sM[(idx/D)*DP + (idx%D)] *= acal;')
s = s.replace('''    for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
        const int jj = idx / D, ii = idx % D;
        s_out[(size_t)h*D*D + (j0+jj)*D + ii] = sM[idx];
    }
}''',
'''    for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
        const int jj = idx / D, ii = idx % D;
        s_out[(size_t)h*D*D + (j0+jj)*D + ii] = sM[jj*DP + ii];
    }
}''')
s = s.replace('            dbg[DBSTATEOFF + blockIdx.z*COLS*D + idx] = sM[idx];',
              '            const int jj = idx / D, ii = idx % D;\n            dbg[DBSTATEOFF + blockIdx.z*COLS*D + idx] = sM[jj*DP + ii];')

# ---- (e) C phase: parallel over (t,s), write to sC ----
s = s.replace('''        // C = QK - tril(QK,0) @ R  (in place over sQK; one thread per row, reads only its own row)
        for (int t = tid; t < L; t += blockDim.x) {
            float row[L];
            for (int s = 0; s < L; s++) row[s] = sQK[t*L + s];
            for (int s = 0; s < L; s++) {
                float acc = 0.f;
                for (int m = s+1; m <= t; m++) acc += row[m]*sRT[s*WP + m];
                sQK[t*L + s] = row[s] - acc;
            }
        }
        __syncthreads();''',
'''        // C = QK - tril(QK,0) @ R   (parallel over (t,s) pairs; QK stays intact)
        for (int idx = tid; idx < L*L; idx += blockDim.x) {
            const int t = idx / L, s = idx % L;
            float acc = 0.f;
            if (s <= t) {
                for (int m = s+1; m <= t; m++) acc += sQK[t*L + m]*sRT[s*WP + m];
            }
            sC[t*L + s] = ((s <= t) ? sQK[t*L + s] : 0.f) - acc;
        }
        __syncthreads();''')

# ---- (f) output loop reads sC ----
s = s.replace('                intra += sQK[t*L + s]*sBt[s]*v[(size_t)(h*t_tokens + t0 + s)*D + (j0+jj)];',
              '                intra += sC[t*L + s]*sBt[s]*v[(size_t)(h*t_tokens + t0 + s)*D + (j0+jj)];')
s = s.replace('            for (int idx = tid; idx < L*COLS; idx += blockDim.x) dbg[off + idx] = sOS[idx];\n            for (int idx = tid; idx < L*L; idx += blockDim.x)    dbg[off + L*COLS + idx] = sQK[idx];  // C',
              '            for (int idx = tid; idx < L*COLS; idx += blockDim.x) dbg[off + idx] = sOS[idx];\n            for (int idx = tid; idx < L*L; idx += blockDim.x)    dbg[off + L*COLS + idx] = sC[idx];  // C')

# ---- (g) smem size + occupancy report ----
s = s.replace('    const size_t smem_bytes = (L*D + L*L + D*WP + COLS*WP + L*WP + L*COLS + L*L + L*COLS + 2*L + COLS*D)*4;',
              '    const size_t smem_bytes = (L*DP + L*L + D*WP + COLS*WP + L*WP + L*COLS + L*L + L*COLS + L*L + 2*L + COLS*DP)*4;')
s = s.replace('    printf("smem = %.1f KB\\n", smem_bytes/1024.0);',
              '''    printf("smem = %.1f KB\\n", smem_bytes/1024.0);
    {
        int occ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, chunked_kernel, 256, smem_bytes);
        printf("occupancy: %d block/SM (smem budget 96KB) -> %d warps/SM\\n", occ, occ*8);
    }''')

io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('V2 patch applied')
