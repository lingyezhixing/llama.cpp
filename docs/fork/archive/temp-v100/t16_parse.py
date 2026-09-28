import io, os, csv

T = os.path.join(os.environ['TEMP'], 'v100')
names = ['base-grid80', 'blocks192', 'blocks96', 'blocks160', 'pb2-grid384']
print("%-14s %10s %8s %10s %10s %8s" % ("config", "FA ms", "FA n", "fixup ms", "fixup n", "FA us/launch"))
for n in names:
    p = os.path.join(T, 't16_ks_%s.csv' % n)
    lines = [l.rstrip('\n') for l in io.open(p, encoding='utf-8', errors='replace')]
    hdr = None
    for i, l in enumerate(lines):
        if l.startswith('Time (%),Total Time (ns)'):
            hdr = i
            break
    fa_ms = fa_n = fx_ms = fx_n = 0.0
    for row in csv.reader(lines[hdr:len(lines)]):
        if len(row) < 9:
            continue
        try:
            tot = float(row[1])
            cnt = int(row[2])
        except ValueError:
            continue
        name = row[8]
        if 'flash_attn_ext_f16' in name:
            fa_ms += tot / 1e6
            fa_n += cnt
        elif 'stream_k_fixup' in name or 'combine_results' in name:
            fx_ms += tot / 1e6
            fx_n += cnt
    us = fa_ms * 1000.0 / fa_n if fa_n else 0
    print("%-14s %10.2f %8d %10.2f %10d %8.2f" % (n, fa_ms, fa_n, fx_ms, fx_n, us))
