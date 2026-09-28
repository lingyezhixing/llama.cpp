import re
import sys

path = sys.argv[1]

txt = open(path, encoding='utf-8', errors='replace').read()
lines = txt.splitlines()

pats = {
    'ckpt_new':   re.compile(r'created context checkpoint (\d+) of (\d+) \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+), size = ([\d.]+) MiB\)'),
    'ckpt_old':   re.compile(r'erasing old context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+), size = ([\d.]+) MiB\)'),
    'ckpt_close': re.compile(r'erasing context checkpoint too close to an earlier one \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+), size = ([\d.]+) MiB\)'),
    'restored':   re.compile(r'restored context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+), n_past = (\d+)'),
    'do_reset':   re.compile(r'forcing full prompt re-processing due to lack of cache data'),
    'reuse':      re.compile(r'after context reuse, new n_past = (\d+)'),
    'save':       re.compile(r'saving prompt with length (\d+), total state size = ([\d.]+) MiB \(draft: ([\d.]+) MiB\)'),
    'room':       re.compile(r'making room for prompt cache entry, removing oldest entry \(size = ([\d.]+) MiB\)'),
    'skip':       re.compile(r'prompt state size ([\d.]+) MiB exceeds cache size limit ([\d.]+) MiB, skipping'),
    'found':      re.compile(r'found better prompt with f_keep = ([\d.]+), f_sim = ([\d.]+)'),
    'better':     re.compile(r'looking for better prompt, base f_keep = (-?[\d.]+), f_sim = ([\d.]+)'),
    'limit':      re.compile(r'cache (size|token) limit .* reached, removing oldest entry \(size = ([\d.]+) MiB\)'),
    'state':      re.compile(r'cache state: (\d+) prompts, ([\d.]+) MiB \(limits: ([\d.]+) MiB'),
}

counts = {}
events = []
for i, ln in enumerate(lines):
    for name, p in pats.items():
        m = p.search(ln)
        if m:
            counts[name] = counts.get(name, 0) + 1
            events.append((i + 1, name, m.groups()))

print(f"file: {path}")
print(f"lines: {len(lines)}")
print("counts:", {k: counts.get(k, 0) for k in pats})
print()
print("--- events ---")
for ln_no, name, g in events:
    print(f"L{ln_no:<7} {name:<10} {g}")
