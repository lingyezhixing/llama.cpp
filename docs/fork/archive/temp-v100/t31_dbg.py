import re
line = open('t31_synth.err', encoding='ascii').read().splitlines()[2]
pat = re.compile(
    r'statistics\s+(\S+): #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), '
    r'#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), '
    r'#gen tokens =\s*(\d+), #acc tokens =\s*(\d+)'
    r'(?:, #mean acc len = ([\d.]+), #acc rate/pos = \(([^)]*)\))?'
    r'(?:, dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms)?')
m = pat.search(line)
print("match:", bool(m))
if m:
    for i,g in enumerate(m.groups(),1):
        print(f"  group {i}: {g!r}")
