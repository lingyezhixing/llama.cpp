import io
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
t = r'<TEMP>\v100'


def extract(name):
    s = io.open(t + '\\t19_gen_' + name + '.txt', encoding='utf-8', errors='replace').read()
    return s.replace('\ufeff', '').strip()


a = extract('OURS')
b = extract('STOCK')
n = min(len(a), len(b))
same = sum(1 for i in range(n) if a[i] == b[i])
first_diff = next((i for i in range(n) if a[i] != b[i]), None)
print('OURS  chars:', len(a))
print('STOCK chars:', len(b))
print('identical prefix: %d chars (%.2f%% of shorter)' % (same, 100.0 * same / max(n, 1)))
print('first diff at:', first_diff)
print('--- OURS  head ---')
print(a[:240].replace('\n', ' | '))
print('--- STOCK head ---')
print(b[:240].replace('\n', ' | '))
if first_diff is not None:
    lo = max(0, first_diff - 60)
    print('--- OURS  around diff ---')
    print(a[lo:first_diff + 60].replace('\n', ' | '))
    print('--- STOCK around diff ---')
    print(b[lo:first_diff + 60].replace('\n', ' | '))
