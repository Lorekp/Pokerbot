import re, collections, sys
svg = open(sys.argv[1], encoding='utf-8').read()
titles = re.findall(r'<title>(.*?)</title>', svg)
pat = re.compile(r'^(.*?) \(([\d,]+) samples, ([\d.]+)%\)$')
rows = []
for t in titles:
    m = pat.match(t)
    if not m:
        continue
    name, samp, pct = m.group(1), int(m.group(2).replace(',', '')), float(m.group(3))
    rows.append((pct, samp, name))
rows.sort(reverse=True)
print('TOP CUMULATIVE FRAMES (py-spy):')
seen = set()
shown = 0
for pct, samp, name in rows:
    short = name.replace('\\', '/').split('/')[-1]
    if short in seen:
        continue
    seen.add(short)
    print(f'  {pct:6.2f}%  {samp:7d}  {short}')
    shown += 1
    if shown >= 34:
        break
