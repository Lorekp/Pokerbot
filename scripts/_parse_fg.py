import re, collections, bisect, sys

svg = open("artifacts/fg_bench_trainer.svg", "r", encoding="utf-8", errors="replace").read()
print(f"read {len(svg)} bytes", flush=True)
tm = re.search(r'total_samples="(\d+)"', svg)
total = int(tm.group(1)) if tm else 0

name_re = re.compile(r'(.*?) \((\d[\d,]*) samples')
y_re = re.compile(r'y="([\d.]+)"')
fgx_re = re.compile(r'fg:x="(\d+)"')
fgw_re = re.compile(r'fg:w="(\d+)"')

F = []  # (name, fgx, fgw, y)
for blk in svg.split("<g><title>")[1:]:
    head = blk[:200]
    nm = name_re.match(blk)
    ym, xm, wm = y_re.search(head), fgx_re.search(head), fgw_re.search(head)
    if not (nm and ym and xm and wm):
        continue
    F.append((nm.group(1), int(xm.group(1)), int(wm.group(1)), float(ym.group(1))))
print(f"parsed {len(F)} frames  total_samples={total}", flush=True)
if not F:
    sys.exit("no frames")

ys = sorted({f[3] for f in F})
fh = min((b - a for a, b in zip(ys, ys[1:])), default=16.0)
by_y = collections.defaultdict(list)
for f in F:
    by_y[round(f[3], 1)].append(f)
for y in by_y:
    by_y[y].sort(key=lambda f: f[1])

CHILD_DY = float(sys.argv[1]) if len(sys.argv) > 1 else -fh  # try -fh or +fh

def child_samples(fgx, fgw, y):
    lvl = by_y.get(round(y + CHILD_DY, 1))
    if not lvl:
        return 0
    xs = [f[1] for f in lvl]
    i = bisect.bisect_left(xs, fgx)
    s = 0
    while i < len(lvl) and lvl[i][1] < fgx + fgw:
        if lvl[i][1] + lvl[i][2] <= fgx + fgw:
            s += lvl[i][2]
        i += 1
    return s

self_by = collections.Counter()
incl_by = collections.Counter()
for name, fgx, fgw, y in F:
    self_by[name] += max(0, fgw - child_samples(fgx, fgw, y))
    incl_by[name] += fgw

base = lambda n: n.split("`")[-1]
resolved = lambda n: not base(n).startswith("0x")
def short(n):
    s = re.sub(r"<.*?>", "<>", base(n).replace("battery_native::", "bn::"))
    return (s[:92] + "...") if len(s) > 95 else s

res = sum(s for n, s in self_by.items() if resolved(n))
print(f"frame_h={fh}  RESOLVED self-time={100*res/total:.1f}% (rest=unresolved 0x)\n", flush=True)
print("=== TOP RESOLVED SELF-TIME (where CPU sat) ===", flush=True)
shown = 0
for name, sw in self_by.most_common():
    if resolved(name) and sw > 0:
        print(f"  self {100*sw/total:5.2f}% | incl {100*incl_by[name]/total:5.2f}%  {short(name)}", flush=True)
        shown += 1
        if shown >= 32:
            break
