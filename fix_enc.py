import re, sys
p = sys.argv[1]
raw = open(p, "rb").read().decode("utf-8")

def fix(line: str) -> str:
    if not re.search("[\u00c3\u00e2\u00f0]", line):
        return line
    out = bytearray()
    for ch in line:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) < 256:
                out.append(ord(ch))
            else:
                return line
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return line

lines = raw.split("\n")
fixed = [fix(l) for l in lines]
changed = sum(a != b for a, b in zip(lines, fixed))
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(fixed))
print("lines fixed:", changed)
bad = [i + 1 for i, l in enumerate(fixed) if re.search("[\u00c3\u00e2\u00f0]", l)]
print("remaining suspicious lines:", bad)
for i, l in enumerate(fixed):
    if any(ord(c) > 127 for c in l):
        print(i + 1, l.strip()[:90])
