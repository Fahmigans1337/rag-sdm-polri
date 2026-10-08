"""Cek sintaks JS di index.html + pastikan semua id yang dipakai JS ada di HTML."""
import re, sys, subprocess, shutil, tempfile, pathlib

html = pathlib.Path("app/static/index.html").read_text(encoding="utf-8")
scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
print(f"Jumlah <script>: {len(scripts)}")

ids_html = set(re.findall(r'id="([^"]+)"', html))
used = set(re.findall(r"\$\('([^']+)'\)", "\n".join(scripts)))
missing = sorted(used - ids_html)
print("ID dipakai JS tapi tidak ada di HTML:", missing or "TIDAK ADA (OK)")

node = shutil.which("node")
if not node:
    print("node tidak ada - lewati syntax check")
    sys.exit(0)
ok = True
for i, s in enumerate(scripts):
    f = pathlib.Path(tempfile.gettempdir()) / f"chk_{i}.js"
    f.write_text(s, encoding="utf-8")
    r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    print(f"script[{i}] syntax:", "OK" if r.returncode == 0 else "ERROR\n" + r.stderr[:600])
    ok &= r.returncode == 0
sys.exit(0 if ok else 1)
