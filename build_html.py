"""Inject data/app_data.json into index.html (idempotent; run after model.py)."""
import json
import re

data = json.dumps(json.load(open("data/app_data.json")), separators=(",", ":"))
data = data.replace("</", "<\\/")  # keep the inline <script> block unbreakable

html = open("index.html").read()
html = re.sub(
    r'(<script id="app-data" type="application/json">).*?(</script>)',
    lambda m: m.group(1) + data + m.group(2),
    html, count=1, flags=re.S,
)
open("index.html", "w").write(html)
print(f"index.html rebuilt ({len(html)/1e6:.2f} MB)")
