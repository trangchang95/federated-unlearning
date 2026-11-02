"""Render index.html and index-en.html from one source: python build.py <repo_with_results>"""
import subprocess, sys
from pathlib import Path

here = Path(__file__).parent
if len(sys.argv) > 1:
    subprocess.run([sys.executable, str(here / "extract.py"), sys.argv[1], str(here / "data.json")], check=True, stdout=subprocess.DEVNULL)
shell, app, data = (here / "shell.html").read_text(), (here / "app.js").read_text(), (here / "data.json").read_text()
for lang, other, title, name in [("vi", "en", "Tiến độ Federated Unlearning", "index.html"), ("en", "vi", "Federated Unlearning Progress", "index-en.html")]:
    html = shell.replace("__LANG__", lang).replace("__OTHER__", other).replace("__TITLE__", title).replace("__DATA__", data).replace("__APP__", app)
    (here.parent / name).write_text(html)
