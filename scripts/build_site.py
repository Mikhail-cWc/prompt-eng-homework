"""Сборка публичной галереи из фактических результатов, без доступа к .env."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
site = ROOT / "site"
items = []
for directory in sorted((ROOT / "outputs").glob("0*")):
    manifest = directory / "manifest.json"
    if not manifest.exists():
        continue
    item = json.loads(manifest.read_text())
    target = site / "assets" / directory.name
    target.mkdir(parents=True, exist_ok=True)
    item["dialogue"] = []
    for path in sorted(directory.glob("0*.json")):
        record = json.loads(path.read_text())
        item["dialogue"].append({k: record[k] for k in
                                ["agent", "model", "api", "input", "output", "parsed", "elapsed_seconds"]})
    for name in ["v1.png", "v2.png", "sd-turbo.png"]:
        if (directory / name).exists():
            shutil.copy2(directory / name, target / name)
    item["openweight"] = None
    if (directory / "sd-turbo.json").exists():
        item["openweight"] = json.loads((directory / "sd-turbo.json").read_text())
    items.append(item)

payload = json.dumps(items, ensure_ascii=False, indent=2)
(site / "data.js").write_text("window.ARTWORKS = " + payload + ";\n", encoding="utf-8")
(site / "collection.json").write_text(payload, encoding="utf-8")
print(f"Галерея собрана: {len(items)} картин")
