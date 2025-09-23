import json
import shutil
from pathlib import Path
import subprocess
import datetime

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication"
NB = ROOT / "main.ipynb"

FIGS = PUB / "figures"
EXAMPLES = PUB / "examples"
MANIFEST = PUB / "manifest.json"


def export_notebook():
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    out_html = PUB / f"main_export_{ts.replace(':','-')}.html"

    # Export notebook to HTML using nbconvert if available
    try:
        subprocess.check_call([
            "python", "-m", "jupyter", "nbconvert", "--to", "html", str(NB), "--output", str(out_html)
        ])
        print(f"Exported notebook to: {out_html}")
    except Exception as e:
        print(f"Note: nbconvert export failed or not available: {e}")

    # Stamp manifest with generation time
    if MANIFEST.exists():
        try:
            data = json.loads(MANIFEST.read_text())
            data["generated_at"] = ts
            MANIFEST.write_text(json.dumps(data, indent=2))
            print("Updated manifest timestamp.")
        except Exception as e:
            print(f"Warning: could not update manifest: {e}")


def main():
    PUB.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    EXAMPLES.mkdir(exist_ok=True)
    export_notebook()


if __name__ == "__main__":
    main()
