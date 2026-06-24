import os
import json
import requests
from datetime import datetime, timezone
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─── Add your PDF URLs here anytime ───────────────
REGULATIONS = {
    # "dpdp_act": "https://example.com/dpdp_act_2023.pdf",
    # "rbi_pa_guidelines": "https://example.com/rbi_guidelines.pdf",
}
# ──────────────────────────────────────────────────

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def download_all():
    os.makedirs(RAW_DIR, exist_ok=True)

    if not REGULATIONS:
        print("No URLs configured in REGULATIONS dict. Skipping download.")
        print(f"Place PDFs manually in: {RAW_DIR}")
        return

    manifest = {}

    for name, url in REGULATIONS.items():
        filename = f"{name}.pdf"
        filepath = os.path.join(RAW_DIR, filename)

        if os.path.exists(filepath):
            print(f"⏭  Already exists, skipping: {filename}")
            continue

        print(f"⬇  Downloading: {name}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))

            with open(filepath, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=filename
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

            file_size_kb = os.path.getsize(filepath) // 1024

            manifest[name] = {
                "filename": filename,
                "source_url": url,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "file_size_kb": file_size_kb
            }

            print(f"✅ Downloaded: {filename} ({file_size_kb} KB)")

        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")

    if manifest:
        manifest_path = os.path.join(RAW_DIR, "manifest.json")

        # merge with existing manifest if present
        existing = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                existing = json.load(f)

        existing.update(manifest)

        with open(manifest_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"\n📄 Manifest updated: {manifest_path}")


if __name__ == "__main__":
    download_all()