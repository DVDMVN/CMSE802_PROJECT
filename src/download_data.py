import os
import requests
from zipfile import ZipFile
from pathlib import Path
import tempfile

from src.config import DATA_PATH, DATA_LINK

def download_and_extract_raw_data():
    output_dir = Path(DATA_PATH)
    if os.path.exists(DATA_PATH):
        print("DATA ALREADY EXISTS")
        return
    print(f"DOWNLOADING AND EXTRACTING DATA TO: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp_path = Path(tmp.name)
        with requests.get(DATA_LINK, stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:  # filter out keep-alive chunks
                    tmp.write(chunk)

    # Extract
    extracted_paths: list[Path] = []
    try:
        with ZipFile(tmp_path) as zf:
            zf.extractall(output_dir)
            for name in zf.namelist():
                extracted_paths.append(output_dir / name)
    finally:
        # Clean up the downloaded zip
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    print("\tFINISHED DOWNLOADING AND EXTRACTING")
    return extracted_paths

if __name__ == '__main__':
    download_and_extract_raw_data()