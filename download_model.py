from pathlib import Path
from urllib.request import Request, urlopen

MODEL_URL = (
    "https://huggingface.co/notgoodkeeper/"
    "cnn-based-drowsiness-detection/resolve/main/model.onnx"
)
MODEL_PATH = Path("models/drowsiness_model.onnx")

def download():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        print(f"Model already exists: {MODEL_PATH}")
        return

    print("Downloading pretrained ONNX drowsiness model...")
    print(MODEL_URL)
    req = Request(MODEL_URL, headers={"User-Agent": "ClassroomDrowsiness/1.0"})
    with urlopen(req, timeout=60) as response, open(MODEL_PATH, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\rDownloaded {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB", end="")
    print()
    print(f"Saved to {MODEL_PATH}")

if __name__ == "__main__":
    download()
