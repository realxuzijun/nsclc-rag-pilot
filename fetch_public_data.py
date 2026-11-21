"""Download public source documents locally using the checked-in manifest."""
import hashlib
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def download(source):
    path = ROOT / source["path"]
    if path.is_file():
        content = path.read_bytes()
        action = "kept"
    else:
        request = Request(source["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
            media_type = response.headers.get_content_type()
        if path.suffix == ".pdf" and not content.startswith(b"%PDF-"):
            raise ValueError("download is not a PDF")
        if path.suffix == ".html" and media_type != "text/html":
            raise ValueError(f"expected HTML, received {media_type}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        partial.write_bytes(content)
        partial.replace(path)
        action = "downloaded"
    digest = hashlib.sha256(content).hexdigest()
    print(f"{action} {source['path']} {digest}")
    if digest != source["sha256"]:
        print(f"  differs from the reference snapshot ({source['version']}); verify the source")


def main():
    manifest = json.loads((ROOT / "public_sources.json").read_bytes())
    failed = False
    for source in manifest:
        try:
            download(source)
        except (OSError, URLError, ValueError) as exc:
            print(f"{source['path']}: {exc}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
