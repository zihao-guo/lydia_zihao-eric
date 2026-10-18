"""Build the static frontend with a public HTTPS backend URL, without secrets."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from urllib.parse import urlsplit


def build(api_url: str, output: Path) -> None:
    url = urlsplit(api_url)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path not in ("", "/")):
        raise ValueError("API URL must be an HTTPS origin without credentials, path, query, or fragment")
    source = Path(__file__).resolve().parents[1] / "src/api/index.html"
    html = source.read_text(encoding="utf-8")
    marker = '<meta name="api-base" content="">'
    if html.count(marker) != 1:
        raise ValueError("Frontend must contain exactly one API base marker")
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(
        html.replace(marker, f'<meta name="api-base" content="{escape(api_url.rstrip("/"), quote=True)}">'),
        encoding="utf-8",
    )
    (output / ".nojekyll").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--out", type=Path, default=Path("dist"))
    args = parser.parse_args()
    build(args.api_url, args.out)
