"""Extract the public transfer cohort without changing the frozen original corpus."""
import hashlib
import json
import re

from bs4 import BeautifulSoup

from build_corpus import ROOT, TOPICS, blocks, clean, gather
from build_structured import split_clauses

URLS = {
    "1872": "https://www.eviq.org.au/radiation-oncology/respiratory/1872-respiratory-lung-primary-palliative-ebrt",
    "1873": "https://www.eviq.org.au/radiation-oncology/respiratory/1873-respiratory-non-small-cell-lung-cancer-adjuva",
    "3679": "https://www.eviq.org.au/radiation-oncology/respiratory/3679-respiratory-small-cell-lung-cancer-consolidat",
}


def build():
    data = ROOT / "data"
    out, sources = [], []
    for code, url in URLS.items():
        path = data / "raw" / f"{code}.html"
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "html.parser")
        head = clean(soup.main)
        version = re.search(rf"\b{code}\s+v\.\d+\b", head).group()
        if not re.search(rf"{code}\s+v\.\d+ Endorsed", head):
            raise ValueError(f"Unexpected public protocol status: {code}")
        title = clean(soup.h1)
        sources.append(dict(source=code, title=title, version=version, url=url,
                            sha256=hashlib.sha256(raw).hexdigest(),
                            snapshot=f"raw/{code}.html", terms="https://www.eviq.org.au/copyright"))
        for panel in soup.select("#protocol > .panel"):
            heading = panel.select_one(".panel-heading")
            body = panel.select_one(".panel-body")
            section = clean(heading.h2) if heading and heading.h2 else None
            if section not in TOPICS or body is None:
                continue
            anchor = heading["id"]
            metadata = dict(source=code, title=title, section=section, version=version,
                            url=f"{url}#{anchor}")
            units = list(blocks(body))
            if section == "Organs at risk":
                text = "\n".join(t for t, table in units if table)
                if not text:
                    raise ValueError(f"No OAR table in {code}")
                out.append(dict(metadata, id=f"{code}-{anchor}-I", text=text,
                                kind="table_inventory"))
            atomic = section in {"Indications and patient population", "Organs at risk"}
            for n, passage in enumerate(gather(units, atomic), 1):
                parent = f"{code}-{anchor}-{n}"
                row = dict(metadata, id=parent, text=passage)
                out.extend(dict(row, id=f"{parent}-L{i:03d}", parent_id=parent, text=clause)
                           for i, clause in enumerate(split_clauses(row), 1))
    (data / "external_clauses.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    (data / "external_sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    print(len(sources), "public protocols;", len(out), "clauses and table inventories")


if __name__ == "__main__":
    build()

