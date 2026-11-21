"""Build source-addressable chunks from the five public HTML snapshots."""
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
URLS = {
    "1871": "https://www.eviq.org.au/radiation-oncology/respiratory/1871-respiratory-non-small-cell-lung-cancer-defini",
    "3389": "https://www.eviq.org.au/radiation-oncology/respiratory/3389-respiratory-non-small-cell-lung-cancer-defini",
    "3697": "https://www.eviq.org.au/radiation-oncology/respiratory/3697-respiratory-non-small-cell-lung-cancer-defini",
    "4018": "https://www.eviq.org.au/radiation-oncology/respiratory/4018-respiratory-non-small-cell-lung-cancer-defini",
    "pdq": "https://www.cancer.gov/types/lung/hp/non-small-cell-lung-treatment-pdq"
}
TOPICS = {
    "Indications and patient population", "Clinical information", "Simulation",
    "Dose prescription", "Target volumes", "Organs at risk", "Target verification",
    "Treatment planning", "Treatment delivery", "Side effects", "Evidence",
}
STAGES = {"_514", "_544", "_48552", "_637", "_4889"}


def clean(node):
    return " ".join(node.get_text(" ", strip=True).replace("\u200b", "").split())


def render_table(table):
    rows = []
    active = {}
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = row.find_all(["th", "td"], recursive=False)
        width = sum(int(cell.get("colspan", 1)) for cell in cells)
        # Some eviQ header rowspans extend into an already complete data row.
        if rows and width >= max(len(line.split(" | ")) for line in rows):
            active.clear()
        values = {}
        for position, (remaining, value) in list(active.items()):
            values[position] = value
            if remaining > 1:
                active[position] = (remaining - 1, value)
            else:
                del active[position]
        position = 0
        for cell in cells:
            value = clean(cell)
            for _ in range(int(cell.get("colspan", 1))):
                while position in values:
                    position += 1
                values[position] = value
                remaining = int(cell.get("rowspan", 1)) - 1
                if remaining > 0:
                    active[position] = (remaining, value)
                position += 1
        if values:
            rows.append(" | ".join(values.get(i, "") for i in range(max(values) + 1)))
    caption = table.find("caption")
    return "\n".join(([clean(caption)] if caption else []) + rows)


def blocks(body):
    kinds = {"h3", "h4", "h5", "p", "ul", "ol", "table"}
    for node in body.find_all(list(kinds)):
        if any(p.name in kinds for p in node.parents if p is not body):
            continue
        value = render_table(node) if node.name == "table" else clean(node)
        if value:
            yield value, node.name == "table"


def gather(items, atomic=False):
    if atomic:
        return ["\n".join(value for value, _ in items)]
    chunks, pending, words = [], [], 0
    for value, is_table in items:
        size = len(value.split())
        if pending and (is_table or words + size > 300):
            chunks.append("\n".join(pending))
            pending, words = [], 0
        pending.append(value)
        words += size
    if pending:
        chunks.append("\n".join(pending))
    return chunks


def build():
    data = ROOT / "data"
    chunks, sources = [], []
    for key, url in URLS.items():
        snapshot = data / "raw" / f"{key}.html"
        html = snapshot.read_bytes()
        soup = BeautifulSoup(html, "html.parser")
        title = clean(soup.h1)
        nci = key == "pdq"
        if nci:
            groups = [(clean(sec.h2), sec["id"], sec)
                      for sec in soup.select(".summary-sections section")
                      if sec.get("id") in STAGES and sec.h2]
            version = clean(soup.select_one("section#_48494 h2"))
            license_url = "https://www.cancer.gov/policies/copyright-reuse"
        else:
            groups = []
            for panel in soup.select("#protocol > .panel"):
                heading = panel.select_one(".panel-heading")
                body = panel.select_one(".panel-body")
                if heading and heading.h2 and clean(heading.h2) in TOPICS and body:
                    groups.append((clean(heading.h2), heading["id"], body))
            version = re.search(rf"{key}\s+v\.\d+", clean(soup.main)).group()
            license_url = "https://www.eviq.org.au/copyright"
            flags = soup.select_one("#flags")
            if flags:
                groups.insert(0, ("Protocol notes and staging edition", "flags", flags))
        assert groups, f"No selected sections: {key}"
        sources.append(dict(source=key, url=url, title=title, version=version,
                            sha256=hashlib.sha256(html).hexdigest(), terms=license_url,
                            snapshot=f"raw/{key}.html"))
        for heading, anchor, body in groups:
            items = list(blocks(body))
            if nci:
                items = [(part, False) for text, table in items if not table
                         for part in re.findall(r"(?:\S+\s*){1,250}", text)
                         if re.search(r"\bradiation\b|\bradiotherapy\b|\bSBRT\b|\bSABR\b", part, re.I)]
            atomic = heading in {"Organs at risk", "Dose prescription",
                                 "Indications and patient population"}
            for number, passage in enumerate(gather(items, atomic), 1):
                if passage:
                    chunks.append(dict(id=f"{key}-{anchor}-{number}", source=key,
                                       title=title, section=heading, version=version,
                                       url=f"{url}#{anchor}", text=passage))
    (data / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (data / "chunks.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    print(f"{len(sources)} sources, {len(chunks)} chunks")


if __name__ == "__main__":
    build()
