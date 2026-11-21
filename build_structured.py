"""Build a source-linked SQLite dose table and clause index from frozen snapshots."""
from contextlib import closing
import hashlib
import json
import re
import sqlite3

from benchmark import ROOT

OAR_HEADINGS = {
    "1871": ("1871-5409-1", 5),
    "3697": ("3697-148370-1", 5),
    "3389": ("3389-146953-1", 10),
    "4018": ("4018-156549-1", 9),
}
ORGAN = {
    "lung": r"正常肺|双肺|肺(?=的?(?:平均|最大|V|D|剂量|约束|限制))|\blungs?\b",
    "heart": r"心脏|心臟|\bheart\b",
    "oesophagus": r"食管|食道|\b(?:o|e)esophag\w*",
    "spinal cord": r"脊髓|spinal cord",
    "brachial plexus": r"臂丛|brachial plexus",
    "chest wall": r"胸壁|chest wall",
    "great vessels": r"大血管|great vessels",
    "trachea/proximal bronchi": r"气管|支气管|trachea|proximal bronchi",
    "skin": r"皮肤|skin",
    "stomach": r"胃|stomach",
}
DOSIMETRIC_LABEL = (
    r"Max(?:imum)?(?:\s+dose)?|Mean(?:\s+\w+)?(?:\s+dose)?|"
    r"D\s*\d+(?:\.\d+)?\s*(?:cc|cm³|%)|V\s*\d+(?:\.\d+)?(?:\s*Gy)?"
)
METRIC = re.compile(rf"\b{DOSIMETRIC_LABEL}", re.I)
FRACTIONS = re.compile(r"(\d+)\s*(?:次|fractions?\b)", re.I)


def metric_key(label):
    label = re.sub(r"\s+", "", label.lower())
    if label.startswith(("max", "mean")):
        return label[:3] if label.startswith("max") else "mean"
    return re.sub(r"gy$", "", label.replace("cm³", "cc"))


def build_doses(db, chunks):
    count = 0
    for source, (chunk_id, organ_count) in OAR_HEADINGS.items():
        row = chunks[chunk_id]
        lines = row["text"].splitlines()
        grid = [(i, [cell.strip() for cell in line.split("|")])
                for i, line in enumerate(lines) if "|" in line]
        if source in {"3389", "4018"}:
            _, headings = grid.pop(0)
            _, regimen = grid.pop(0)
            assert all("prescription" in heading.lower() for heading in headings[1:-1])
            columns = [(i, int(FRACTIONS.search(header).group(1)), header)
                       for i, header in enumerate(regimen[1:-1], 1)]
        else:
            _, headings = grid.pop(0)
            assert headings[1] == "Constraints"
            columns = [(1, 0, headings[1])]
        assert len(grid) == organ_count, (source, len(grid))
        tail = "\n".join(lines[grid[-1][0] + 1:])
        for ri, (line_no, cells) in enumerate(grid, 1):
            assert len(cells) == len(columns) + 2, (source, cells)
            organ = cells[0].rstrip("*").lower()
            if organ == "normal lung":
                organ = "lung"
            assert organ in ORGAN, (source, organ)
            for ci, fractions, label in columns:
                cell_id = f"{chunk_id}-T{ri:02d}C{ci}"
                db.execute("INSERT INTO dose_cell VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           (cell_id, source, row["version"], row["url"], chunk_id,
                            organ, fractions, label, cells[ci], cells[-1], tail, line_no + 1))
                metrics = list(METRIC.finditer(cells[ci]))
                for mi, match in enumerate(metrics):
                    end = metrics[mi + 1].start() if mi + 1 < len(metrics) else len(cells[ci])
                    literal = re.sub(r"\s+r\s*$", "", cells[ci][match.start():end].strip())
                    db.execute("INSERT INTO dose_metric VALUES (?,?,?,?)",
                               (cell_id, metric_key(match.group()), mi, literal))
                count += 1
    return count


SCHEMA = """
CREATE TABLE dose_cell (
    id TEXT PRIMARY KEY, protocol TEXT NOT NULL, version TEXT NOT NULL,
    source_url TEXT NOT NULL, parent_id TEXT NOT NULL, organ TEXT NOT NULL,
    fractions INTEGER NOT NULL, regimen TEXT NOT NULL, cell TEXT NOT NULL,
    row_notes TEXT NOT NULL, table_notes TEXT NOT NULL, source_line INTEGER NOT NULL,
    UNIQUE (protocol, version, organ, fractions)
);
CREATE INDEX dose_lookup ON dose_cell(protocol, organ, fractions);
CREATE TABLE dose_metric (
    cell_id TEXT NOT NULL REFERENCES dose_cell(id),
    metric TEXT NOT NULL, ordinal INTEGER NOT NULL, literal TEXT NOT NULL,
    PRIMARY KEY (cell_id, metric, ordinal)
);
"""


def split_clauses(row):
    prefix = ""
    for line in row["text"].splitlines():
        if "|" in line:
            if line.startswith(("Organ |", "Simulation |", "Modality", "Volume |")):
                prefix = line
                continue
            yield f"{prefix}\n{line}" if prefix else line
        else:
            for unit in re.split(r"(?<!e\.g\.)(?<!i\.e\.)(?<=[.?!])\s+(?:r\s+)*(?=[A-Z])", line):
                if len(unit.split()) >= 4:
                    yield unit.strip()


def build(root):
    data = root / "data"
    raw = (data / "chunks.jsonl").read_bytes()
    chunks = {row["id"]: row for row in map(json.loads, raw.splitlines())}
    db_path = data / "dose_tables.sqlite"
    draft = db_path.with_suffix(".tmp")
    if draft.exists():
        draft.unlink()
    with closing(sqlite3.connect(draft)) as db, db:
        db.executescript(SCHEMA)
        count = build_doses(db, chunks)
    draft.replace(db_path)
    clauses = [dict(row, id=f"{row['id']}-L{i:03d}", parent_id=row["id"], text=clause)
               for row in chunks.values()
               for i, clause in enumerate(split_clauses(row), 1)]
    (data / "clauses.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in clauses), encoding="utf-8")
    print(f"{count} dose cells, {len(clauses)} clauses, source SHA-256 {hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    build(ROOT)

