"""Build source-linked verification and contour tables from frozen eviQ rows."""
from contextlib import closing
import json
import re
import sqlite3

from benchmark import ROOT

TABLE_DB = ROOT / "data" / "protocol_tables.sqlite"
IMAGING = {
    "1871": "1871-5413-1-L001",
    "3697": "3697-148372-1-L001",
    "3389": "3389-156525-2-L001",
    "4018": "4018-156551-2-L001",
    "1872": "1872-5390-1-L001",
    "1873": "1873-5130-1-L001",
    "3679": "3679-150650-1-L001",
}
METHOD = {"kv_planar": "kV 平片", "mv_planar": "MV 平片", "cbct_generic": "CBCT",
          "cbct_3d": "3D CBCT", "cbct_4d": "4D CBCT"}
ORGANS = {
    "chest wall": "胸壁", "spinal cord": "脊髓", "heart": "心脏",
    "oesophagus": "食管", "stomach": "胃", "skin": "皮肤",
    "trachea/proximal bronchi": "近端气管和支气管", "great vessels": "大血管",
    "lung": "肺",
}
SCHEMA = """
CREATE TABLE source_row (
  id TEXT PRIMARY KEY, protocol TEXT NOT NULL, version TEXT NOT NULL,
  section TEXT NOT NULL, url TEXT NOT NULL, original TEXT NOT NULL
);
CREATE TABLE verification (
  source_id TEXT NOT NULL REFERENCES source_row(id),
  modality TEXT NOT NULL, interval_days INTEGER CHECK(interval_days > 0),
  stage TEXT CHECK(stage='pre_treatment'), during_optional INTEGER NOT NULL,
  match_basis TEXT NOT NULL CHECK(match_basis IN (
    'bone','bone_or_soft','soft_and_or_marker','bone_or_soft_by_ct','bone_check_soft')),
  soft_target TEXT CHECK(soft_target IN ('GTV','iGTV')),
  departmental_protocol INTEGER NOT NULL, oar_evaluation INTEGER NOT NULL,
  modality_en TEXT NOT NULL, frequency_en TEXT NOT NULL, match_en TEXT NOT NULL,
  PRIMARY KEY (source_id, modality)
);
CREATE TABLE verification_phase (
  source_id TEXT NOT NULL REFERENCES source_row(id), ordinal INTEGER NOT NULL,
  first_day INTEGER, last_day INTEGER, after_initial INTEGER NOT NULL,
  maximum_interval_days INTEGER,
  PRIMARY KEY(source_id, ordinal),
  CHECK((after_initial=0 AND first_day IS NOT NULL AND last_day IS NOT NULL
         AND maximum_interval_days IS NULL) OR
        (after_initial=1 AND first_day IS NULL AND last_day IS NULL AND maximum_interval_days > 0))
);
CREATE TABLE cbct_reference (
  source_id TEXT NOT NULL REFERENCES source_row(id), planning_ct_dimension INTEGER NOT NULL,
  soft_target TEXT NOT NULL, PRIMARY KEY(source_id, planning_ct_dimension)
);
CREATE TABLE oar_inventory (
  protocol TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES source_row(id)
);
CREATE TABLE oar_inventory_item (
  protocol TEXT NOT NULL REFERENCES oar_inventory(protocol), organ TEXT NOT NULL,
  source_id TEXT NOT NULL REFERENCES source_row(id), has_note INTEGER NOT NULL,
  PRIMARY KEY(protocol, organ)
);
CREATE TABLE contour (
  source_id TEXT PRIMARY KEY REFERENCES source_row(id), organ TEXT NOT NULL,
  boundary TEXT, components TEXT, description TEXT,
  upper_min REAL, lower_min REAL, unit TEXT, reference TEXT,
  contour_en TEXT NOT NULL
);
CREATE INDEX contour_organ ON contour(organ);
"""


def put_source(db, row):
    db.execute("INSERT OR IGNORE INTO source_row VALUES (?,?,?,?,?,?)",
               tuple(row[k] for k in ("id", "source", "version", "section", "url", "text")))


def imaging_rows(db, rows):
    for protocol, key in IMAGING.items():
        if key not in rows:
            if protocol in {"1872", "1873", "3679"}:
                continue
            raise ValueError(f"Missing frozen verification row: {key}")
        row = rows[key]
        labels, values = [[s.strip() for s in line.split("|")] for line in row["text"].splitlines()]
        assert [x.rstrip(" *") for x in labels] == ["Modality", "Frequency", "Match point"]
        modalities, frequency, match = values
        assert row["source"] == protocol
        daily = 1
        if protocol in {"1871", "3697", "1873"}:
            assert modalities == (
                "kV planar or 3D cone beam CT (3D CBCT) or 4D cone beam CT (4D CBCT)")
            assert frequency == "Daily"
            assert match == ("kV planar: bone Match to bone or soft tissue i.e. GTV "
                             "(3D CBCT) or iGTV (4D CBCT) depending on departmental protocol, "
                             "and evaluation of OARs.")
            fields = [("kv_planar", "bone", None, 0, 0),
                      ("cbct_3d", "bone_or_soft", "GTV", 1, 1),
                      ("cbct_4d", "bone_or_soft", "iGTV", 1, 1)]
            stage, during = None, 0
        elif protocol in {"3389", "4018"}:
            assert modalities == "3D CBCT or 4D CBCT"
            assert frequency == "Daily pre-treatment imaging. Consider intra-fraction imaging."
            assert match == "Soft tissue and/or fiducial markers."
            fields = [(modality, "soft_and_or_marker", None, 0, 0)
                      for modality in ("cbct_3d", "cbct_4d")]
            stage, during = "pre_treatment", 1
        elif protocol == "1872":
            assert modalities == "kV/MV planar or Cone beam CT (CBCT)"
            assert match == ("kV or MV planar: Bone CBCT: Match to bone or soft tissue i.e. "
                             "GTV (3DCT) or iGTV (4DCT) depending on departmental protocol "
                             "and evaluation of OARs.")
            phase = re.fullmatch(r"Day (\d+)-(\d+) then at least weekly", frequency)
            if phase is None:
                raise ValueError("Unrecognized palliative schedule")
            put_source(db, row)
            db.executemany("INSERT INTO verification_phase VALUES (?,?,?,?,?,?)", [
                (key, 1, int(phase[1]), int(phase[2]), 0, None), (key, 2, None, None, 1, 7)])
            db.executemany("INSERT INTO cbct_reference VALUES (?,?,?)",
                           [(key, 3, "GTV"), (key, 4, "iGTV")])
            fields = [("kv_planar", "bone", None, 0, 0), ("mv_planar", "bone", None, 0, 0),
                      ("cbct_generic", "bone_or_soft_by_ct", None, 1, 1)]
            daily, stage, during = None, None, 0
        else:
            assert protocol == "3679"
            assert modalities == "kV planar OR MV planar OR CBCT."
            assert frequency == "Daily." and match == "Bone. On CBCT: check soft tissue alignment."
            fields = [("kv_planar", "bone", None, 0, 0), ("mv_planar", "bone", None, 0, 0),
                      ("cbct_generic", "bone_check_soft", None, 0, 0)]
            stage, during = None, 0
        put_source(db, row)
        for method, basis, ref, departmental, oar in fields:
            db.execute("INSERT INTO verification VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (key, method, daily, stage, during, basis, ref, departmental, oar,
                        modalities, frequency, match))


def contour_fields(organ, note):
    text = note.casefold()
    boundary = components = description = unit = reference = None
    upper = lower = None
    if organ == "chest wall":
        pat = (r"^contour as a structure between the parietal pleura and encompassing the "
               r"intercostal muscles and ribs\. contour from ≥(\d+(?:\.\d+)?) (cm|mm) "
               r"superior to ≥(\d+(?:\.\d+)?) \2 inferior to the ptv\.")
        m = re.search(pat, text)
        if not m:
            return None
        boundary = "以壁层胸膜为界"
        components = "涵盖肋间肌和肋骨"
        upper, unit, lower = float(m[1]), m[2], float(m[3])
        reference = "PTV"
    elif organ == "spinal cord" and "bony" in text:
        description = "按骨性椎管勾画或定义"
        m = re.search(r"extending at ≥(\d+(?:\.\d+)?) (mm|cm) above and below the ptv", text)
        if m:
            upper = lower = float(m[1])
            unit, reference = m[2], "PTV"
    elif organ == "heart" and (
            "contour heart and pericardial sac." in text
            and "just inferior to the left pulmonary artery" in text):
        boundary, components = "上界从左肺动脉正下方开始", "包括心脏和心包"
    elif organ == "oesophagus" and ("external surface of oesophagus" in text
                                     and "from cricoid cartilage to gastro-oesophageal junction" in text):
        boundary, description = "从环状软骨到胃食管交界处", "勾画食管外表面"
        if "should not compromise target coverage" in text:
            description += "；不应为满足食管约束而牺牲靶区覆盖"
    elif organ == "stomach" and ("from the gastro-oesophageal junction to the pylorus" in text
                                  and "including stomach contents" in text):
        boundary, components = "从胃食管交界处到幽门，轮廓到胃外壁外缘", "包含胃内容物"
    elif organ == "skin":
        m = re.search(r"volume the (\d+(?:\.\d+)?) (mm|cm) inner ring of the external body contour", text)
        if m:
            description = f"取体表外轮廓向内 {m[1]} {m[2]} 的环形体积"
    elif organ == "trachea/proximal bronchi" and "as a single structure or as two" in text:
        description = "在肺窗上将近端气管和支气管树勾成一个结构，或分为两个结构"
    elif organ == "great vessels" and "outer fatty adventitia" in text:
        description = "在纵隔窗勾画主动脉、上腔静脉和下腔静脉，覆盖管壁及肌层至外侧脂肪外膜"
    elif organ == "lung" and "whole lungs contoured with no subtraction" in text:
        description = "勾画全肺，不扣除"
    elif organ == "lung" and ("4dct: volume both lungs and subtract igtv" in text
                             and "3dct: volume both lungs and subtract gtv" in text):
        description = "4D CT：双肺减 iGTV；3D CT：双肺减 GTV"
    if not any((boundary, components, description, unit)):
        return None
    return boundary, components, description, upper, lower, unit, reference


def build():
    rows = {r["id"]: r for r in map(json.loads, (ROOT / "data/clauses.jsonl").read_bytes().splitlines())}
    transfer = ROOT / "data/external_clauses.jsonl"
    if transfer.is_file():
        rows.update((r["id"], r) for r in map(json.loads, transfer.read_bytes().splitlines()))
    draft = TABLE_DB.with_suffix(".tmp")
    if draft.exists():
        draft.unlink()
    with closing(sqlite3.connect(draft)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        imaging_rows(db, rows)
        inventories = {r["source"]: r for r in rows.values() if r.get("kind") == "table_inventory"}
        for protocol, row in inventories.items():
            put_source(db, row)
            db.execute("INSERT INTO oar_inventory VALUES (?,?)", (protocol, row["id"]))
        for row in rows.values():
            if (row["source"] not in IMAGING or row["section"] != "Organs at risk"
                    or row.get("kind") == "table_inventory"):
                continue
            cells = [v.strip() for v in row["text"].splitlines()[-1].split("|")]
            organ, note = cells[0].lower().rstrip("*"), cells[-1]
            if organ == "spinal canal":
                organ = "spinal cord"
            if len(cells) >= 3 and organ in ORGANS and row["source"] in inventories:
                put_source(db, row)
                db.execute("INSERT INTO oar_inventory_item VALUES (?,?,?,?)",
                           (row["source"], organ, row["id"], int(bool(note))))
            fields = contour_fields(organ, note) if len(cells) >= 3 and organ in ORGANS else None
            if fields:
                put_source(db, row)
                db.execute("INSERT INTO contour VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (row["id"], organ, *fields, note))
        print("Verification rows:", db.execute("SELECT COUNT(*) FROM verification").fetchone()[0])
        print("Contour rows:", db.execute("SELECT COUNT(*) FROM contour").fetchone()[0])
        publisher_file = ROOT / "data/publisher_tables.json"
        if publisher_file.is_file():
            from publisher_tables import install
            install(db, publisher_file)
    draft.replace(TABLE_DB)


if __name__ == "__main__":
    build()

