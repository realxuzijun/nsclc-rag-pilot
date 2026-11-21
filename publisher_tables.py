"""Install page-linked PDF facts in the shared SQLite and answer scoped queries."""
from contextlib import closing
import json
import re
import sqlite3
import time

from build_protocol_tables import TABLE_DB

PUBLISHER_IDS = {
    "wmrn": r"\bwmrn\b|west\s+midlands|西米德兰",
    "wfb": r"wake\s+forest|\bwfb(?:ccc)?\b|\b62718\b|nct03867175|维克森林",
}
NAMES = {"wmrn": "West Midlands", "wfb": "Wake Forest 62718"}
ORGAN_TERMS = {
    "spinal cord": r"椎管|脊髓|spinalcanal|spinal cord",
    "heart": r"心脏|心包|heart",
    "brachial plexus": r"臂丛|brachial\s*plex",
    "oesophagus": r"食管|食道|oesophag|esophag",
    "chest wall": r"胸壁|chest\s*wall",
    "lung": r"双肺|肺.{0,4}(?:v20|平均剂量)|lungs?",
    "skin": r"皮肤|skin",
    "kidney": r"肾脏?|kidney",
    "liver": r"肝脏?|liver",
}
METRIC = re.compile(r"dmax\s*\(\s*\d+(?:\.\d+)?\s*cc\s*\)|[dv]\s*\d+(?:\.\d+)?\s*(?:cc|gy)?", re.I)
TIER = {"optimal": "最优", "mandatory": "强制", "recommended": "建议"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS pdf_source (
  source_id TEXT PRIMARY KEY REFERENCES source_row(id), page INTEGER NOT NULL,
  publisher TEXT NOT NULL, header_url TEXT
);
CREATE TABLE IF NOT EXISTS pdf_contour_extra (
  source_id TEXT PRIMARY KEY REFERENCES contour(source_id), radial TEXT, excluded TEXT
);
CREATE TABLE IF NOT EXISTS pdf_dose_cell (
  id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES source_row(id),
  organ TEXT NOT NULL, organ_en TEXT NOT NULL, metric TEXT NOT NULL, metric_en TEXT NOT NULL,
  fraction_count INTEGER, tier TEXT NOT NULL, regimen_condition TEXT NOT NULL,
  denominator TEXT, comparator TEXT, value REAL, unit TEXT, raw_cell TEXT NOT NULL,
  listed INTEGER NOT NULL, footnote TEXT, note_source_id TEXT REFERENCES source_row(id)
);
CREATE INDEX IF NOT EXISTS pdf_dose_lookup ON pdf_dose_cell(organ,metric,fraction_count,tier);
"""


def install(db, file):
    if not file.is_file():
        return
    db.executescript(SCHEMA)
    data = json.loads(file.read_bytes())
    for r in data["rows"]:
        db.execute("INSERT INTO source_row VALUES (?,?,?,?,?,?)", tuple(
            r[k] for k in ("id", "protocol", "version", "section", "url", "text")))
        db.execute("INSERT INTO pdf_source VALUES (?,?,?,?)",
                   (r["id"], r["page"], r["publisher"], r.get("header_url")))
    raw = {r["id"]: r["text"] for r in data["rows"]}
    for r in data["contours"]:
        db.execute("INSERT INTO contour VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (r["source_id"], r["organ"], r.get("boundary"), r.get("components"),
                    r.get("description"), None, None, None, None, raw[r["source_id"]]))
        db.execute("INSERT INTO pdf_contour_extra VALUES (?,?,?)",
                   (r["source_id"], r.get("radial"), r.get("excluded")))
    keys = ("id", "source_id", "organ", "organ_en", "metric", "metric_en", "fractions", "tier",
            "condition", "denominator", "comparator", "value", "unit", "raw", "listed",
            "footnote", "note_source_id")
    for r in data["cells"]:
        db.execute("INSERT INTO pdf_dose_cell VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   tuple(r[k] for k in keys))
    print("PDF dose cells:", len(data["cells"]), "PDF contour rows:", len(data["contours"]))


def publisher_scope(question):
    return {source for source, pattern in PUBLISHER_IDS.items() if re.search(pattern, question, re.I)}


def source(db, key):
    r = db.execute("""SELECT s.*,x.page,x.publisher,x.header_url
                      FROM source_row s JOIN pdf_source x ON x.source_id=s.id WHERE id=?""",
                   (key,)).fetchone()
    return dict(id=r["id"], source=r["protocol"], version=r["version"], section=r["section"],
                url=r["url"], text=r["original"], page=r["page"], publisher=r["publisher"],
                header_url=r["header_url"])


def dosage_groups(db, code, question, organs):
    fractions = {int(n) for n in re.findall(r"(\d+)\s*(?:次|fractions?\b)", question, re.I)}
    tiers = {k for k, v in TIER.items() if v in question}
    metrics = {re.sub(r"\s+|gy$", "", token.lower()) for token in METRIC.findall(question)}
    if re.search(r"平均剂量|\bmean\b", question, re.I):
        metrics.add("mean")
    if re.search(r"最大剂量|点最大剂量|最大点剂量", question):
        metrics.add("max")
    fallback = "wmrn-p13-dose-header" if code == "wmrn" else "wfb-p22-oar-inventory"
    if code == "wfb" and fractions:
        return [dict(protocol=code, display_name=NAMES[code], subject="§7.6 每次 <3 Gy 表",
                     source_id=fallback, facts=[["总分次数",
                         "该表未按总分次数分列；同页独立的 SBRT 表不属于本次所问的常规表"]])]
    rows = db.execute("""SELECT d.* FROM pdf_dose_cell d JOIN source_row s ON s.id=d.source_id
        WHERE s.protocol=? ORDER BY organ,metric,fraction_count,
        CASE tier WHEN 'optimal' THEN 0 WHEN 'mandatory' THEN 1 ELSE 2 END""", (code,))
    groups = []
    for r in rows:
        if organs and r["organ"] not in organs:
            continue
        if metrics and r["metric"] not in metrics:
            continue
        if fractions and r["fraction_count"] not in fractions:
            continue
        if tiers and r["tier"] not in tiers:
            continue
        facts = [["适用条件", r["regimen_condition"]], ["栏目", TIER[r["tier"]]],
                 ["原格", r["raw_cell"] if r["listed"] else "“–”，未列限值"]]
        if r["denominator"]:
            facts.append(["计算体积", r["denominator"]])
        if r["listed"] and r["unit"] is None:
            facts.append(["单位", "原格未明写单位"])
        if r["listed"] and r["comparator"] is None:
            facts.append(["比较符", "原格未写比较符"])
        ids = [r["source_id"]]
        if r["note_source_id"]:
            ids.append(r["note_source_id"])
        groups.append(dict(protocol=code, display_name=NAMES[code],
                           subject=f"{r['organ_en']} · {r['metric_en']}",
                           source_ids=ids, facts=facts))
    if groups:
        return groups
    display = ("该常规分割表未列所问器官或指标" if code == "wfb"
               else "已收录的 §8.2 表没有与问题完全匹配的器官、指标及分割栏")
    return [dict(protocol=code, display_name=NAMES[code], subject="表内查找",
                 source_id=fallback, facts=[["结果", display]])]


def contour_groups(db, code, organs):
    groups = []
    for organ in organs:
        row = db.execute("""SELECT c.*,e.radial,e.excluded FROM contour c
            JOIN source_row s ON s.id=c.source_id JOIN pdf_contour_extra e ON e.source_id=c.source_id
            WHERE s.protocol=? AND c.organ=?""", (code, organ)).fetchone()
        if row is None:
            return []
        facts = [[label, row[key]] for key, label in (
            ("radial", "厚度和位置"), ("boundary", "解剖边界"), ("components", "涵盖组织"),
            ("excluded", "排除结构"), ("description", "勾画说明")) if row[key]]
        groups.append(dict(protocol=code, display_name=NAMES[code], subject="勾画",
                           source_id=row["source_id"], facts=facts))
    return groups


def publisher_answer(base, question):
    scope = publisher_scope(question)
    if not scope:
        return None
    from table_answers import format_answer

    start = time.monotonic()
    organs = {k for k, pattern in ORGAN_TERMS.items() if re.search(pattern, question, re.I)}
    is_contour = bool(organs and re.search(r"勾画|环形层|纵向起止", question))
    is_policy = bool(re.search(r"入组|必需条件", question))
    groups = []
    with closing(sqlite3.connect(f"{TABLE_DB.as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pdf_source'").fetchone():
            return dict(route="pdf", status="not_indexed", groups=[], highlights=[],
                        answer="当前数据库未构建所问的 PDF 协议表。", seconds=0.0)
        for code in sorted(scope):
            if code == "wfb" and is_policy:
                groups.append(dict(protocol=code, display_name=NAMES[code], subject="§7.6 使用条件",
                                   source_id="wfb-p21-oar-policy",
                                   facts=[["约束性质", "强烈推荐遵守；并非入组必须条件"]]))
            elif is_contour:
                groups.extend(contour_groups(db, code, sorted(organs)))
            elif organs or re.search(r"约束|限值|[Gg][Yy]|分割", question):
                groups.extend(dosage_groups(db, code, question, organs))
        keys = {k for g in groups for k in g.get("source_ids", [g.get("source_id")]) if k}
        source_list = [source(db, key) for key in sorted(keys)]
    if not groups:
        return dict(route="pdf", status="not_indexed", groups=[], highlights=[],
                    answer="没有已结构化的本协议对应字段。请核对原 PDF。", seconds=round(time.monotonic()-start, 3))
    answer, layout = format_answer(base, question, groups, source_list)
    return dict(route="pdf", status="answered", groups=groups, highlights=source_list,
                answer=answer, layout=layout, seconds=round(time.monotonic()-start, 3))

