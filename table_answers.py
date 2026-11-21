"""Answer matching table questions with structured fields and source row citations."""
from contextlib import closing
import re
import sqlite3
import time

from build_protocol_tables import METHOD, ORGANS, TABLE_DB
from citation_guard import structured_chat
from relevance import CYBERKNIFE

VERIFY = re.compile(r"靶区验证|影像验证|验证表|影像配准|配准表|影像表|匹配点|配准参照|成像(?:方式|频率)|影像频率")
MODALITIES = {
    "kv_planar": r"\bkv\s*(?:平片|平面|planar)",
    "mv_planar": r"\bmv\s*(?:平片|平面|planar)",
    "cbct_3d": r"3\s*d\s*(?:(?:或|/|和|与)\s*4\s*d\s*)?(?:cbct|锥形束)",
    "cbct_4d": r"4\s*d\s*(?:cbct|锥形束)",
}
ANATOMY = {
    "chest wall": r"胸壁",
    "spinal cord": r"脊髓|椎管",
    "heart": r"心脏|心包",
    "oesophagus": r"食管|食道",
    "stomach": r"胃(?:的|壁|内容物|勾画)|幽门",
    "skin": r"皮肤",
    "lung": r"全肺|双肺|剩余肺|肺轮廓|肺的勾画|肺勾画",
    "great vessels": r"大血管",
    "trachea/proximal bronchi": r"近端(?:气管|支气管)",
}
CONTOUR = re.compile(r"勾画|轮廓|边界|上界|下界|(?:向内|上方|下方|头侧|尾侧).{0,12}(?:厚|延伸)|外表面")
MATCH = re.compile(r"配准|匹配点|对骨|对位|软组织|定位标记|GTV", re.I)
FREQUENCY = re.compile(r"频率|多久|每隔|每日|每天|何时")
PRE = re.compile(r"治疗前|照射前")
DURING = re.compile(r"治疗(?:过程)?中|分次内|照射中")
POST = re.compile(r"治疗(?:结束|完成)?后|照射后")
MODE = re.compile(r"几种维度|哪.{0,8}成像|成像方式|什么(?:成像|影像)")
LAYOUT_SCHEMA = {"type": "object", "required": ["style"], "properties": {
    "style": {"type": "string", "enum": ["compact", "list"]}
}}


def route_table(question):
    text = re.sub(r"(?:不问|不要回答|不用列)[^。；，？！]*", "", question)
    if CYBERKNIFE.search(text):
        return None
    organs = [key for key, pattern in ANATOMY.items() if re.search(pattern, text)]
    if organs and CONTOUR.search(text):
        return "contour", text, organs
    if VERIFY.search(text) or ("CBCT" in text.upper() and MATCH.search(text)):
        return "verification", text, []
    return None


def match_text(db, row):
    basis = row["match_basis"]
    if basis == "bone":
        return "骨性结构"
    if basis == "soft_and_or_marker":
        return "软组织和/或定位标记物；定位标记物并非必选"
    if basis == "bone_check_soft":
        return "对骨；使用 CBCT 时还要检查软组织对齐，表未指定 GTV 或 iGTV 为配准参照"
    if basis == "bone_or_soft_by_ct":
        targets = list(db.execute("""SELECT planning_ct_dimension,soft_target
            FROM cbct_reference WHERE source_id=? ORDER BY planning_ct_dimension""", (row["source_id"],)))
        conditions = "、".join(f"{r['planning_ct_dimension']}D CT 对应 {r['soft_target']}" for r in targets)
        value = f"按科室协议对骨或对软组织；软组织参照按计划 CT 条件区分：{conditions}"
    else:
        value = f"按科室协议对骨或对软组织；软组织参照为 {row['soft_target']}"
    if row["oar_evaluation"]:
        value += "；并评估危及器官"
    return value


def frequency_text(db, row):
    interval = row["interval_days"]
    if interval is not None:
        return "每日" if interval == 1 else f"每 {interval} 天"
    phases = db.execute("SELECT * FROM verification_phase WHERE source_id=? ORDER BY ordinal",
                        (row["source_id"],))
    labels = []
    for part in phases:
        if not part["after_initial"]:
            labels.append(f"第 {part['first_day']}–{part['last_day']} 天")
        else:
            days = part["maximum_interval_days"]
            labels.append("此后至少每周一次" if days == 7 else f"此后每 {days} 天至少一次")
    return "，".join(labels)


def source_of(row):
    return dict(id=row["source_id"], source=row["protocol"], version=row["version"],
                section=row["section"], url=row["url"], text=row["original"])


def select_verification(db, scope, question):
    sql = """SELECT * FROM verification JOIN source_row ON source_row.id=verification.source_id
             WHERE protocol=? ORDER BY CASE modality WHEN 'kv_planar' THEN 0
                                      WHEN 'mv_planar' THEN 1 WHEN 'cbct_3d' THEN 2 ELSE 3 END"""
    explicit = {key for key, pattern in MODALITIES.items() if re.search(pattern, question, re.I)}
    generic = not (explicit & {"cbct_3d", "cbct_4d"}) and bool(re.search(r"CBCT|锥形束", question, re.I))
    show_match = bool(MATCH.search(question))
    show_during = bool(DURING.search(question))
    show_after = bool(POST.search(question))
    show_daily = bool(PRE.search(question) or (FREQUENCY.search(question) and not show_after
                                               and not show_during))
    default = not (show_match or show_during or show_after or show_daily or MODE.search(question))
    show_match = show_match or default
    show_daily = show_daily or default
    groups, seen_sources = [], {}
    for protocol in scope:
        rows = list(db.execute(sql, (protocol,)))
        if not rows:
            groups.append(dict(protocol=protocol, subject="影像验证", facts=[["范围","未收录该协议的表"]]))
            continue
        available = {r["modality"] for r in rows}
        requested = explicit | (available & {"cbct_generic", "cbct_3d", "cbct_4d"} if generic else set())
        seen_sources[rows[0]["source_id"]] = source_of(rows[0])
        if show_after:
            groups.append(dict(protocol=protocol, subject="治疗结束后",
                               source_id=rows[0]["source_id"],
                               facts=[["频率", "这张表未列治疗结束后的影像频率"]]))
            if not (show_match or show_daily or show_during):
                continue
        by_fields = {}
        for row in rows:
            if requested and row["modality"] not in requested:
                continue
            facts = []
            if show_daily:
                facts.append(("频率", frequency_text(db, row)))
                if row["stage"] == "pre_treatment":
                    facts.append(("时点", "治疗前"))
                elif PRE.search(question):
                    facts.append(("时点", "该表未明确写成治疗前"))
            if show_during:
                when = ("可考虑分次治疗中成像；不是必须" if row["during_optional"]
                        else "该表未列分次治疗中成像")
                facts.append(("治疗中", when))
            if show_match:
                facts.append(("配准对象", match_text(db, row)))
            by_fields.setdefault(tuple(facts), []).append(METHOD[row["modality"]])
        for facts, labels in by_fields.items():
            groups.append(dict(protocol=protocol, subject=" 或 ".join(labels),
                               source_id=rows[0]["source_id"],
                               facts=[["成像方式", " 或 ".join(labels)], *[list(f) for f in facts]]))
        missing = explicit - available
        if missing:
            names = "、".join(METHOD[k] for k in METHOD if k in missing)
            groups.append(dict(protocol=protocol, subject="影像验证", source_id=rows[0]["source_id"],
                               facts=[["成像方式", f"该表未列 {names}"]]))
            if "cbct_generic" in available and missing & {"cbct_3d", "cbct_4d"}:
                generic_row = next(r for r in rows if r["modality"] == "cbct_generic")
                facts = [["成像方式", "该表仅统称为 CBCT，未区分 3D 或 4D CBCT 模态"]]
                if show_match:
                    facts.append(["配准对象", match_text(db, generic_row)])
                groups.append(dict(protocol=protocol, subject="表内 CBCT",
                                   source_id=generic_row["source_id"], facts=facts))
    return groups, list(seen_sources.values())


def select_contours(db, scope, organs):
    sql = """SELECT * FROM contour JOIN source_row ON source_row.id=contour.source_id
             WHERE protocol=? AND organ=?"""
    groups, sources = [], []
    for protocol in scope:
        for organ in organs:
            row = db.execute(sql, (protocol, organ)).fetchone()
            if row is None:
                inventory = db.execute(
                    """SELECT s.*, i.source_id FROM oar_inventory AS i
                       JOIN source_row AS s ON s.id=i.source_id WHERE i.protocol=?""",
                    (protocol,)).fetchone()
                record = db.execute(
                    """SELECT s.*, i.source_id, i.has_note FROM oar_inventory_item AS i
                       JOIN source_row AS s ON s.id=i.source_id WHERE i.protocol=? AND i.organ=?""",
                    (protocol, organ)).fetchone()
                if record is not None:
                    display = ("该行未列勾画备注" if not record["has_note"]
                               else "该行没有可直接读取的结构化勾画字段；请核对原表备注")
                    evidence = record
                elif inventory is not None:
                    display = "该危及器官表未列此器官的勾画项"
                    evidence = inventory
                else:
                    display = "当前结构化库未收录该协议的此器官勾画字段"
                    evidence = None
                group = dict(protocol=protocol, subject=ORGANS[organ],
                             facts=[["勾画备注", display]])
                if evidence is not None:
                    group["source_id"] = evidence["source_id"]
                    sources.append(source_of(evidence))
                groups.append(group)
                continue
            facts = []
            if row["boundary"]:
                facts.append(["解剖边界", row["boundary"]])
            if row["components"]:
                facts.append(["涵盖组织", row["components"]])
            if row["description"]:
                facts.append(["勾画说明", row["description"]])
            if row["upper_min"] is not None:
                facts.append(["上方延伸", f"{row['reference']} 上方 ≥{row['upper_min']:g} {row['unit']}"])
            if row["lower_min"] is not None:
                facts.append(["下方延伸", f"{row['reference']} 下方 ≥{row['lower_min']:g} {row['unit']}"])
            sources.append(source_of(row))
            groups.append(dict(protocol=protocol, subject=ORGANS[organ],
                               source_id=row["source_id"], facts=facts))
    if len(scope) > 1 and len(organs) == 1 and len(sources) == len(scope):
        first = groups[0]["facts"]
        if all(g["facts"] == first for g in groups):
            groups.append(dict(protocol=" / ".join(scope), subject="比较",
                               source_ids=[g["source_id"] for g in groups],
                               facts=[["结果", "两份表的这些勾画字段一致"]]))
    return groups, list({r["id"]: r for r in sources}.values())


def format_answer(base, question, groups, sources):
    layout = structured_chat(
        base, "你负责中文放疗协议表格说明的排版。只选 compact（同一对象一行）或 list（逐字段列出）。"
              "不要生成或修改任何字段值，多协议比较通常选 list。",
        {"question": question, "groups": [
            {"protocol": g["protocol"], "subject": g["subject"],
             "field_labels": [field for field, _ in g["facts"]]} for g in groups]},
        LAYOUT_SCHEMA, limit=48,
    )
    style = layout.get("style") if isinstance(layout, dict) else None
    if not isinstance(style, str) or style not in {"list", "compact"}:
        style = "list"
    out = ["原文"]
    out += [f"[{r['id']}] {r['version']} / {r['section']}\n{r['text']}\n来源：{r['url']}"
            + (f"\n跨页表头：{r['header_url']}" if r.get("header_url") else "")
            for r in sources]
    out.append("中文说明（按表格字段）")
    for group in groups:
        ids = group.get("source_ids", [group["source_id"]] if "source_id" in group else [])
        publisher = group.get("display_name", f"eviQ {group['protocol']}")
        title = " ".join([f"{publisher} · {group['subject']}", *[f"[{k}]" for k in ids]]).strip()
        if style == "compact":
            summary = "；".join(f"{k}：{v}" for k, v in group["facts"])
            out.append(f"{title}\n{summary}。")
        else:
            summary = "\n".join(f"- {k}：{v}" for k, v in group["facts"])
            out.append(f"{title}\n{summary}")
    return "\n\n".join(out), style


def structured_table_answer(base, question, scope):
    planned = route_table(question)
    if planned is None:
        return None
    start = time.monotonic()
    route, text, organs = planned
    if not scope:
        return dict(route=route, status="needs_protocol", groups=[], highlights=[],
                    answer="请明确要核对哪一份 eviQ 协议的表格。", seconds=0.0)
    drivers = sorted(scope)
    with closing(sqlite3.connect(f"{TABLE_DB.as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        if route == "verification":
            groups, sources = select_verification(db, drivers, text)
        else:
            groups, sources = select_contours(db, drivers, organs)
    sql_seconds = time.monotonic() - start
    if not sources:
        return dict(route=route, status="not_indexed", groups=groups, highlights=[],
                    answer="当前结构化数据库没有收录所问协议的对应表格字段。",
                    seconds=round(sql_seconds, 3))
    answer, style = format_answer(base, question, groups, sources)
    return dict(route=route, status="answered", groups=groups, highlights=sources,
                answer=answer, layout=style, sql_seconds=round(sql_seconds, 4),
                seconds=round(time.monotonic() - start, 3))

