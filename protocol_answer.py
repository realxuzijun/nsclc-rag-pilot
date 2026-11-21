"""Look up protocol facts; show source clauses before Chinese explanations."""
import argparse
from contextlib import closing
import json
import re
import sqlite3
import time

from benchmark import ROOT, retrieve, vectors
from citation_guard import protocol_scope, quoted_in, structured_chat
from relevance import CYBERKNIFE, DEFAULT_RERANK, choose_relevant, compares_devices
from round3 import dose_query, embed_clauses
from table_answers import structured_table_answer
from publisher_tables import publisher_answer

MAX_PASSAGES = 4
EXPLANATIONS = {"type": "object", "required": ["items"], "properties": {
    "items": {"type": "array", "maxItems": MAX_PASSAGES, "items": {
        "type": "object", "required": ["id", "zh"], "properties": {
            "id": {"type": "string"}, "zh": {"type": "string"}
        }
    }}
}}
# This detects missing surface markers. It cannot certify a translation.
MARKERS = (
    (r"\b(?:above|superior to)\b", r"上方|以上|上缘之外|上侧", "upper_boundary"),
    (r"\b(?:below|inferior to)\b", r"下方|以下|下缘之外|下侧", "lower_boundary"),
    (r"\bunless\b", r"除非|除外|例外|有禁忌|无禁忌|没有禁忌", "exception"),
    (r"\bdoes not apply\b", r"不适用|不适合|不包括|排除", "scope_exclusion"),
    (r"\bnot recommended\b", r"不推荐|不建议|不主张", "negative_recommendation"),
    (r"\bwithout (?:a )?demonstrated\b",
     r"(?:未|不曾|没有).{0,15}(?:证实|证明|显示|表明)|(?:尚无|没有).{0,15}证据",
     "no_demonstrated_benefit"),
    (r"\bno margin\b", r"不.{0,12}(?:边界|外扩|余量|边缘)|无.{0,12}(?:边界|外扩|余量|边缘)", "no_margin"),
    (r"\bnot progressed\b", r"(?:未|没有|无|不曾)(?:发生|出现|观察到|见)?(?:疾病)?进展",
     "no_progression"),
    (r"\boptional\b", r"可选|可选择|酌情|非强制|不是必须|并非必须", "optionality"),
    (r"\bideally\b", r"理想|尽量|最好|优先", "preference"),
    (r"\bdepending on\b", r"取决|根据|视.{0,15}而定|视|依据", "dependency"),
    (r"\bshould generally be limited\b", r"通常|一般|原则上", "generally_limited"),
)


def marker_issues(source, explanation, enhanced=False):
    marks = MARKERS
    if enhanced:
        marks = (
            (r"\b(?:above(?!\s+(?:the\s+)?head\b)|superior to)\b",
             r"上方|以上|上缘之外|上侧", "upper_boundary"),
            (r"\babove\s+(?:the\s+)?head\b",
             r"头(?:部|顶)?(?:的)?上方|头顶以上|过头(?:顶)?|上举至头(?:部|顶)上方",
             "above_head_position"),
        ) + MARKERS[1:]
    return [label for en, zh, label in marks
            if re.search(en, source, re.I) and not re.search(zh, explanation)]


def explanation_issues(source, zh, enhanced):
    if not zh:
        return ["no_explanation"]
    issues = marker_issues(source["text"], zh, enhanced)
    if enhanced and source.get("device_role") == "常规患者" and CYBERKNIFE.search(zh):
        issues.append("general_evidence_cannot_describe_named_device")
    return issues


SELECT_SYSTEM = (
    "你在为肺癌放疗协议问答寻找原文。只返回提供的候选条款 ID；不要写答案、翻译或引用文字。"
    "选择能直接回答问题的最少条款。问题的每个子问都要有依据，"
    "包括例外、适用条件、否定词、方位、比较符以及推荐强度。"
    "只能选题面所问协议的证据。相同主题但不同时间点或情形不算相关。"
    "没有直接证据时返回空数组。"
)
EXPLAIN_SYSTEM = (
    "将选定的原文按 ID 各写一段简明中文说明，直接解释问题需要的内容。"
    "以对应原文为唯一依据；同一 ID 只能解释该条款。不要复述题目或添加临床建议。"
    "保留条件限定和例外，以及原文的否定和推荐强度。"
    "解剖边界必须把相对方向放在地标之后，例如“从某地标上方到另一地标下方”，"
    "避免写成从地标本身开始或结束。保留括号里的覆盖范围。"
    "与问题有关的不等符号、数值及其单位必须原样保留。"
)
EXPLAIN_ALL = EXPLAIN_SYSTEM + (
    "逐一回答问题要求的子问；对同一条款中的多个限定条件、原因或比较对象分别写明。"
    "问‘为什么’时，复述原文具体原因，不能只重复结论或证据强弱。"
)
COVERAGE_CUES = re.compile(r"为什么|为何|原因|分别|各自|各(?:用|至少)|哪两|还说|还要|还哪些|两者")


def coverage_gaps(base, question, highlights):
    sources = {r["id"]: r for r in highlights}
    schema = {"type": "object", "required": ["gaps"], "properties": {
        "gaps": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "required": ["id", "missing", "quote"], "properties": {
                "id": {"type": "string", "enum": list(sources)},
                "missing": {"type": "string"}, "quote": {"type": "string"}
            }
        }}
    }}
    answer = structured_chat(
        base,
        "检查中文解释是否漏掉题面明确问到的子问。只报告能在对应英文原文中找到的漏答，"
        "每条包含条款 ID、缺漏要点和可连续复制的英文原句。"
        "特别区分原文给出的具体原因和单纯重复结论。不要要求回答无关背景或原文没有的内容。"
        "整组中文解释已覆盖的要点不得重复报漏；如无漏答返回空数组。",
        {"question": question, "items": [
            {"id": r["id"], "source_text": r["text"], "zh": r["explanation"]} for r in highlights]},
        schema, limit=512,
    )
    gaps = answer.get("gaps") if isinstance(answer, dict) else None
    if not isinstance(gaps, list):
        return None
    verified = []
    for g in gaps:
        if (isinstance(g, dict) and isinstance(g.get("id"), str)
                and g["id"] in sources and isinstance(g.get("missing"), str)
                and g["missing"].strip() and isinstance(g.get("quote"), str)
                and quoted_in(g["quote"].casefold(), sources[g["id"]]["text"].casefold())):
            verified.append(g)
    return verified if verified or not gaps else None


def repair_coverage(base, question, highlights, audit):
    gaps = coverage_gaps(base, question, highlights)
    if gaps is None:
        audit.append(dict(stage="coverage", issue="unverified"))
        return False
    if not gaps:
        audit.append(dict(stage="coverage", initial_gaps=[], remaining_gaps=[]))
        return True
    affected = {g["id"] for g in gaps}
    repair = structured_chat(
        base, EXPLAIN_ALL,
        {"question": question, "sources": [
            {"id": r["id"], "source_text": r["text"], "current_zh": r["explanation"],
             "device_role": r.get("device_role"),
             "supported_missing_points": [g for g in gaps if g["id"] == r["id"]]}
            for r in highlights if r["id"] in affected]},
        EXPLANATIONS, limit=1024,
    )
    items = repair.get("items", []) if isinstance(repair, dict) else []
    patches = {x["id"]: x["zh"].strip() for x in (items if isinstance(items, list) else [])
               if isinstance(x, dict)
               and isinstance(x.get("id"), str) and isinstance(x.get("zh"), str)}
    amended = []
    for source in highlights:
        zh = patches.get(source["id"])
        if (source["id"] in affected and zh and not zh.endswith(("?", "？"))
                and not explanation_issues(source, zh, enhanced=True)):
            source["explanation"] = zh
            amended.append(source["id"])
    remaining = coverage_gaps(base, question, highlights) if amended else gaps
    audit.append(dict(stage="coverage", initial_gaps=gaps, amended_ids=amended,
                      remaining_gaps=remaining))
    return remaining == []


def explain_passages(base, question, evidence, scope, preselected=False, enhanced=False):
    start = time.monotonic()
    allowed = {r["id"]: r for r in evidence if scope is None or r["source"] in scope}
    audit = []
    sources = list(allowed.values())[:MAX_PASSAGES] if preselected else []
    if allowed and not preselected:
        payload = {"question": question, "passages": list(allowed.values())}
        schema = {"type": "object", "required": ["ids"], "properties": {
            "ids": {"type": "array", "maxItems": MAX_PASSAGES,
                    "items": {"type": "string", "enum": list(allowed)}}
        }}
        result = structured_chat(base, SELECT_SYSTEM, payload, schema, limit=256)
        ids = result.get("ids") if isinstance(result, dict) else None
        accepted = set()
        for key in (ids if isinstance(ids, list) else [])[:MAX_PASSAGES]:
            if not isinstance(key, str) or key not in allowed:
                audit.append({"stage": "select", "id": key, "issue": "id_not_in_allowed_candidates"})
            elif key not in accepted:
                sources.append(allowed[key])
                accepted.add(key)

    if sources:
        payload = {"question": question, "sources": [
            dict(id=r["id"], section=r["section"], source_text=r["text"],
                 **({"device_role": r["device_role"]} if "device_role" in r else {}))
            for r in sources]}
        explanation_prompt = EXPLAIN_ALL if enhanced else EXPLAIN_SYSTEM
        if enhanced and any("device_role" in r for r in sources):
            explanation_prompt += ("逐条说明 device_role 指定的对象。常规患者条款只写常规患者，"
                                   "不要提及或推断 CyberKnife 的做法，专用设备另有条款。")
        translated = structured_chat(base, explanation_prompt, payload, EXPLANATIONS)
        output = translated.get("items", []) if isinstance(translated, dict) else []
        translations = {}
        selected_ids = {source["id"] for source in sources}
        for item in output if isinstance(output, list) else []:
            if not isinstance(item, dict):
                continue
            key, zh = item.get("id"), item.get("zh")
            if not isinstance(key, str) or key not in selected_ids or not isinstance(zh, str):
                audit.append({"stage": "explain", "id": key, "issue": "invalid_explanation_reference"})
                continue
            zh = zh.strip()
            if zh and not zh.endswith(("?", "？")):
                translations.setdefault(key, zh)
        rendered = []
        for source in sources:
            key = source["id"]
            zh = translations.get(key)
            issues = explanation_issues(source, zh, enhanced)
            if issues:
                first_zh = zh
                retry = structured_chat(
                    base, explanation_prompt,
                    {"question": question,
                     "sources": [dict(id=key, source_text=source["text"],
                                      **({"device_role": source["device_role"]}
                                         if "device_role" in source else {}))],
                     "missing_markers_to_check": issues}, EXPLANATIONS, limit=512,
                )
                items = retry.get("items") if isinstance(retry, dict) else None
                if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
                    candidate = items[0].get("zh")
                    if (items[0].get("id") == key and isinstance(candidate, str)
                            and candidate.strip() and not candidate.strip().endswith(("?", "？"))
                            and not explanation_issues(source, candidate, enhanced)):
                        zh = candidate.strip()
                pending = explanation_issues(source, zh, enhanced)
                audit.append({"stage": "explain", "id": key, "initial_issues": issues,
                              "remaining_issues": pending, "initial_explanation": first_zh,
                              "retry_output": retry})
                if pending:
                    zh = None
            source_record = dict(source, explanation=zh)
            rendered.append(source_record)

        highlights = rendered
        coverage_ok = True
        if enhanced and (COVERAGE_CUES.search(question)
                         or len(re.findall(r"[？?]", question)) >= 2):
            coverage_ok = repair_coverage(base, question, highlights, audit)
        if enhanced and compares_devices(question):
            if {r.get("device_role") for r in highlights} != {"CyberKnife 专用", "常规患者"}:
                coverage_ok = False
                audit.append(dict(stage="coverage", issue="missing_device_comparison_side"))
        original = ["原文"]
        chinese = ["中文解释"]
        for source in highlights:
            role = f"（{source['device_role']}）" if "device_role" in source else ""
            original.append(f"[{source['id']}] {source['version']} / {source['section']}{role}\n"
                            f"{source['text']}\n来源：{source['url']}")
            text = source["explanation"] or "中文解释未通过限定词检查，请核对上方原文。"
            chinese.append(f"[{source['id']}]{role} {text}")
        if not coverage_ok:
            chinese.append("部分子问尚未核对完整，请对照上方原文。")
        answer = "\n\n".join(original + chinese)
        status = ("answered" if coverage_ok and all(s["explanation"] for s in highlights)
                  else "source_only_or_partial")
    else:
        highlights = []
        answer = "未找到能直接回答问题的对应条款。"
        status = "no_direct_evidence"
    return dict(answer=answer, status=status, highlights=highlights, audit=audit,
                seconds=round(time.monotonic() - start, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--terms", default="")
    parser.add_argument("--base", default="http://127.0.0.1:11543")
    parser.add_argument("--rerank-base", default=DEFAULT_RERANK)
    args = parser.parse_args()
    pdf = publisher_answer(args.base, args.question)
    if pdf is not None:
        print(pdf["answer"])
        return
    scope = protocol_scope(args.question)
    table_answer = structured_table_answer(args.base, args.question, scope)
    if table_answer is not None:
        print(table_answer["answer"])
        return
    data = ROOT / "data"
    db_file = data / "dose_tables.sqlite"
    with closing(sqlite3.connect(f"{db_file.as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        dose = dose_query(args.question, scope, db)
    if dose is not None:
        print(dose["answer"])
        return
    corpus_paths = [data / "clauses.jsonl", data / "external_clauses.jsonl"]
    clauses = [c for path in corpus_paths if path.is_file()
               for c in map(json.loads, path.read_bytes().splitlines())
               if scope is None or c["source"] in scope]
    query = vectors(args.base, [args.question])[0]
    docs = [f"{r['section']}\n{r['text']}" for r in clauses]
    index = embed_clauses(args.base, docs, [])
    evidence = retrieve(clauses, docs, {"search_terms": args.terms}, query, index,
                        allowed_sources=scope) if clauses else []
    selected, _ = choose_relevant(args.base, args.rerank_base, args.question, evidence, scope,
                                  device_guard=True)
    print(explain_passages(args.base, args.question, selected, scope, preselected=True,
                           enhanced=True)["answer"])


if __name__ == "__main__":
    main()

