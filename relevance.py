"""Rerank evidence with a separate cross-encoder, then check direct relevance."""
import json
import math
import re
import time
from urllib.request import Request, urlopen

from citation_guard import quoted_in, structured_chat

DEFAULT_RERANK = "http://127.0.0.1:11544"
SECTIONS = {
    "Simulation": r"模拟定位(?:章节|说明|表)",
    "Indications and patient population": r"适应[证症](?:章节|部分|说明|条款)",
    "Clinical information": r"临床信息",
    "Treatment delivery": r"治疗实施",
    "Treatment planning": r"计划章节|计划设计章节",
}
RESEARCH_SECTION = re.compile(r"证据章节|evidence section", re.I)
RESEARCH_QUERY = re.compile(r"试验(?:的)?(?:入组|纳入|随机分组)|(?:研究|试验)结果|PACIFIC", re.I)
CYBERKNIFE = re.compile(r"cyber\s*knife|射波刀", re.I)
DEVICE_COMPARISON = re.compile(
    r"对比|比较|相比|分别|与(?:常规|普通|一般)|和(?:常规|普通|一般)", re.I)
GENERAL_DEVICE = re.compile(r"(?:常规|普通|一般)(?:患者|影像|设备|方案)|generic", re.I)


def device_scope(question):
    stated = re.sub(r"(?:不问|不讨论|不要引用)[^，。；]*", "", question)
    if CYBERKNIFE.search(stated) and not compares_devices(stated):
        return CYBERKNIFE
    return None


def compares_devices(question):
    return bool(CYBERKNIFE.search(question) and DEVICE_COMPARISON.search(question)
                and GENERAL_DEVICE.search(question))


def requested_sections(question):
    unnegated = re.sub(r"(?:不问|不讨论|不要引用)[^，。；]*", "", question)
    sections = {name for name, pattern in SECTIONS.items() if re.search(pattern, unnegated)}
    research_section = bool(RESEARCH_SECTION.search(unnegated))
    if research_section:
        sections.add("Evidence")
    research_query = research_section or bool(RESEARCH_QUERY.search(unnegated))
    return sections, research_query


def cross_rerank(base, question, evidence):
    documents = [f"{r['section']}\n{r['text']}" for r in evidence]
    req = Request(base.rstrip("/") + "/v1/rerank",
                  data=json.dumps({"query": question, "documents": documents}).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=90) as response:
        result = json.load(response)
    scores = {}
    for row in result["results"]:
        index, score = row["index"], row["relevance_score"]
        if (type(index) is not int or not 0 <= index < len(evidence)
                or index in scores or not math.isfinite(score)):
            raise ValueError("Invalid reranker response")
        scores[index] = score
    if len(scores) != len(evidence):
        raise ValueError("Reranker omitted a candidate")
    return [dict(evidence[i], rerank_score=scores[i])
            for i in sorted(scores, key=lambda i: (-scores[i], i))]


RELEVANCE_SYSTEM = (
    "你是放疗协议原文的相关性审核员。逐条判断条款是否直接回答题目中的至少一个具体子问。"
    "核对问题指向的检查、动作、治疗时点、患者场景、条件和所问信息类型。"
    "同一个章节或相似术语不是理由，例如增强 CT 与自由呼吸的 4D CT 是不同对象，治疗后与治疗中不同。"
    "问的是协议要求时，试验入组或随机分组条件不能代替协议条款；明确问试验时才可用相应研究。"
    "问及资料是否有某个品牌、阈值或条件时，只讨论该对象的条款不构成对缺失细节的直接支持。"
    "按题面选能覆盖全部子问的最少条款，拒绝重复佐证及不相干补充。"
    "complete 只在这条原文能独立回答全部子问及其限定条件时为 true。"
    "同一条能答全时，不需要用别的条款补充一般背景或其他做法。"
    "对每条记录给出 direct、complete、简短原因及可从该条款连续复制的支持句（无直接支持时留空）。"
)


def check_relevance(base, question, ranked, scope, device_guard=False):
    sections, research_query = requested_sections(question)
    device = device_scope(question) if device_guard else None
    comparison = device_guard and compares_devices(question)
    log, eligible = [], []
    for row in ranked:
        entry = dict(id=row["id"], section=row["section"], rerank_score=row["rerank_score"])
        if (scope is not None and row["source"] not in scope):
            entry["decision"] = "outside_protocol"
        elif row["section"] == "Evidence" and not research_query:
            entry["decision"] = "study_evidence_for_protocol_question"
        elif sections and row["section"] not in sections:
            entry["decision"] = "outside_requested_section"
        elif device is not None and not device.search(row["text"]):
            entry["decision"] = "outside_requested_device"
        else:
            if comparison:
                row = dict(row, device_role=(
                    "CyberKnife 专用" if CYBERKNIFE.search(row["text"]) else "常规患者"))
            eligible.append(row)
            entry["decision"] = "not_selected"
        log.append(entry)
    if not eligible:
        return [], log

    keys = {r["id"]: r for r in eligible}
    schema = {"type": "object", "required": ["checks"], "properties": {
        "checks": {"type": "array", "maxItems": len(keys), "items": {
            "type": "object", "required": ["id", "direct", "complete", "reason", "quote"],
            "properties": {"id": {"type": "string", "enum": list(keys)},
                           "direct": {"type": "boolean"}, "complete": {"type": "boolean"},
                           "reason": {"type": "string"},
                           "quote": {"type": "string"}}
        }}
    }}
    prompt = RELEVANCE_SYSTEM
    if comparison:
        prompt += ("这是一道设备比较题，分别寻找 CyberKnife 专用条款和常规患者条款。"
                   "通用 CBCT 与专用 kV 条款答的是不同对象。两类证据均直接支持时都保留，"
                   "并分别判断一条原文是否答全该对象下的子问。")
    raw = structured_chat(base, prompt, {"question": question, "candidates": [
        {k: r[k] for k in ("id", "section", "text", "device_role") if k in r}
        for r in eligible]}, schema, limit=1024)
    checks = raw.get("checks") if isinstance(raw, dict) else None
    audit_by_id = {entry["id"]: entry for entry in log}
    seen = set()
    selected = set()
    for item in checks if isinstance(checks, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        key = item["id"]
        if key not in keys or key in seen:
            continue
        seen.add(key)
        entry = audit_by_id[key]
        entry["relevance"] = item
        if item.get("direct") is not True:
            entry["decision"] = "not_direct_answer"
            continue
        quote = item.get("quote")
        if not isinstance(quote, str) or not quoted_in(quote.casefold(), keys[key]["text"].casefold()):
            entry["decision"] = "relevance_quote_unverified"
            continue
        selected.add(key)
    ordered = [row for row in eligible if row["id"] in selected]
    def smallest(group, limit):
        complete = [row for row in group if audit_by_id[row["id"]]["relevance"].get("complete") is True]
        return complete[:1] if complete else group[:limit]

    if comparison:
        accepted = [row for role in ("CyberKnife 专用", "常规患者")
                    for row in smallest([r for r in ordered if r["device_role"] == role], 2)]
    else:
        accepted = smallest(ordered, 4)
    accepted_ids = {r["id"] for r in accepted}
    for key in selected:
        audit_by_id[key]["decision"] = (
            "accepted" if key in accepted_ids else "redundant_or_lower_rank")
    return accepted, log


def choose_relevant(base, rerank_base, question, evidence, scope, device_guard=False):
    start = time.monotonic()
    ranked = cross_rerank(rerank_base, question, evidence) if evidence else []
    rerank_seconds = round(time.monotonic() - start, 3)
    accepted, decisions = check_relevance(base, question, ranked, scope, device_guard=device_guard)
    return accepted, dict(rerank_seconds=rerank_seconds,
                          pre_explanation_seconds=round(time.monotonic() - start, 3),
                          ranked_ids=[r["id"] for r in ranked], decisions=decisions)

