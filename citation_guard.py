"""Protocol routing and claim-level citation verification for public-source RAG."""
import json
import re
import time

from benchmark import MODEL, request

ALIASES = {
    "1871": r"常规(?:根治性)?(?:外照射|放疗|分割)|常规协议|标准分割",
    "3697": r"大分割|低分割|hypofractionat",
    "3389": r"外周型|周围型|周边型|peripheral",
    "4018": r"(?<!超)中央型|中心型|\bcentral\b",
}
KNOWN_ID = re.compile(r"(?<!\d)(1871|1872|1873|3679|3697|3389|4018)(?!\d)")
NAMED_ID = re.compile(r"(?:eviq\s*(?:协议)?|协议(?:编号)?|ID)\s*[:：#]?\s*(\d{4})(?!\d)", re.I)
SABR = re.compile(r"SBRT|SABR|立体定向", re.I)


def protocol_scope(question):
    """Return an allowlist, or None when no protocol restriction is identifiable."""
    named = set(NAMED_ID.findall(question))
    known = set(KNOWN_ID.findall(question))
    if named or known:
        return named | known
    matched = {key for key, alias in ALIASES.items() if re.search(alias, question, re.I)}
    if matched:
        return matched
    if SABR.search(question):
        return {"3389", "4018"}
    if re.search(r"NCI|PDQ", question, re.I):
        return {"pdq"}
    return None


def quoted_in(quote, text):
    normalize = lambda value: " ".join(value.split())
    return len(normalize(quote)) >= 12 and normalize(quote) in normalize(text)


def structured_chat(base, system, payload, schema, limit=1024):
    response = request(base, "chat", {
        "model": MODEL, "think": False, "stream": False, "keep_alive": "5m",
        "options": {"seed": 41, "temperature": 0.2, "num_ctx": 8192,
                    "num_predict": limit},
        "format": schema,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    })
    try:
        if response.get("done_reason") != "stop" or response["message"].get("thinking"):
            return None
        return json.loads(response["message"]["content"])
    except (KeyError, TypeError, ValueError):
        return None


STR = {"type": "string"}
CLAIM_SCHEMA = {"type": "object", "required": ["claims"], "properties": {
    "claims": {"type": "array", "maxItems": 4, "items": {
        "type": "object", "required": ["claim", "id", "quote"], "properties": {
            "claim": STR, "id": STR, "quote": STR,
        }
    }}
}}
CHECK_SCHEMA = {"type": "object", "required": ["supported", "reason"], "properties": {
    "supported": {"type": "boolean"}, "reason": STR,
}}


def checked_answer(base, question, evidence, scope):
    start = time.monotonic()
    by_id = {row["id"]: row for row in evidence}
    audit, verified = [], []
    if scope and not evidence:
        status = "out_of_corpus"
    elif (scope == {"3389", "4018"} and not any(
            re.search(ALIASES[x], question, re.I) or x in KNOWN_ID.findall(question)
            for x in scope) and re.search(r"约束|限值|上限|剂量|分割", question)):
        # A single regimen-dependent number is underdetermined without identifying the protocol.
        status = "needs_protocol"
    elif evidence:
        draft = structured_chat(
            base,
            "你在做放疗协议文献问答。只返回能由提供原文直接支持的中文要点。"
            "每条只写一个可核查要点，使用原始证据ID和原文中可连续复制的英文引文。"
            "若必须引用表格某列，请复制该表的相关行；保留分割数、指标、单位和大小关系。"
            "问题指定协议时按该协议作答。比较题分别标明协议。缺少直接证据时省略该要点。",
            {"question": question, "evidence": [
                {k: row[k] for k in ("id", "source", "section", "text")} for row in evidence]},
            CLAIM_SCHEMA,
        )
        claims = draft.get("claims", []) if isinstance(draft, dict) else []
        for item in claims[:4]:
            if not isinstance(item, dict):
                continue
            claim, key, quote = (item.get(k) for k in ("claim", "id", "quote"))
            if not all(isinstance(value, str) and value.strip() for value in (claim, key, quote)):
                continue
            row = by_id.get(key)
            record = dict(claim=claim, id=key, quote=quote)
            audit.append(record)
            if row is None or (scope is not None and row["source"] not in scope):
                record["reason"] = "citation_outside_retrieved_scope"
                continue
            if not quoted_in(quote, row["text"]):
                record["reason"] = "quote_not_found"
                continue
            stated = set(KNOWN_ID.findall(claim))
            if stated and row["source"] not in stated:
                record["reason"] = "protocol_mismatch"
                continue
            verdict = structured_chat(
                base,
                "你是引用复核员。只判断这一个章节是否直接支持这条中文要点，且要点与问题相关。"
                "一条要点可以只回答一个子问。不要因为它没有回答其他子问而否决，也不要评价其他要点。"
                "以英文原文为准，检查协议、章节、术语、条件、方向、单位、数值和表格列。"
                "相同主题、标题相似、答非所问、数值方向不明或只有部分支持均判 false。"
                "不得以常识、其他协议或未提供章节补全。表格可查本片段的表头对齐引文所在列。",
                {"question": question, "claim": claim, "quote": quote, "evidence": row},
                CHECK_SCHEMA, limit=320,
            )
            record["verdict"] = verdict
            if isinstance(verdict, dict) and verdict.get("supported") is True:
                record["reason"] = "accepted"
                verified.append(dict(claim=claim, id=key, source=row["source"], section=row["section"]))
            else:
                record["reason"] = "unsupported_or_unverified"
        status = "answered" if verified else "no_verified_claim"
    else:
        status = "no_evidence"

    if verified:
        answer = "\n".join(f"{item['claim']} [{item['id']}]" for item in verified)
    elif status == "needs_protocol":
        answer = "请补充协议所对应的肿瘤位置（中央型或外周型）及适用分割方案，才能确定协议里的相应条款。"
    elif status == "out_of_corpus":
        answer = "当前实验资料未收录指定协议，无法核验该协议的内容。"
    else:
        answer = "当前检索片段未能为答案提供通过核验的直接证据。"
    return dict(answer=answer, status=status, scope=sorted(scope) if scope is not None else None,
                verified=verified, audit=audit, seconds=round(time.monotonic() - start, 2))

