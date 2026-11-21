"""Compare ordinary, guarded, structured and gold-evidence Qwen3-8B answers."""
import argparse
import hashlib
import json
import re
import sqlite3
import time
from urllib.error import HTTPError

from benchmark import ROOT, MODEL, EMBED, answer, request, retrieve, vectors
from build_structured import FRACTIONS, ORGAN, metric_key, split_clauses
from citation_guard import checked_answer, protocol_scope

METRIC_QUERY = re.compile(r"[DV]\s*\d+(?:\.\d+)?(?:\s*(?:cc|cm³|%|Gy))?", re.I)
PRESCRIPTION = re.compile(r"(\d+(?:\.\d+)?)\s*Gy\s*(?:分|/)\s*\d+\s*次", re.I)


def dose_query(question, scope, db):
    action = re.search(r"平均|最大|限值|约束|上限|限制", question)
    organs = [name for name, pattern in ORGAN.items() if re.search(pattern, question, re.I)]
    if not action or not organs:
        return None
    start = time.monotonic()
    explicit_metrics = [metric_key(m.group()) for m in METRIC_QUERY.finditer(question)]
    explicit_metrics.extend(key for word, key in [("平均", "mean"), ("最大", "max")] if word in question)
    metrics = set(explicit_metrics)
    fractions = {int(n) for n in FRACTIONS.findall(question)}
    rx = PRESCRIPTION.search(question)

    def result(status, message, records=()):
        return dict(route="sqlite", status=status, answer=message, evidence=list(records),
                    seconds=round(time.monotonic() - start, 3))

    if scope is None or len(scope) != 1 or len(organs) != 1:
        return result("needs_fields", "请明确一份协议和要查询的危及器官。")
    protocol = next(iter(scope))
    rows = [dict(row) for row in db.execute(
        "SELECT * FROM dose_cell WHERE protocol=? AND organ=? ORDER BY fractions",
        (protocol, organs[0]))]
    if not rows:
        return result("not_listed", "所选协议的表中未找到该器官。")
    columns = {row["fractions"] for row in rows}
    if columns != {0}:
        if len(fractions) != 1:
            return result("needs_fields", "请补充该协议使用的分割次数，表中有多个方案列。")
        fraction = next(iter(fractions))
        rows = [row for row in rows if row["fractions"] == fraction]
        if rx:
            dose = float(rx.group(1))
            rows = [row for row in rows if dose in
                    [float(x) for x in re.findall(r"\d+(?:\.\d+)?", row["regimen"].split(" Gy")[0])]]
    elif fractions:
        return result("needs_fields", "该约束表未按分割次数分列，需要核对方案后才能按所问分割返回约束。")
    if not rows:
        return result("not_listed", "所选协议的表中没有对应的分割方案列。")
    row = rows[0]
    selected = [dict(metric=m["metric"], literal=m["literal"]) for m in db.execute(
        "SELECT metric, literal FROM dose_metric WHERE cell_id=? ORDER BY ordinal", (row["id"],))
        if not metrics or m["metric"] in metrics]
    if metrics and not metrics <= {m["metric"] for m in selected}:
        return result("not_listed", "对应的表格单元未明确列出所有所问指标。", rows)
    row["selected_metrics"] = selected
    display = "；".join(m["literal"] for m in selected) if metrics else row["cell"]
    regimen = f"，{row['regimen']}" if row["fractions"] else ""
    return result("answered", f"eviQ {row['version']}{regimen}：{display} [{row['id']}]", rows)


def gold_passages(spec, by_id):
    parent = by_id[spec["parent"]]
    pattern = re.compile(spec["match"], re.I)
    if spec["table"]:
        lines = parent["text"].splitlines()
        lines = [line for line in lines if "|" in line]
        row = next(line for line in lines if pattern.search(line))
        if spec["column"] not in {None, "all"}:
            header = lines[1].split("|")
            ci = next(i for i, value in enumerate(header)
                      if str(spec["column"]) in FRACTIONS.findall(value))
            evidence = f"{header[ci].strip()}\n{row.split('|')[0].strip()} | {row.split('|')[ci].strip()}"
        else:
            n = 2 if spec["column"] == "all" else 1
            evidence = "\n".join(lines[:n] + [row])
    else:
        evidence = "\n".join(text for text in split_clauses(parent) if pattern.search(text))
    if not evidence:
        raise ValueError(f"No gold evidence for {spec['id']}")
    return [dict(parent, id=f"{parent['id']}-G01", parent_id=parent["id"], text=evidence)]


def store(path, data):
    draft = path.with_suffix(".tmp")
    draft.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    draft.replace(path)


def embed_clauses(base, docs, log):
    def batch(items, offset):
        try:
            return vectors(base, items)
        except HTTPError as exc:
            if exc.code != 500:
                raise
            if len(items) > 1:
                half = len(items) // 2
                return batch(items[:half], offset) + batch(items[half:], offset + half)
            try:
                result = request(base, "embed", {"model": EMBED, "input": items,
                                                  "truncate": False, "options": {"num_gpu": 0}})
                log.append({"clause_index": offset, "fallback": "cpu"})
                return result["embeddings"]
            except HTTPError as solo:
                if solo.code != 500:
                    raise
                log.append({"clause_index": offset, "fallback": "bm25_only"})
                return [None]
    result = []
    for offset in range(0, len(docs), 12):
        result.extend(batch(docs[offset:offset + 12], offset))
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--base", default="http://127.0.0.1:11543")
    args = cli.parse_args()
    data = ROOT / "data"
    corpus_file = data / "chunks.jsonl"
    clause_file = data / "clauses.jsonl"
    question_file = ROOT / "questions_round3.json"
    gold_file = ROOT / "round3_gold.json"
    db_file = data / "dose_tables.sqlite"
    files = [corpus_file, clause_file, question_file, gold_file, db_file]
    chunks, clauses = [[json.loads(line) for line in f.read_bytes().splitlines()]
                       for f in (corpus_file, clause_file)]
    questions = json.loads(question_file.read_bytes())
    specs = {row["id"]: row for row in json.loads(gold_file.read_bytes())}
    by_id = {row["id"]: row for row in chunks}
    gold = {q["id"]: gold_passages(specs[q["id"]], by_id) for q in questions}
    models = {x["name"]: x["digest"] for x in request(args.base, "tags")["models"]}
    run = dict(model=MODEL, models={k: models[k] for k in (MODEL, EMBED + ":latest")},
               ollama=request(args.base, "version")["version"],
               sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, results=[])
    chunk_docs = [f"{r['title']} / {r['section']}\n{r['text']}" for r in chunks]
    clause_docs = [f"{r['section']}\n{r['text']}" for r in clauses]
    queries = vectors(args.base, [q["question"] for q in questions])
    chunk_index = vectors(args.base, chunk_docs)
    run["clause_embedding_fallbacks"] = []
    clause_index = embed_clauses(args.base, clause_docs, run["clause_embedding_fallbacks"])
    target = ROOT / "results/round3.json"
    with sqlite3.connect(f"{db_file.as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for q, query in zip(questions, queries):
            question = q["question"]
            scope = protocol_scope(question)
            broad = retrieve(chunks, chunk_docs, q, query, chunk_index)
            restricted = retrieve(chunks, chunk_docs, q, query, chunk_index, allowed_sources=scope)
            ordinary = answer(args.base, question, broad)
            guarded = checked_answer(args.base, question, restricted, scope)
            structured = dose_query(question, scope, db)
            if structured is None:
                fine = retrieve(clauses, clause_docs, q, query, clause_index, allowed_sources=scope)
                structured = checked_answer(args.base, question, fine, scope)
                structured.update(route="clause", evidence=fine)
            exact = gold[q["id"]]
            oracle = answer(args.base, question, exact)
            run["results"].append(dict(id=q["id"], question=question, ordinary=ordinary,
                                       guarded=guarded, structured=structured, oracle=oracle,
                                       ordinary_evidence=broad, guarded_evidence=restricted,
                                       oracle_evidence=exact))
            store(target, run)
            print(q["id"], structured["route"], structured["status"], flush=True)


if __name__ == "__main__":
    main()

