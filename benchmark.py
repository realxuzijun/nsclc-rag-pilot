"""Run paired Qwen3-8B inference with and without retrieved public evidence."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MODEL = "qwen3:8b"
EMBED = "bge-m3"
TOKENS = re.compile(r"[a-z0-9]+")
SYSTEM = ("你在参加非小细胞肺癌放疗文献问答测试。用中文简洁回答，最多四句话。"
          "区分适用条件、排除条件以及未提供的信息。避免把单一协议的条件说成普遍标准。"
          "如果提供了证据，只依据这些片段作答，每个关键结论标注对应的 [证据ID]。"
          "如果材料不足，说明缺少什么。没有提供证据时根据自身知识作答，不要编造引用。")


def request(base, path, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    query = Request(base + "/api/" + path, data=payload,
                    headers={"Content-Type": "application/json"})
    with urlopen(query, timeout=300) as response:
        return json.load(response)


def bm25(query, passages):
    docs = [Counter(TOKENS.findall(text.lower())) for text in passages]
    lengths = list(map(lambda doc: sum(doc.values()), docs))
    mean = sum(lengths) / len(docs)
    terms = set(TOKENS.findall(query.lower()))
    weight = {}
    for word in terms:
        df = sum(word in doc for doc in docs)
        weight[word] = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
    return [sum(weight[w] * docs[i][w] * 2.2 /
                (docs[i][w] + 1.2 * (0.25 + 0.75 * lengths[i] / mean))
                for w in terms) for i in range(len(docs))]


def vectors(base, texts):
    result = []
    for i in range(0, len(texts), 12):
        result.extend(request(base, "embed", {"model": EMBED, "input": texts[i:i+12],
                                               "truncate": False})["embeddings"])
    if len(result) != len(texts):
        raise ValueError("Incorrect number of embeddings")
    return result


def retrieve(rows, docs, question, vector=None, index=None, scoped=False, allowed_sources=None):
    lexical = bm25(question["search_terms"], docs)
    candidates = range(len(rows))
    if scoped:
        # Diagnostic scope derived from the protocol names in the question.
        text = question["question"]
        if "超中央型" in text or "中央型肺立体定向" in text:
            sources = {"4018"}
        elif "外周型" in text:
            sources = {"3389"}
        elif "常规协议" in text:
            sources = {"1871"}
        elif "大分割根治性" in text:
            sources = {"3697"}
        elif "SBRT" in text and "剂量约束" in text:
            sources = {"3389", "4018"}
        else:
            sources = {r["source"] for r in rows}
        candidates = [i for i, row in enumerate(rows) if row["source"] in sources]
    if allowed_sources is not None:
        candidates = [i for i in candidates if rows[i]["source"] in allowed_sources]
    rankings = [sorted(candidates, key=lambda i: -lexical[i])] if any(
        lexical[i] > 0 for i in candidates) else []
    if vector is not None:
        dense = {i: sum(a*b for a, b in zip(vector, index[i]))
                 for i in candidates if index[i] is not None}
        rankings.append(sorted(dense, key=lambda i: -dense[i]))
    scores = Counter()
    for ranking in rankings:
        for rank, i in enumerate(ranking[:24], 1):
            scores[i] += 1 / (30 + rank)
    ranked = [i for i, _ in scores.most_common()]
    if allowed_sources and len(allowed_sources) > 1:
        first_by_source = {}
        for i in ranked:
            first_by_source.setdefault(rows[i]["source"], i)
        reserved = list(first_by_source.values())[:6]
        keep = set(reserved + [i for i in ranked if i not in reserved][:6 - len(reserved)])
        ranked = [i for i in ranked if i in keep]
    return [rows[i] for i in ranked[:6]]


def answer(base, question, rows):
    evidence = "\n\n".join(f"[{r['id']}] {r['title']} ({r['version']}) / {r['section']}\n{r['text']}"
                              for r in rows)
    options = {"seed": 41, "temperature": 0.2, "num_ctx": 8192, "num_predict": 512}
    prompt = f"问题：{question}"
    if evidence:
        prompt += f"\n\n资料片段：\n{evidence}"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    start = time.monotonic()
    response = request(base, "chat", {"model": MODEL, "messages": messages,
                                     "stream": False, "think": False, "options": options,
                                     "keep_alive": "5m"})
    content = response["message"]["content"]
    if not content.strip() or response["message"].get("thinking"):
        raise RuntimeError("Expected a nonempty response with thinking disabled")
    return dict(answer=content, seconds=round(time.monotonic()-start, 2), options=options,
                done_reason=response.get("done_reason"),
                prompt_tokens=response.get("prompt_eval_count"),
                output_tokens=response.get("eval_count"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:11543")
    parser.add_argument("--retrieval", choices=["hybrid", "bm25", "scoped"], default="hybrid")
    args = parser.parse_args()
    model_info = request(args.base, "tags")["models"]
    digests = {entry["name"]: entry["digest"] for entry in model_info}
    if MODEL not in digests:
        raise RuntimeError(f"{MODEL} is not installed")
    raw = (ROOT / "data/chunks.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    docs = [f"{r['title']} / {r['section']}\n{r['text']}" for r in rows]
    index, queries = None, [None] * len(questions)
    if args.retrieval != "bm25":
        if f"{EMBED}:latest" not in digests:
            raise RuntimeError(f"{EMBED} is not installed")
        index = vectors(args.base, docs)
        queries = vectors(args.base, [q["question"] for q in questions])
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    run = dict(model=MODEL, model_digest=digests[MODEL], retrieval=args.retrieval,
               embedding_digest=digests.get(f"{EMBED}:latest") if index else None,
               ollama=request(args.base, "version")["version"],
               corpus_sha256=hashlib.sha256(raw).hexdigest(), results=[])
    target = out / f"{args.retrieval}.json"
    for question, query in zip(questions, queries):
        evidence = retrieve(rows, docs, question, query, index,
                            scoped=args.retrieval == "scoped")
        baseline = answer(args.base, question["question"], [])
        rag = answer(args.base, question["question"], evidence)
        run["results"].append(dict(id=question["id"], question=question["question"],
                                   search_terms=question["search_terms"],
                                   baseline=baseline, rag=rag, retrieved=evidence))
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
        print(f"{question['id']}: {baseline['seconds']:.2f}s / {rag['seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
