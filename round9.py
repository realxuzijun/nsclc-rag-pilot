"""Evaluate PDF publisher transfer on frozen questions and old-source distractors."""
import argparse
import hashlib
import json
import time
from urllib.request import urlopen

from benchmark import ROOT, MODEL, EMBED, request, retrieve, vectors
from build_protocol_tables import TABLE_DB
from publisher_tables import publisher_answer, publisher_scope
from relevance import DEFAULT_RERANK, cross_rerank
from round3 import embed_clauses, store
from round6 import run_arm


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--base", default="http://127.0.0.1:11543")
    cli.add_argument("--rerank-base", default=DEFAULT_RERANK)
    args = cli.parse_args()
    data = ROOT / "data"
    pdf_file = data / "publisher_tables.json"
    old_file = data / "clauses.jsonl"
    external_file = data / "external_clauses.jsonl"
    qfile = ROOT / "questions_round9.json"
    known = json.loads(pdf_file.read_bytes())
    questions = json.loads(qfile.read_bytes())
    new = [dict(row, source=row["protocol"]) for row in known["rows"]]
    old = [r for f in (old_file, external_file)
           for r in map(json.loads, f.read_bytes().splitlines()) if r["source"].isdigit()]
    docs = [f"{r['section']}\n{r['text']}" for r in new]
    decoy_docs = [f"{r['section']}\n{r['text']}" for r in old]
    models = {r["name"]: r["digest"] for r in request(args.base, "tags")["models"]}
    with urlopen(args.rerank_base.rstrip("/") + "/v1/models", timeout=10) as fp:
        reranker = json.load(fp)
    run = dict(model=MODEL, models={k: models[k] for k in (MODEL, EMBED + ":latest")},
               reranker=reranker, sources=known["sources"], ollama=request(args.base, "version")["version"],
               sha256={f.name: sha(f) for f in (pdf_file, old_file, external_file, qfile, TABLE_DB)},
               results=[], embedding_fallbacks=[])
    index = embed_clauses(args.base, docs, run["embedding_fallbacks"])
    for i, q in enumerate(questions):
        text = q["question"]
        scope = publisher_scope(text)
        if len(scope) != 1:
            raise ValueError(f"Question must specify one PDF publisher: {q['id']}")
        start_retrieval = time.monotonic()
        query = vectors(args.base, [text])[0]
        candidates = retrieve(new, docs, q, query, index, allowed_sources=scope)
        retrieval_seconds = time.monotonic() - start_retrieval
        decoys = retrieve(old, decoy_docs, q)[:2]
        if q["id"].startswith("A"):
            other_publisher = {"wmrn", "wfb"} - scope
            decoys += retrieve(new, docs, q, query, index, allowed_sources=other_publisher)[:2]

        def control():
            start = time.monotonic()
            ranked = cross_rerank(args.rerank_base, text, candidates + decoys)
            elapsed = time.monotonic() - start
            return run_arm(args.base, text, ranked, scope, True, elapsed)

        if i % 2:
            after = publisher_answer(args.base, text)
            before = control()
        else:
            before = control()
            after = publisher_answer(args.base, text)
        assert after is not None
        before["retrieval_seconds"] = round(retrieval_seconds, 3)
        before["total_seconds"] = round(retrieval_seconds + before["seconds"], 3)
        run["results"].append(dict(id=q["id"], question=text, scope=sorted(scope),
                                   candidates=candidates, old_decoys=decoys, before=before, after=after))
        store(ROOT / "results/round9.json", run)
        print(q["id"], before["status"], after["status"], flush=True)


if __name__ == "__main__":
    main()

