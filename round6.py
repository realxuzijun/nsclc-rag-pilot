"""Test device scoping and Chinese subquestion coverage on matched evidence."""
import argparse
import hashlib
import json
import time
from urllib.request import urlopen

from benchmark import ROOT, MODEL, EMBED, retrieve, request, vectors
from citation_guard import protocol_scope
from protocol_answer import explain_passages
from relevance import DEFAULT_RERANK, check_relevance, cross_rerank
from round3 import embed_clauses, store


def run_arm(base, question, ranked, scope, enhanced, rerank_seconds):
    start = time.monotonic()
    selected, notes = check_relevance(base, question, ranked, scope, device_guard=enhanced)
    out = explain_passages(base, question, selected, scope, preselected=True, enhanced=enhanced)
    out["seconds"] = round(rerank_seconds + time.monotonic() - start, 3)
    return dict(out, selection=notes)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--base", default="http://127.0.0.1:11543")
    cli.add_argument("--rerank-base", default=DEFAULT_RERANK)
    args = cli.parse_args()
    base, rerank = args.base, args.rerank_base
    q_file = ROOT / "questions_round6.json"
    clause_file = ROOT / "data/clauses.jsonl"
    past_file = ROOT / "results/round5_final.json"
    questions = json.loads(q_file.read_bytes())
    past = json.loads(past_file.read_bytes())
    frozen = {r["question"]: r["candidates"] for r in past["results"]}
    models = {x["name"]: x["digest"] for x in request(base, "tags")["models"]}
    file_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in (q_file, clause_file, past_file)}
    if past["sha256"]["clauses.jsonl"] != file_hashes["clauses.jsonl"]:
        raise ValueError("Frozen regression controls use a different corpus")
    with urlopen(rerank.rstrip("/") + "/v1/models", timeout=10) as response:
        rerank_models = json.load(response)
    run = dict(model=MODEL, models={k: models[k] for k in (MODEL, EMBED + ":latest")},
               rerank_models=rerank_models, ollama=request(base, "version")["version"],
               sha256=file_hashes, embedding_fallbacks=[], results=[])
    scopes = [protocol_scope(q["question"]) for q in questions]
    clauses = [c for c in map(json.loads, clause_file.read_bytes().splitlines())
               if any(scope is None or c["source"] in scope for scope in scopes)]
    docs = [f"{c['section']}\n{c['text']}" for c in clauses]
    index = embed_clauses(base, docs, run["embedding_fallbacks"])
    queries = vectors(base, [q["question"] for q in questions])
    for n, (q, scope, vector) in enumerate(zip(questions, scopes, queries)):
        question = q["question"]
        if q["id"].startswith("R"):
            evidence = frozen[question]
        else:
            evidence = retrieve(clauses, docs, q, vector, index, allowed_sources=scope)
        start = time.monotonic()
        ranked = cross_rerank(rerank, question, evidence) if evidence else []
        rerank_seconds = time.monotonic() - start
        outputs = {}
        order = [False, True] if n % 2 == 0 else [True, False]
        for flag in order:
            outputs["after" if flag else "before"] = run_arm(
                base, question, ranked, scope, flag, rerank_seconds)
        run["results"].append(dict(id=q["id"], question=question, candidates=evidence,
                                   ranking=ranked, rerank_seconds=round(rerank_seconds, 3),
                                   order=["after" if flag else "before" for flag in order],
                                   **outputs))
        store(ROOT / "results/round6.json", run)
        print(q["id"], outputs["before"]["status"], outputs["after"]["status"], flush=True)


if __name__ == "__main__":
    main()

