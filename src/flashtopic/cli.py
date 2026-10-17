"""flashtopic CLI: cluster a JSONL corpus into labeled topics.

    flashtopic backends                       # list available clustering backends
    flashtopic cluster docs.jsonl.gz --k 60   # fixed k
    flashtopic cluster docs.jsonl --k-range 20 200 --strategy elbow --merge
    flashtopic cluster docs.jsonl --k 60 --label-model anthropic/claude-haiku-4.5 \
        --topic "1990s science fiction novels"   # optional, only if the corpus has one
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_backends(_args) -> int:
    from flashtopic.cluster import available_backends

    avail = available_backends()
    print("available clustering backends:", ", ".join(avail) if avail else "(none)")
    if not avail:
        print("install one: pip install flashtopic[mlx] | [cuda] | [cpu]", file=sys.stderr)
        return 1
    return 0


def _cmd_cluster(args) -> int:
    from flashtopic.config import Workspace
    from flashtopic.ingest import load_docs
    from flashtopic.pipeline import run_pipeline

    docs = load_docs(args.jsonl, text_field=args.text_field, id_field=args.id_field)
    ws = Workspace(
        base_dir=Path(args.base_dir),
        corpus_topic=args.topic or "",
        domain_stopwords=set(args.stopwords or []),
        embed_model=args.embed_model,
        cluster_backend=args.backend,
    )
    res = run_pipeline(
        docs, ws, run_id=args.run_id,
        k=args.k, k_range=tuple(args.k_range) if args.k_range else None,
        backend=args.backend, strategy=args.strategy,
        merge=args.merge, label_model=args.label_model, show_progress=True,
    )
    print(f"\nrun '{res.run_id}': {res.n_docs} docs · {res.n_uniques} unique sentences · k={res.k}")
    if res.sweep:
        print(f"swept k over {res.sweep.ks[0]}..{res.sweep.ks[-1]} ({res.sweep.strategy}) → {res.k}")
    if res.dropped_generic_terms:
        print(f"merge dropped generic terms: {', '.join(res.dropped_generic_terms[:10])}")
    print("\ntopics (largest first):")
    for t in sorted(res.topics, key=lambda t: -t.n_members)[:20]:
        label = t.label or "(" + ", ".join(t.distinctive_terms[:4]) + ")"
        print(f"  {t.topic_id:<6} n={t.n_members:<5} {label}")
    print(f"\nartifacts under {ws.run_dir(res.run_id)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flashtopic", description="Fast GPU topic modeling + prevalence.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("backends", help="list available clustering backends").set_defaults(func=_cmd_backends)

    c = sub.add_parser("cluster", help="cluster a JSONL corpus into labeled topics")
    c.add_argument("jsonl", help="path to a .jsonl or .jsonl.gz corpus")
    c.add_argument("--base-dir", default="flashtopic_runs", help="root for run artifacts")
    c.add_argument("--run-id", default="run")
    c.add_argument("--text-field", default="text", help="dotted path to the document text")
    c.add_argument("--id-field", default=None, help="dotted path to the document id (default: content hash)")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--k", type=int, help="fixed number of clusters")
    g.add_argument("--k-range", type=int, nargs=2, metavar=("MIN", "MAX"), help="sweep k in [MIN, MAX]")
    c.add_argument("--strategy", default="elbow", choices=["elbow", "silhouette"])
    c.add_argument("--backend", default="auto", choices=["auto", "flash_mlx", "flash_cuda", "fastkmeans"])
    c.add_argument("--embed-model", default="minishlab/potion-retrieval-32M")
    c.add_argument("--topic", default="", help="optional overall corpus topic (for LLM labeling)")
    c.add_argument("--stopwords", nargs="*", help="optional domain stopwords to suppress")
    c.add_argument("--merge", action="store_true", help="merge near-duplicate clusters (opt-in)")
    c.add_argument("--label-model", default=None, help="OpenRouter model id to LLM-label topics")
    c.set_defaults(func=_cmd_cluster)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
