"""
sensitivity_analysis.py

Threshold sensitivity analysis for HERMES retrieval pipeline.

Varies `sim_threshold` in get_top_k_nodes (keeping min_k=10 fixed) across
a small set of competency questions and two representative knowledge graphs,
measuring subgraph size and Grounding Fidelity (GF) for each configuration.

Run from the project root:
    python sensitivity_analysis.py
"""

import csv
import numpy as np
import networkx as nx

from utils.load_graph import load_graph
from utils.embeddings import load_graph_embeddings, build_graph_embeddings, embed_text
from utils.retrieve import choose_hops, extract_subgraph
from utils.qa import run_qa, verify_answer

# -----------------------------
# Experiment configuration
# -----------------------------

GRAPHS = ["gpt-4o-mini", "mixtral8x7b"]

MODEL_KEY = "openai_gpt4omini"

MIN_K = 10

THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]

# Subset of CQs from RAGE-KG 2025 (CQ1, CQ3, CQ7, CQ9)
QUERIES = {
    "CQ1": "¿Qué especies de pinnípedos fueron estudiados a lo largo de la costa marítima argentina e islas adyacentes?",
"CQ3": "¿Qué organismo gubernamental avaló y facilitó las investigaciones descritas en el texto?",
"CQ7": "¿Dónde estaban ubicadas las colonias de lobos marinos de dos pelos (Arctocephalus australis) según el estudio?",
"CQ9": "¿Cómo se describe la locomoción de los elefantes marinos del sur (Mirounga leonina) en el documento?",
}

OUTPUT_CSV = "sensitivity_results_es.csv"


# -----------------------------
# Helper: top-k nodes with explicit threshold, returning extra stats
# -----------------------------

def get_top_k_nodes_with_stats(query, embeddings, node_list, min_k, sim_threshold):
    """
    Same logic as utils.retrieve.get_top_k_nodes, but also returns how many
    nodes passed the similarity threshold "naturally" (before the min_k
    guarantee kicks in).
    """
    query_emb = np.array(embed_text(query))

    sims = embeddings @ query_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )

    idx_sorted = np.argsort(sims)[::-1]

    selected = []
    n_above_threshold = 0
    for i in idx_sorted:
        if sims[i] >= sim_threshold:
            n_above_threshold += 1

        if sims[i] < sim_threshold and len(selected) >= min_k:
            break
        selected.append((node_list[i], float(sims[i])))

    return selected, n_above_threshold


# -----------------------------
# Main experiment loop
# -----------------------------

def main():
    rows = []

    for graph_name in GRAPHS:
        print(f"\n=== Loading graph: {graph_name} ===")
        G = load_graph(graph_name)

        embeddings, node_list = load_graph_embeddings(graph_name)
        if embeddings is None:
            print(f"Embeddings not found for {graph_name}, building them now...")
            embeddings, node_list = build_graph_embeddings(G, graph_name)

        for q_id, question in QUERIES.items():
            hops = choose_hops(question)

            for threshold in THRESHOLDS:
                print(f"  -> {graph_name} | {q_id} | threshold={threshold}")

                top_nodes, n_above = get_top_k_nodes_with_stats(
                    question, embeddings, node_list, MIN_K, threshold
                )
                nodes_only = [n for n, _ in top_nodes]

                subg = extract_subgraph(G, nodes_only, hops=hops)

                answer = run_qa(question, subg, MODEL_KEY)
                verification = verify_answer(answer, subg, MODEL_KEY)

                rows.append({
                    "graph": graph_name,
                    "query_id": q_id,
                    "query_length": len(question),
                    "hops": hops,
                    "threshold": threshold,
                    "min_k": MIN_K,
                    "n_seed_nodes": len(nodes_only),
                    "n_above_threshold": n_above,
                    "subgraph_nodes": subg.number_of_nodes(),
                    "subgraph_edges": subg.number_of_edges(),
                    "total_claims": verification["total"],
                    "supported": verification["supported"],
                    "partial": verification["partial"],
                    "unsupported": verification["unsupported"],
                    "gf_strict": verification["gf_strict"],
                    "gf_weighted": verification["gf_weighted"],
                    "parse_error": verification["parse_error"],
                })

    # Write results to CSV
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results written to {OUTPUT_CSV} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
