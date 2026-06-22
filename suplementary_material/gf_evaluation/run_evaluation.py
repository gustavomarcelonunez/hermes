"""
run_evaluation.py

Main evaluation script for HERMES.
Runs the full N x M x J matrix: 6 KGs x 4 LLMs x 10 CQs = 240 executions.
Each execution generates an answer and verifies it against the retrieved subgraph,
recording Grounding Fidelity metrics (GF_strict, GF_weighted).

Run from the project root:
    python run_evaluation.py

Output:
    evaluation_results.csv
"""

import csv
import time
import traceback

from utils.load_graph import load_graph
from utils.embeddings import load_graph_embeddings, build_graph_embeddings
from utils.retrieve import get_top_k_nodes, extract_subgraph, choose_hops
from utils.qa import run_qa, verify_answer

# -----------------------------
# Experiment configuration
# -----------------------------

GRAPHS = [
    "gpt-4o-mini",
    "mixtral8x7b",
    "mistral7b",
    "ministral-14b-2512",
    "mistral-large3",
    "llama3-8b-instruct",
]

LLM_KEYS = [
    "openai_gpt4omini",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemini-2.5-flash",
]


QUERIES = {
    "CQ1":  "¿Qué especies de pinnípedos fueron estudiadas a lo largo de la costa marítima argentina e islas adyacentes?",
    "CQ2":  "¿Durante qué años se realizaron las investigaciones oficiales sobre pinnípedos, pingüinos y guano de aves marinas?",
    "CQ3":  "¿Qué organismo gubernamental avaló y facilitó las investigaciones descritas en el texto?",
    "CQ4":  "¿Cuáles fueron los principales métodos y medios de transporte utilizados durante las diferentes comisiones de investigación?",
    "CQ5":  "¿Cómo se caracterizan morfológica y conductualmente los lobos marinos del sur (Otaria flavescens)?",
    "CQ6":  "¿Cuáles son las principales diferencias físicas entre machos y hembras de Otaria flavescens?",
    "CQ7":  "¿Dónde estaban ubicadas las colonias de lobos marinos de dos pelos (Arctocephalus australis) según el estudio?",
    "CQ8":  "¿Qué razones da el texto para la prohibición de la caza de Arctocephalus australis?",
    "CQ9":  "¿Cómo se describe la locomoción de los elefantes marinos del sur (Mirounga leonina) en el documento?",
    "CQ10": "¿Qué papel desempeñaron la fotografía aérea y la cartografía en la recopilación de datos de las investigaciones?",
}

SIM_THRESHOLD = 0.55
MIN_K         = 10
OUTPUT_CSV    = "evaluation_results.csv"

FIELDNAMES = [
    "graph", "llm_qa", "cq_id",
    "hops", "subgraph_nodes", "subgraph_edges",
    "total_claims", "supported", "partial", "unsupported",
    "gf_strict", "gf_weighted",
    "parse_error", "runtime_error",
]

# -----------------------------
# Main experiment loop
# -----------------------------

def main():
    total   = len(GRAPHS) * len(LLM_KEYS) * len(QUERIES)
    current = 0
    rows    = []

    for graph_name in GRAPHS:
        print(f"\n{'='*60}")
        print(f"Loading graph: {graph_name}")
        print(f"{'='*60}")

        try:
            G = load_graph(graph_name)
        except Exception as e:
            print(f"  [ERROR] Could not load graph {graph_name}: {e}")
            continue

        embeddings, node_list = load_graph_embeddings(graph_name)
        if embeddings is None:
            print(f"  Embeddings not found, building them now...")
            embeddings, node_list = build_graph_embeddings(G, graph_name)

        for cq_id, question in QUERIES.items():
            hops = choose_hops(question)

            top_nodes = get_top_k_nodes(
                question, embeddings, node_list,
                min_k=MIN_K, sim_threshold=SIM_THRESHOLD
            )
            nodes_only = [n for n, _ in top_nodes]
            subg = extract_subgraph(G, nodes_only, hops=hops)

            for llm_key in LLM_KEYS:
                current += 1
                print(f"  [{current}/{total}] {graph_name} | {cq_id} | {llm_key}")

                row = {
                    "graph":          graph_name,
                    "llm_qa":         llm_key,
                    "cq_id":          cq_id,
                    "hops":           hops,
                    "subgraph_nodes": subg.number_of_nodes(),
                    "subgraph_edges": subg.number_of_edges(),
                    "total_claims":   0,
                    "supported":      0,
                    "partial":        0,
                    "unsupported":    0,
                    "gf_strict":      0.0,
                    "gf_weighted":    0.0,
                    "parse_error":    None,
                    "runtime_error":  None,
                }

                try:
                    answer = run_qa(question, subg, llm_key)
                    verification = verify_answer(answer, subg, llm_key)

                    row.update({
                        "total_claims": verification["total"],
                        "supported":    verification["supported"],
                        "partial":      verification["partial"],
                        "unsupported":  verification["unsupported"],
                        "gf_strict":    verification["gf_strict"],
                        "gf_weighted":  verification["gf_weighted"],
                        "parse_error":  verification["parse_error"],
                    })
                    if verification["parse_error"]:
                        print(f"    RAW OUTPUT: {repr(verification['raw_json'][:300])}")

                except Exception as e:
                    error_msg = traceback.format_exc()
                    print(f"    [ERROR] {e}")
                    row["runtime_error"] = str(e)

                rows.append(row)

    # Write results to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    errors = sum(1 for r in rows if r["runtime_error"] or r["parse_error"])
    print(f"\n{'='*60}")
    print(f"Done. {len(rows)} rows written to {OUTPUT_CSV}.")
    print(f"Errors: {errors}/{len(rows)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()