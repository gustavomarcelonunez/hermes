"""
rerun_failed.py

Re-runs failed rows from evaluation_results.csv and replaces them in place.
Targets rows where parse_error contains 'GEMINI ERROR 503'.

Run from the project root:
    python rerun_failed.py
"""

import csv
import time
import traceback

from utils.load_graph import load_graph
from utils.embeddings import load_graph_embeddings, build_graph_embeddings
from utils.retrieve import get_top_k_nodes, extract_subgraph, choose_hops
from utils.qa import run_qa, verify_answer

INPUT_CSV  = "evaluation_results.csv"
OUTPUT_CSV = "evaluation_results.csv"  # overwrite in place

SIM_THRESHOLD = 0.55
MIN_K         = 10

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

FIELDNAMES = [
    "graph", "llm_qa", "cq_id",
    "hops", "subgraph_nodes", "subgraph_edges",
    "total_claims", "supported", "partial", "unsupported",
    "gf_strict", "gf_weighted",
    "parse_error", "runtime_error",
]


def is_gemini_503(row):
    err = row.get("parse_error", "") or ""
    return row.get("llm_qa") == "gemini-2.5-flash" and err != ""


def rerun_row(row, graph_cache):
    graph_name = row["graph"]
    llm_key    = row["llm_qa"]
    cq_id      = row["cq_id"]
    question   = QUERIES[cq_id]

    # Load graph and embeddings (cached)
    if graph_name not in graph_cache:
        G = load_graph(graph_name)
        embeddings, node_list = load_graph_embeddings(graph_name)
        if embeddings is None:
            embeddings, node_list = build_graph_embeddings(G, graph_name)
        graph_cache[graph_name] = (G, embeddings, node_list)

    G, embeddings, node_list = graph_cache[graph_name]

    hops = choose_hops(question)
    top_nodes  = get_top_k_nodes(question, embeddings, node_list,
                                  min_k=MIN_K, sim_threshold=SIM_THRESHOLD)
    nodes_only = [n for n, _ in top_nodes]
    subg       = extract_subgraph(G, nodes_only, hops=hops)

    new_row = dict(row)
    new_row.update({
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
    })

    try:
        answer       = run_qa(question, subg, llm_key)
        verification = verify_answer(answer, subg, llm_key)
        new_row.update({
            "total_claims": verification["total"],
            "supported":    verification["supported"],
            "partial":      verification["partial"],
            "unsupported":  verification["unsupported"],
            "gf_strict":    verification["gf_strict"],
            "gf_weighted":  verification["gf_weighted"],
            "parse_error":  verification["parse_error"],
        })
    except Exception as e:
        new_row["runtime_error"] = str(e)
        print(f"    [ERROR] {e}")

    return new_row


def main():
    # Read all rows
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Identify failed rows
    failed_indices = [i for i, r in enumerate(rows) if is_gemini_503(r)]
    print(f"Found {len(failed_indices)} Gemini 503 rows to re-run.")

    graph_cache = {}
    rerun_count = 0
    error_count = 0

    for idx in failed_indices:
        row = rows[idx]
        rerun_count += 1
        print(f"  [{rerun_count}/{len(failed_indices)}] "
              f"{row['graph']} | {row['cq_id']} | {row['llm_qa']}")

        new_row = rerun_row(row, graph_cache)

        if new_row.get("parse_error") or new_row.get("runtime_error"):
            error_count += 1
            print(f"    [STILL FAILING] parse_error={new_row.get('parse_error')} "
                  f"runtime_error={new_row.get('runtime_error')}")

        rows[idx] = new_row

        # Small sleep to avoid hammering Gemini
        time.sleep(2)

    # Write updated CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {OUTPUT_CSV} updated.")
    print(f"Re-ran: {rerun_count} | Still failing: {error_count}")


if __name__ == "__main__":
    main()