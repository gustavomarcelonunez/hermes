import streamlit as st
from utils.load_graph import load_graph
from utils.visualize_graph import visualize_graph
from utils.compute_metrics import compute_metrics
from utils.embeddings import load_graph_embeddings, build_graph_embeddings
from utils.retrieve import get_top_k_nodes, extract_subgraph, choose_hops
from utils.qa import run_qa, verify_answer
from utils.video_popup import show_video
from utils.changelog import show_changelog
from utils.welcome_popup import show_welcome_popup
import streamlit.components.v1 as components
import pandas as pd


show_welcome_popup()

st.set_page_config(page_title="HERMES – Graph Explorer", page_icon="🧭", layout="wide")

st.title("🧭 HERMES – Historical Ecology Retrieval & Multi-LLM Evaluation System.")
st.markdown("### **Step 1: Visualization and Interaction with Graphs**")

# --- Sidebar ---
st.sidebar.header("Select KG")
graph_choice = st.sidebar.selectbox(
    "Available Knowledge Graphs",
    ["gpt-4o-mini", "mixtral8x7b", "mistral7b", "ministral-14b-2512", "mistral-large3", "llama3-8b-instruct"],
    format_func=lambda x: {
        "gpt-4o-mini": "GPT-4o-mini",
        "mixtral8x7b": "Mixtral 8x7B",
        "mistral7b": "Mistral 7B",
        "ministral-14b-2512": "Ministral 14B",
        "mistral-large3": "Mistral Large 3",
        "llama3-8b-instruct": "LLaMA3 8B",
    }[x],
)

if st.sidebar.button("Load Graph"):
    st.session_state["graph"] = load_graph(graph_choice)
    st.session_state["graph_name"] = graph_choice

model_names = {
    "openai_gpt4omini": "OpenAI – GPT-4o-mini",
#    "ministral-8b-2512": "Ministral 3 8B",
#    "ministral-14b-2512": "Ministral 14B",
#    "mistral-small-2506": "Mistral Small 3.2",
#    "mistral-large-2512": "Mistral Large",
    "llama-3.3-70b-versatile": "Groq - Llama 3.3 70B",
    "llama-3.1-8b-instant": "Groq - Llama 3.1 8B",
#    "groq/compound": "Groq - Compound",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}

selected_pretty = st.sidebar.selectbox(
    "Choose a language model:",
    list(model_names.values())
)

# reverse lookup from pretty to internal key
selected_model = {v: k for k, v in model_names.items()}[selected_pretty]

col1, col2 = st.sidebar.columns(2)
 
with col1:
    if st.button("Watch demo"):
        show_video()  # your existing function
 
with col2:
    if st.button("Changelog"):
        show_changelog()


# --- Main content ---
if "graph" in st.session_state:

    G = st.session_state["graph"]
    graph_name = st.session_state["graph_name"]

    st.subheader(f"📌 Selected graph: **{graph_name.upper()}**")    

    # Visualización
    st.info("Interaction: You can zoom, drag nodes, and click to see details.")
    graph_html = visualize_graph(G)
    components.html(graph_html, height=650, scrolling=True)

    # Métricas
    st.subheader("📊 Structural metrics")
    metrics = compute_metrics(G)

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Nodes", metrics["nodes"])
    col2.metric("Edges", metrics["edges"])
    col3.metric("Density", f"{metrics['density']:.4f}")
    col4.metric("Average degree", f"{metrics['avg_degree']:.2f}")
    col5.metric("Components", metrics["components"])

    if metrics["diameter"] is not None:
        col6.metric("Diameter", metrics["diameter"])
    else:
        st.warning("The diameter could not be calculated (graph too fragmented).")

    # Chat
    st.subheader("💬 Chat with the Selected Graph")

    question = st.text_input("Your question:")

    # -----------------------------------------------------------------------
    # CHANGES IN app.py — replace only the "if st.button('Ask'):" block
    # Everything else in app.py remains unchanged.
    # -----------------------------------------------------------------------
    # Add this import at the top of app.py (alongside the existing qa import):
    #   from utils.qa import run_qa, graph_to_triplets_text, verify_answer
    # -----------------------------------------------------------------------

    if st.button("Ask"):
        # 1. Load or build embeddings
        embeddings, node_list = load_graph_embeddings(graph_name)
        if embeddings is None:
            st.warning("Generating embeddings for the first time… this may take a few minutes.")
            embeddings, node_list = build_graph_embeddings(G, graph_name)

        # 2. Top-K retrieval
        top_nodes = get_top_k_nodes(question, embeddings, node_list)
        nodes_only = [n for n, _ in top_nodes]

        # 3. Subgraph extraction
        hops = choose_hops(question)
        subg = extract_subgraph(G, nodes_only, hops=hops)

        # 4. Answer generation
        answer = run_qa(question, subg, selected_model)

        st.markdown("### 🧠 Answer")
        st.write(answer)

        # 5. RDF triplets table
        st.markdown("### 📊 Knowledge Graph Triplets (RDF-style)")
        triplets = []
        for u, v, data in subg.edges(data=True):
            triplets.append({
                "Subject":   u,
                "Predicate": data.get("description", "related_to"),
                "Object":    v
            })
        df_triplets = pd.DataFrame(triplets)
        st.dataframe(df_triplets, width='stretch')

        # 6. Neuro-symbolic grounding verification
        st.markdown("### 🔍 Neuro-Symbolic Grounding Verification")

        with st.spinner("Verifying answer against knowledge graph…"):
            verification = verify_answer(answer, subg, selected_model)

        if verification["parse_error"]:
            st.warning(f"Could not parse verification output: {verification['parse_error']}")
            with st.expander("Raw verifier output"):
                st.code(verification["raw_json"])
        else:
            # --- Metrics summary ---
            total       = verification["total"]
            supported   = verification["supported"]
            partial     = verification["partial"]
            unsupported = verification["unsupported"]
            gf_strict   = verification["gf_strict"]
            gf_weighted = verification["gf_weighted"]

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total claims",  total)
            col2.metric("✅ Supported",  supported)
            col3.metric("⚠️ Partial",    partial)
            col4.metric("❌ Unsupported", unsupported)
            col5.metric("GF (weighted)", f"{gf_weighted:.2%}")

            # Secondary metric in smaller text
            st.caption(f"Grounding Fidelity — Strict: **{gf_strict:.2%}** · Weighted (partial = 0.5): **{gf_weighted:.2%}**")

            # --- Claims detail table ---
            st.markdown("#### Claim-level verification")

            STATUS_ICON = {
                "SUPPORTED":   "✅",
                "PARTIAL":     "⚠️",
                "UNSUPPORTED": "❌",
            }

            claims_rows = []
            for c in verification["claims"]:
                status = c.get("status", "UNSUPPORTED")
                evidence = c.get("evidence", [])
                claims_rows.append({
                    "Status":   f"{STATUS_ICON.get(status, '')} {status}",
                    "Claim":    c.get("claim", ""),
                    "Evidence": " | ".join(evidence) if evidence else "—",
                })

            df_claims = pd.DataFrame(claims_rows)
            st.dataframe(df_claims, width='stretch')


            if verification.get("explanation"):
                st.markdown("#### 🗒️ Grounding Summary")
                st.write(verification["explanation"])

else:
    st.info("Select a graph in the side panel and press 'Load Graph'.")