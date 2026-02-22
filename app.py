import streamlit as st
from utils.load_graph import load_graph
from utils.visualize_graph import visualize_graph
from utils.compute_metrics import compute_metrics
from utils.embeddings import load_graph_embeddings, build_graph_embeddings
from utils.retrieve import get_top_k_nodes, extract_subgraph
from utils.qa import run_qa
import streamlit.components.v1 as components

st.set_page_config(page_title="HERMES – Graph Explorer", page_icon="🧭", layout="wide")

st.title("🧭 HERMES – Historical Ecology Retrieval & Multi-LLM Evaluation System.")
st.markdown("### **Step 1: Visualization and Interaction with Graphs**")

# --- Sidebar ---
st.sidebar.header("Select KG")
graph_choice = st.sidebar.selectbox(
    "Available Knowledge Graphs",
    ["gpt-4o-mini", "mixtral8x7b", "mistral7b", "llama3-8b-instruct"],
    format_func=lambda x: {
        "gpt-4o-mini": "GPT-4o-mini",
        "mixtral8x7b": "Mixtral 8x7B",
        "mistral7b": "Mistral 7B",
        "llama3-8b-instruct": "LLaMA3 8B",
    }[x],
)

if st.sidebar.button("Load Graph"):
    st.session_state["graph"] = load_graph(graph_choice)
    st.session_state["graph_name"] = graph_choice

model_names = {
    "openai_gpt4omini": "OpenAI – GPT-4o-mini",
    "ministral-8b-2512": "Ministral 3 8B",
    "mistral-small-2506": "Mistral Small 3.2"
}

selected_pretty = st.sidebar.selectbox(
    "Choose a language model:",
    list(model_names.values())
)

# reverse lookup from pretty to internal key
selected_model = {v: k for k, v in model_names.items()}[selected_pretty]


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

    if st.button("Ask"):
        # 1. Cargar o construir embeddings
        embeddings, node_list = load_graph_embeddings(graph_name)

        if embeddings is None:
            st.warning("Generating embeddings for the first time… this may take a few minutes.")
            embeddings, node_list = build_graph_embeddings(G, graph_name)

        # 2. Recuperación top-K
        top_nodes = get_top_k_nodes(question, embeddings, node_list, k=8)
        nodes_only = [n for n, _ in top_nodes]

        # 3. Subgrafo
        subg = extract_subgraph(G, nodes_only, hops=1)

        # 4. LLM
        answer = run_qa(question, subg, selected_model)

        st.markdown("### 🧠 Answer")
        st.write(answer)

        # 5. Mostrar nodos relevantes
        st.markdown("### 🔍 Recovered relevant nodes")
        for node, score in top_nodes:
            st.write(f"- {node} (sim={score:.3f})")

else:
    st.info("Select a graph in the side panel and press 'Load Graph'.")