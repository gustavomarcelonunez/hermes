# HERMES  
**Historical Ecology Retrieval & Multi-LLM Evaluation System**

HERMES is an interactive system designed to explore, analyze, and query knowledge graphs generated from historical scientific literature using multiple Large Language Models (LLMs).  
It provides:

- Interactive visualization of 4 pre-generated knowledge graphs  
- Structural graph metrics  
- Node-level inspection  
- A semantic chatbot that answers questions using relevant subgraphs  
- A modular architecture designed for future ontology mapping and multi-model comparison  

---

## 🚀 Features (Stage 1 & Stage 2 implemented)

### ✔ Interactive Graph Visualization
- Select one of four knowledge graphs  
- Explore nodes and edges using a PyVis network view  
- Click nodes to display metadata  
- Structural metrics displayed automatically:
  - Nodes  
  - Edges  
  - Density  
  - Average degree  
  - Connected components  
  - Diameter  

### ✔ Knowledge Graph Chatbot (Stage 2)
- Ask natural language questions  
- The system retrieves the most relevant nodes using embeddings  
- Extracts a local subgraph  
- Sends only the relevant context to an LLM (GPT-4o-mini by default)  
- Ensures efficient token usage and avoids hallucinations  

---

## 📂 Project Structure
    HERMES/
    │
    ├── app.py
    ├── data/ # 4 GraphRAG-generated knowledge graphs
    ├── embeddings/ # Node embeddings (saved for reproducibility)
    ├── utils/
    │ ├── load_graph.py
    │ ├── visualize_graph.py
    │ ├── compute_metrics.py
    │ ├── embeddings.py
    │ ├── retrieve.py
    │ └── qa.py
    └── README.md

---

## 🔧 Installation

python -m venv hermes_env
source hermes_env/bin/activate
pip install -r requirements.txt

Create a `.env` file with:
    OPENAI_API_KEY=your_key_here

---

## ▶️ Run
streamlit run app.py

---

## 📄 License
MIT License.