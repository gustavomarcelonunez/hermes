# HERMES
**Historical Ecology Retrieval & Multi-LLM Evaluation System**

HERMES is a web-based platform for natural-language querying and comparative evaluation of marine knowledge graphs generated from historical scientific literature using Microsoft GraphRAG. It supports multi-graph, multi-LLM evaluation with a built-in neuro-symbolic grounding verifier.

🌐 **Live system:** https://hermes-project.streamlit.app

---

## 📖 About

HERMES was developed as part of a doctoral research project on neuro-symbolic AI for marine science data. The system is built around six knowledge graphs derived from a digitized 19th-century marine biology corpus (*Lobos marinos, pingüinos y guaneras de las costas del litoral marítimo e islas adyacentes de la República Argentina*, Carrara, 1952), each constructed using a different LLM under a shared GraphRAG prompt configuration.

The platform is described in detail in the following paper (under review):

> Nuñez, G., Zárate, M., Fillottrani, P. *HERMES: Grounding Fidelity Evaluation in GraphRAG-Based Marine Knowledge Graphs Across LLM Configurations*. RAGE-KG 2026, co-located with ISWC 2026.

---

## ✨ Features

### 🔍 Knowledge Graph Selection
- Six pre-generated knowledge graphs, each constructed with a different LLM:
  - GPT-4o-mini, Mixtral 8×7B, Mistral 7B, Ministral 14B, Mistral Large 3, LLaMA 3 8B
- Interactive graph visualization (zoom, drag, node inspection)
- Structural metrics panel per selected graph: nodes, edges, density, average degree, connected components, diameter

### 💬 Multi-LLM Question Answering
- Natural language Q&A grounded in retrieved subgraph triples
- Four answer-generation LLMs available:
  - GPT-4o-mini (OpenAI), Llama 3.3 70B, Llama 3.1 8B (via Groq), Gemini 2.5 Flash
- Embedding-based node retrieval with structural subgraph expansion (1–3 hops)
- Retrieved triples displayed in Subject / Predicate / Object format

### 🔬 Neuro-Symbolic Grounding Verifier
- Post-generation verification of response claims against retrieved subgraph triples
- Each claim classified as SUPPORTED, PARTIAL, or UNSUPPORTED
- Two novel metrics computed automatically:
  - **GF_strict**: proportion of fully supported claims
  - **GF_weighted**: partial credit for partially supported claims (weight = 0.5)
- Claim-level verification table with cited triple evidence
- Grounding Summary: natural-language explanation of overall grounding quality

---

## 📂 Project Structure

```
HERMES/
│
├── app.py                      # Main Streamlit application
├── data/                       # 6 GraphRAG-generated KGs (.graphml)
├── embeddings/                 # Pre-computed node embeddings
├── supplementary_material/     # Evaluation CSVs and sensitivity analysis results
├── media/                      # Demo video
├── utils/
│   ├── load_graph.py
│   ├── models.py
│   ├── visualize_graph.py
│   ├── compute_metrics.py
│   ├── embeddings.py
│   ├── retrieve.py
│   ├── qa.py
│   └── video_popup.py
└── README.md
```

---

## 🔧 Installation

```bash
python -m venv hermes_env
source hermes_env/bin/activate  # On Windows: hermes_env\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

---

## ▶️ Run

```bash
streamlit run app.py
```

---

## 📊 Supplementary Material

The `supplementary_material/` folder contains:
- `evaluation_results.csv`: Full factorial evaluation results (235 valid cells out of 240)
- `sensitivity_results.csv`: Threshold sensitivity analysis (English queries)
- `sensitivity_results_es.csv`: Threshold sensitivity analysis (Spanish queries)

---

## 📄 License

MIT License.
