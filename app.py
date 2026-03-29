import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import requests
import os
from sentence_transformers import CrossEncoder
import numpy as np

# --- Configuration ---
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama2:latest"
OLLAMA_API_URL = "http://localhost:11434/api"
CHROMA_COLLECTION_NAME = "bns_documents"
BATCH_SIZE = 100

hnsw_params = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 90,
    "hnsw:M": 16,
    "hnsw:search_ef": 50
}

st.set_page_config(
    page_title="🔍 Legal Search with Explanations",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
    <style>
        .main {background-color: #f9fafb;}
        div.stButton>button {width: 100%;}
        .result-header {
            background: #4a90e2;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
        }
        .small-label {color: #505050; font-size:0.95em;}
        .result-container {
            background: #fff;
            padding: 18px 24px;
            border-radius: 10px;
            margin-bottom: 16px;
            box-shadow: 0 0 4px #e8e8e8;
        }
        .explanation-box {
            background: #f5f6fa;
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 6px;
            color: #36414c;
            border: 1px solid #dde3ed;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Info ---
with st.sidebar:
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", width=64)
    st.header("Legal Search")
    st.write("""
    Search legal documents using:
    - Ollama Embeddings
    - ChromaDB (ANN)
    - Cross-Encoder Reranking
    - LLama2 LLM Explanations (on demand)
    """)
    st.markdown("---")


@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

@st.cache_resource(show_spinner=False)
def load_chroma_collection():
    try:
        ollama_ef = OllamaEmbeddingFunction(
            model_name=OLLAMA_EMBED_MODEL,
            url=f"{OLLAMA_API_URL}/embeddings"
        )
    except Exception as e:
        st.error(f"Error initializing Ollama embedding function: {e}")
        return None

    client = chromadb.Client()
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
        return collection
    except Exception:
        pass

    try:
        df = pd.read_csv("bns.csv")
        df['description'] = df['description'].fillna('')
    except FileNotFoundError:
        st.error("Local file 'bns.csv' not found in current directory.")
        return None

    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata=hnsw_params
    )

    total_docs = len(df)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_docs, BATCH_SIZE):
        batch_number = (i // BATCH_SIZE) + 1
        batch_df = df.iloc[i:i+BATCH_SIZE]
        batch_ids = [f"doc_{idx}" for idx in batch_df.index]
        batch_docs = batch_df['description'].tolist()
        batch_metas = batch_df[['act', 'section']].to_dict('records')

        status_text.text(f"Adding batch {batch_number} of {total_batches}...")
        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
        except Exception as e:
            st.error(f"Error adding batch {batch_number}: {e}")
            status_text.empty()
            progress_bar.empty()
            return None

        progress = batch_number / total_batches
        progress_bar.progress(progress)

    status_text.text("Completed embedding and indexing.")
    progress_bar.empty()
    return collection

def rerank_results(query, docs, metas, distances, reranker_model):
    rerank_n = len(docs)  # rerank all results returned for display
    pairs = [(query, doc) for doc in docs[:rerank_n]]
    scores = reranker_model.predict(pairs)
    ranked_indices = np.argsort(scores)[::-1]
    reranked_docs = [docs[i] for i in ranked_indices]
    reranked_metas = [metas[i] for i in ranked_indices]
    reranked_distances = [distances[i] for i in ranked_indices]
    reranked_scores = [scores[i] for i in ranked_indices]
    return reranked_docs, reranked_metas, reranked_distances, reranked_scores

def generate_relevance_explanation_llm(query_text: str, doc_text: str) -> str:
    prompt = (
        f"User query: \"{query_text}\"\n"
        f"Document excerpt: \"{doc_text}\"\n"
        "Explain concisely and in 2-3 sentences why this document excerpt could be relevant to the user query."
    )
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 150  # increased for fuller explanations
                }
            },
            timeout=60
        )
        response.raise_for_status()
        output = response.json()
        explanation = output.get("response", "").strip()
        return explanation if explanation else "No explanation generated."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def query_and_search(collection, query_text: str, k: int, reranker_model):
    try:
        results = collection.query(query_texts=[query_text], n_results=k)
    except Exception as e:
        st.error(f"Error querying collection: {e}")
        return None

    if not results or not results.get("ids") or not results["ids"][0]:
        return None

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if reranker_model:
        docs_rerank, metas_rerank, distances_rerank, scores_rerank = rerank_results(
            query_text, docs, metas, distances, reranker_model
        )
    else:
        docs_rerank = docs
        metas_rerank = metas
        distances_rerank = distances
        scores_rerank = [None] * len(docs)

    docs_final = docs_rerank[:k]
    metas_final = metas_rerank[:k]
    distances_final = distances_rerank[:k]
    rerank_scores_final = scores_rerank[:k]

    results_final = []
    for i in range(len(docs_final)):
        meta = metas_final[i]
        results_final.append({
            "act": meta.get("act", ""),
            "section": meta.get("section", ""),
            "description": docs_final[i],
            "distance": distances_final[i],
            "rerank_score": rerank_scores_final[i]
        })
    return results_final

# -- MAIN PAGE LOGIC START --

st.markdown("<div class='result-header'><h2>🔍 Legal Search with Explanations</h2></div>", unsafe_allow_html=True)
st.write("Efficiently locate relevant legal documents. Click 'Show Explanation' on any result to get an LLM-generated explanation.")

c1, c2 = st.columns([2, 1])
with c1:
    query_text = st.text_area(
        "Enter your legal search query:",
        height=120,
        placeholder="e.g., a person has committed suicide after murdering a person",
        key='main_query'
    )
with c2:
    top_k = st.slider(
        "Number of top results",
        min_value=1, max_value=10,
        value=5,
        key='main_top_k'
    )
    st.markdown("<span class='small-label'></span>", unsafe_allow_html=True)

with st.container():
    st.markdown(" ")
    search_clicked = st.button("🔍 Run Semantic Search", use_container_width=True, key="run_search_btn")

# Session state for results and explanations
if 'detailed_results' not in st.session_state:
    st.session_state['detailed_results'] = None
if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ""
if 'explanations' not in st.session_state:
    st.session_state['explanations'] = dict()

with st.spinner("Loading embeddings and building index, please wait..."):
    collection = load_chroma_collection()
reranker_model = load_reranker()

if collection is None:
    st.stop()

if search_clicked:
    if not query_text.strip():
        st.warning("Please enter a non-empty query.")
    else:
        with st.spinner("Searching and reranking..."):
            detailed_results = query_and_search(collection, query_text.strip(), top_k, reranker_model)
        st.session_state['detailed_results'] = detailed_results
        st.session_state['search_query'] = query_text.strip()
        st.session_state['explanations'] = dict()  # reset explanations on new search

if st.session_state['detailed_results']:
    st.subheader(f"🎯 Top {top_k} Re-Ranked Results:")
    for idx, res in enumerate(st.session_state['detailed_results']):
        exp_key = f"explanation_{idx}_{st.session_state['search_query']}"
        with st.container():
            st.markdown(
                f"<div class='result-container'>"
                f"<span style='font-size:1.05em;'><strong>Result #{idx + 1}</strong></span><br>"
                f"<span style='color:#666;'>Cosine Distance:</span> <b>{res['distance']:.4f}</b> &nbsp; | &nbsp; "
                + (f"<span style='color:#666;'>Rerank Score:</span> <b>{res['rerank_score']:.4f}</b>" if res['rerank_score'] is not None else "")
                + f"<br><span style='color:#38916e;'><b>Act:</b></span> {res['act']}  "
                f"&nbsp;&nbsp;<span style='color:#8c6feb;'><b>Section:</b></span> {res['section']}<br><br>"
                f"<b>Description:</b><br>{res['description']}<br><hr style='margin:8px 0;'>"
                "</div>",
                unsafe_allow_html=True
            )
            if exp_key not in st.session_state['explanations']:
                if st.button("Show Explanation", key=exp_key):
                    with st.spinner("Generating explanation..."):
                        expl = generate_relevance_explanation_llm(
                            st.session_state['search_query'], res["description"]
                        )
                        st.session_state['explanations'][exp_key] = expl
                    st.rerun()
            else:
                st.markdown(
                    f"<div class='explanation-box'><b>LLM Explanation:</b><br>{st.session_state['explanations'][exp_key]}</div>",
                    unsafe_allow_html=True
                )
elif search_clicked:
    st.warning("❌ No relevant results found for your query.")

st.markdown(
    """
    <hr>
    <div style='text-align:center; color:#888'>
        Powered by <b>Ollama</b>, <b>ChromaDB</b>, <b>Transformers</b>, and <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True
)
