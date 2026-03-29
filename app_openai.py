import streamlit as st
import pandas as pd
import chromadb
import os
import html as html_mod
import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_LLM_MODEL = "gpt-4o-mini"
CHROMA_COLLECTION_NAME = "bns_openai_v1"
BATCH_SIZE = 100

# --- Validate API Key ---
if not OPENAI_API_KEY:
    st.error("🔑 Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# --- Initialize OpenAI Client ---
client_openai = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB-compatible embedding function
# Exposes __call__, embed_documents, and embed_query so both old and new
# ChromaDB versions work without AttributeError.
# ─────────────────────────────────────────────────────────────────────────────
class ChromaOpenAIEmbeddingFunction:
    def __init__(self, model_name: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def _embed(self, input):
        input = [t for t in input if t and t.strip()]
        if not input:
            return []
        response = self.client.embeddings.create(model=self.model_name, input=input)
        return [item.embedding for item in response.data]

    def __call__(self, input):
        return self._embed(input)

    def embed_documents(self, input):
        return self._embed(input)

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        return self._embed(input)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nyaya AI — BNS Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

    <style>
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0d0f14; color: #e8e6df; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

    .hero {
        display: flex; align-items: flex-end; gap: 18px;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid #2a2d36; padding-bottom: 1.5rem;
    }
    .hero-glyph { font-size: 3.2rem; line-height: 1; filter: drop-shadow(0 0 18px #c9a96e88); }
    .hero-text h1 {
        font-family: 'Playfair Display', serif; font-size: 2.6rem;
        font-weight: 900; color: #c9a96e; margin: 0; letter-spacing: -0.5px; line-height: 1;
    }
    .hero-text p {
        font-size: 0.85rem; color: #7a7d8a; margin: 6px 0 0 2px;
        letter-spacing: 0.06em; text-transform: uppercase;
    }

    .stTextArea textarea {
        background: #161820 !important; border: 1.5px solid #2a2d3a !important;
        border-radius: 10px !important; color: #e8e6df !important;
        font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
        padding: 14px 16px !important; transition: border-color 0.2s; caret-color: #c9a96e;
    }
    .stTextArea textarea:focus { border-color: #c9a96e !important; box-shadow: 0 0 0 3px #c9a96e22 !important; }
    .stTextArea label { color: #9a9caa !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

    div.stButton > button {
        background: linear-gradient(135deg, #c9a96e 0%, #a07840 100%) !important;
        color: #0d0f14 !important; font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important; font-size: 1rem !important;
        border: none !important; border-radius: 8px !important;
        padding: 0.65rem 2rem !important; transition: opacity 0.2s, transform 0.1s !important;
        width: 100%;
    }
    div.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
    div.stButton > button:active { transform: translateY(0px) !important; }

    .stSlider > div > div > div > div { background: #c9a96e !important; }
    .stSlider label { color: #9a9caa !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

    .result-card {
        background: #161820; border: 1px solid #252830;
        border-left: 4px solid #c9a96e; border-radius: 10px;
        padding: 1.2rem 1.5rem; margin-bottom: 0.5rem;
        transition: box-shadow 0.2s;
    }
    .result-card:hover { box-shadow: 0 4px 24px #c9a96e18; }
    .result-meta { display: flex; align-items: center; gap: 10px; margin-bottom: 0.6rem; }
    .badge-act {
        background: #c9a96e22; color: #c9a96e; border: 1px solid #c9a96e44;
        border-radius: 5px; padding: 2px 10px; font-size: 0.72rem;
        font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
    }
    .badge-section {
        background: #1e2230; color: #7a8aaa; border-radius: 5px;
        padding: 2px 10px; font-size: 0.72rem; letter-spacing: 0.05em;
    }
    .badge-ipc {
        background: #1a2235; color: #7ab0e0; border: 1px solid #2a4a7a;
        border-radius: 5px; padding: 2px 10px; font-size: 0.72rem;
        letter-spacing: 0.05em; white-space: nowrap;
    }
    .badge-ipc-label {
        font-size: 0.62rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #4a7aaa; margin-right: 3px;
    }
    .score-pill {
        margin-left: auto; background: #0d0f14;
        border: 1px solid #2a2d36; border-radius: 20px;
        padding: 2px 12px; font-size: 0.72rem;
    }
    .result-text { font-size: 0.95rem; line-height: 1.75; color: #c8c6be; margin-top: 0.5rem; }

    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.05rem; font-weight: 700;
        color: #e8e6df; margin: 0.6rem 0 0.4rem 0;
        letter-spacing: 0.01em; border-bottom: 1px solid #252830;
        padding-bottom: 0.4rem;
    }

    .explain-box {
        background: #1a1f14; border: 1px solid #3a4a2a; border-radius: 8px;
        padding: 1rem 1.2rem; margin-top: 0.5rem; margin-bottom: 0.8rem;
    }
    .explain-label {
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: #6a8a4a; margin-bottom: 8px; font-weight: 600;
    }
    .explain-body {
        color: #b8d898; font-size: 0.93rem; line-height: 1.75;
    }

    .no-results { text-align: center; padding: 3rem 0; color: #4a4d5a; }
    .no-results .icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .no-results p { font-size: 0.9rem; }

    [data-testid="stSidebar"] { background: #0a0c10 !important; border-right: 1px solid #1e2028; }
    [data-testid="stSidebar"] .stMarkdown h3 { color: #c9a96e; font-family: 'Playfair Display', serif; font-size: 1.1rem; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li { color: #7a7d8a; font-size: 0.84rem; line-height: 1.6; }
    .sidebar-stat {
        background: #161820; border: 1px solid #252830; border-radius: 8px;
        padding: 0.8rem 1rem; margin: 0.5rem 0; text-align: center;
    }
    .sidebar-stat .stat-value { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: #c9a96e; }
    .sidebar-stat .stat-label { font-size: 0.7rem; color: #4a4d5a; text-transform: uppercase; letter-spacing: 0.08em; }

    hr { border: none; border-top: 1px solid #1e2028; margin: 1.5rem 0; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #c9a96e, #a07840) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Nyaya AI")
    st.markdown("---")
    st.markdown(
        """
        **Bharatiya Nyaya Sanhita**
        AI-powered semantic search across BNS provisions using:

        - 🔎 **OpenAI** text-embedding-3-small
        - 🏆 **Cross-Encoder** re-ranking (ms-marco)
        - 💬 **GPT-4o-mini** relevance explanations
        - 🗄️ **ChromaDB** vector store (in-memory)
        """
    )
    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-stat">
            <div class="stat-value">BNS</div>
            <div class="stat-label">Active Corpus</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#4a4d5a; font-size:0.75rem;'>Queries are not stored. "
        "API calls are made per search.</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


@st.cache_resource(show_spinner=False)
def get_openai_ef():
    return ChromaOpenAIEmbeddingFunction(model_name=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def load_chroma_collection():
    openai_ef = get_openai_ef()
    chroma_client = chromadb.Client()

    try:
        collection = chroma_client.get_collection(
            name=CHROMA_COLLECTION_NAME, embedding_function=openai_ef
        )
        if collection.count() > 0:
            return collection
    except Exception:
        pass

    try:
        df = pd.read_csv("bns.csv")
        df["description"] = df["description"].fillna("").astype(str)
    except FileNotFoundError:
        st.error("❌ 'bns.csv' not found. Please place it in the project root.")
        return None

    # Auto-detect the IPC equivalent column (handles different naming conventions)
    IPC_COL_CANDIDATES = [
        "ipc_section", "ipc_equivalent", "old_section", "old_ipc",
        "equivalent_ipc", "ipc", "corresponding_ipc", "former_section",
        "old_provision", "ipc_provision",
    ]
    ipc_col = next(
        (c for c in IPC_COL_CANDIDATES if c in df.columns),
        None
    )
    # Fallback: any column whose name contains "ipc"
    if ipc_col is None:
        ipc_col = next((c for c in df.columns if "ipc" in c.lower()), None)

    meta_cols = ["act", "section"] + ([ipc_col] if ipc_col else [])

    collection = chroma_client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    total_docs = len(df)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_docs, BATCH_SIZE):
        batch_number = (i // BATCH_SIZE) + 1
        batch_df = df.iloc[i : i + BATCH_SIZE]
        raw = list(
            zip(
                batch_df["description"].tolist(),
                batch_df[meta_cols].fillna("").astype(str).to_dict("records"),
                [f"doc_{idx}" for idx in batch_df.index],
            )
        )
        filtered = [(doc.strip(), meta, id_) for doc, meta, id_ in raw if doc and doc.strip()]
        if not filtered:
            continue
        batch_docs, batch_metas, batch_ids = zip(*filtered)
        collection.add(
            documents=list(batch_docs),
            metadatas=list(batch_metas),
            ids=list(batch_ids),
        )
        progress_bar.progress(batch_number / total_batches)
        status_text.text(f"⏳ Indexing batch {batch_number} / {total_batches} …")

    status_text.empty()
    progress_bar.empty()
    return collection


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def rerank_results(query, docs, metas, distances, reranker_model):
    pairs = [(query, doc) for doc in docs]
    scores = reranker_model.predict(pairs)
    ranked = np.argsort(scores)[::-1]
    return (
        [docs[i] for i in ranked],
        [metas[i] for i in ranked],
        [distances[i] for i in ranked],
        [scores[i] for i in ranked],
    )


def query_and_search(collection, query_text, k, reranker_model):
    results = collection.query(query_texts=[query_text], n_results=k * 2)
    if not results or not results["ids"][0]:
        return None
    docs, metas, dists, scores = rerank_results(
        query_text,
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        reranker_model,
    )
    # Detect which IPC key was stored (mirrors the auto-detect in load_chroma_collection)
    IPC_COL_CANDIDATES = [
        "ipc_section", "ipc_equivalent", "old_section", "old_ipc",
        "equivalent_ipc", "ipc", "corresponding_ipc", "former_section",
        "old_provision", "ipc_provision",
    ]
    sample_meta = metas[0] if metas else {}
    ipc_key = next((k for k in IPC_COL_CANDIDATES if k in sample_meta), None)
    if ipc_key is None:
        ipc_key = next((k for k in sample_meta if "ipc" in k.lower()), None)

    return [
        {
            "act": metas[i].get("act", "Unknown Act"),
            "section": metas[i].get("section", "—"),
            "ipc_section": metas[i].get(ipc_key, "") if ipc_key else "",
            "description": docs[i],
            "rerank_score": float(scores[i]),
        }
        for i in range(min(k, len(docs)))
    ]


def generate_explanation(query_text: str, doc_text: str) -> str:
    try:
        response = client_openai.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior Indian legal expert specialising in the Bharatiya Nyaya Sanhita. "
                        "Given a search query and a legal provision, explain in 2–3 concise sentences "
                        "exactly why this provision is relevant to the query. Be precise and practical."
                    ),
                },
                {"role": "user", "content": f"Query: {query_text}\n\nProvision: {doc_text}"},
            ],
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate explanation: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# Results and explanations are stored here so they survive the Streamlit rerun
# that fires when an "Explain" button is clicked.
# ─────────────────────────────────────────────────────────────────────────────
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "explanations" not in st.session_state:
    st.session_state.explanations = {}   # {result_index: explanation_text}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-glyph">⚖️</div>
        <div class="hero-text">
            <h1>Nyaya AI</h1>
            <p>Bharatiya Nyaya Sanhita · Semantic Legal Intelligence</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading re-ranker model …"):
    reranker = load_reranker()

with st.spinner("Connecting to vector store …"):
    collection = load_chroma_collection()

if collection is None:
    st.stop()

# Search controls
col_query, col_k = st.columns([4, 1], gap="large")

with col_query:
    query_input = st.text_area(
        "Your Legal Query",
        placeholder="e.g.  Familial disputed land and sisters asking for their share in generational properties …",
        height=110,
    )

with col_k:
    top_k = st.slider("Results", min_value=1, max_value=10, value=5)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Search")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Trigger search ───────────────────────────────────────────────────────────
if search_clicked:
    q = query_input.strip()
    if not q:
        st.warning("Please enter a query before searching.")
    else:
        with st.spinner("Searching and re-ranking …"):
            results = query_and_search(collection, q, top_k, reranker)
        st.session_state.search_results = results or []
        st.session_state.last_query = q
        st.session_state.explanations = {}   # clear old explanations for new query

# ── Render results ───────────────────────────────────────────────────────────
# Runs on every rerun (including when Explain is clicked) so results persist.
results     = st.session_state.search_results
current_query = st.session_state.last_query

if results:
    st.markdown(
        f"<p style='color:#7a7d8a; font-size:0.82rem; margin-bottom:1.2rem;'>"
        f"Found <strong style='color:#c9a96e'>{len(results)}</strong> relevant provisions</p>",
        unsafe_allow_html=True,
    )

    for idx, res in enumerate(results):
        score     = res["rerank_score"]
        ipc_val   = (res.get("ipc_section") or "").strip()
        show_ipc  = ipc_val and ipc_val.lower() not in ("nan", "—", "", "none")

        # ── Score colour helper ──
        score_color = "#9dc97a" if score > 5 else ("#c9a96e" if score > 0 else "#e06c75")

        # ── Card wrapper (static HTML — no dynamic content injected) ──────────
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # ── Badge row: all dynamic text rendered via st.markdown (auto-escaped) ──
        badge_cols = st.columns([3, 1.2, 1.5, 2])

        with badge_cols[0]:
            # Act name — plain text, Streamlit escapes it
            act_safe = res["act"]
            st.markdown(
                f"<span class='badge-act'>{html_mod.escape(act_safe)}</span>",
                unsafe_allow_html=True,
            )

        with badge_cols[1]:
            st.markdown(
                f"<span class='badge-section'>BNS &sect;&nbsp;{html_mod.escape(str(res['section']))}</span>",
                unsafe_allow_html=True,
            )

        with badge_cols[2]:
            if show_ipc:
                st.markdown(
                    f"<span class='badge-ipc'>"
                    f"<span class='badge-ipc-label'>IPC</span>"
                    f"&sect;&nbsp;{html_mod.escape(ipc_val)}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='badge-ipc' style='opacity:0.3'>No IPC equiv.</span>",
                    unsafe_allow_html=True,
                )

        with badge_cols[3]:
            st.markdown(
                f"<div style='text-align:right'>"
                f"<span class='score-pill' style='color:{score_color}'>▲ Match {score:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Section heading ────────────────────────────────────────────────────
        section_title = f"{res['act']} — Section {res['section']}"
        if show_ipc:
            section_title += f"  (Earlier: IPC § {ipc_val})"
        st.markdown(
            f"<div class='section-title'>{html_mod.escape(section_title)}</div>",
            unsafe_allow_html=True,
        )

        # ── Full description text (plain Streamlit — never breaks on special chars) ──
        st.markdown(
            f"<div class='result-text'>{html_mod.escape(res['description'])}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)   # close result-card

        # ── Explain button ─────────────────────────────────────────────────────
        has_explanation = idx in st.session_state.explanations
        btn_label = "🔄 Regenerate AI explanation" if has_explanation else "✦ What does this mean for my case?"

        if st.button(btn_label, key=f"explain_btn_{idx}"):
            with st.spinner("Generating plain-English explanation …"):
                st.session_state.explanations[idx] = generate_explanation(
                    current_query, res["description"]
                )
            st.rerun()

        # ── Explanation box ────────────────────────────────────────────────────
        if has_explanation:
            st.markdown(
                f"""
                <div class="explain-box">
                    <div class="explain-label">⚖️ Plain-English Explanation</div>
                    <div class="explain-body">{html_mod.escape(st.session_state.explanations[idx])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

elif current_query and not results:
    st.markdown(
        """
        <div class="no-results">
            <div class="icon">🔎</div>
            <p>No matching provisions found. Try describing your situation differently.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="no-results">
            <div class="icon">📜</div>
            <p>Describe your legal situation above in plain language and press <strong>Search</strong>.</p>
            <p style="font-size:0.78rem; margin-top:8px; color:#3a3d4a;">
                Example: "My landlord is refusing to return the security deposit after I vacated the house"
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )