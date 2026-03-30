# ⚖️ Nyaya AI — Bharatiya Nyaya Sanhita Legal Search

> Semantic legal search powered by OpenAI embeddings, cross-encoder re-ranking, and GPT-4o-mini explanations — built for common people, not just lawyers.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 What is Nyaya AI?

Nyaya AI lets anyone describe their legal situation in plain language and instantly finds the most relevant provisions from the **Bharatiya Nyaya Sanhita (BNS), 2023** — India's replacement for the Indian Penal Code (IPC), 1860.

Each result shows:
- The **BNS section** number and full provision text
- The **equivalent old IPC section** it replaced
- An optional **plain-English AI explanation** of why the law is relevant to your situation

No legal background required.

---

## 🧠 How It Works

```
User Query (plain language)
        │
        ▼
OpenAI text-embedding-3-small
(converts query to a vector)
        │
        ▼
ChromaDB Vector Store
(retrieves top 2× candidate provisions by cosine similarity)
        │
        ▼
Cross-Encoder Re-Ranker  (ms-marco-MiniLM-L-6-v2)
(re-scores candidates for precise relevance)
        │
        ▼
Top-K Results rendered with BNS § + IPC § badges
        │
        ▼  [optional]
GPT-4o-mini
(generates plain-English explanation of relevance)
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔎 Semantic Search | Understands meaning, not just keywords |
| 🏆 Re-ranking | Cross-encoder model boosts precision |
| 🏷️ Dual Section Badges | Shows both BNS § and old IPC § equivalent |
| 💬 AI Explanations | GPT-4o-mini explains relevance in plain language |
| 📱 Common-user UI | Designed for people with no legal background |
| ⚡ In-memory indexing | ChromaDB auto-indexes on first run, cached thereafter |
| 🔒 Privacy-first | Queries are never stored; API calls are per-search only |

---

## 🗂️ Project Structure

```
nyaya-ai/
├── app_openai.py        # Main Streamlit application
├── bns.csv              # BNS provisions dataset (you provide this)
├── .env                 # Environment variables (never commit this)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 📋 Requirements

- Python 3.9 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A `bns.csv` file with at minimum these columns:

| Column | Description |
|---|---|
| `description` | Full text of the legal provision |
| `act` | Name of the act (e.g. `The Bharatiya Nyaya Sanhita (BNS), 2023`) |
| `section` | BNS section number (e.g. `80`, `108`) |
| `ipc_section` *(optional)* | Equivalent old IPC section number |

> **IPC column auto-detection:** The app checks for these column names automatically:
> `ipc_section`, `ipc_equivalent`, `old_section`, `old_ipc`, `equivalent_ipc`, `ipc`, `corresponding_ipc`, `former_section`, `old_provision`, `ipc_provision` — or any column containing `"ipc"` in its name.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nyaya-ai.git
cd nyaya-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...your-key-here...
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 5. Add your dataset

Place your `bns.csv` file in the project root directory.

### 6. Run the app

```bash
streamlit run app_openai.py
```

The app will open at `http://localhost:8501`. On first run it will index all provisions into ChromaDB automatically — this may take a few minutes depending on dataset size and your OpenAI API rate limits.

---

## 📦 requirements.txt

```
streamlit>=1.32.0
openai>=1.0.0
chromadb>=0.5.0
pandas>=2.0.0
numpy>=1.24.0
sentence-transformers>=2.7.0
python-dotenv>=1.0.0
```

---

## ⚙️ Configuration

All key settings are at the top of `app_openai.py`:

```python
OPENAI_EMBED_MODEL    = "text-embedding-3-small"   # Embedding model
OPENAI_LLM_MODEL      = "gpt-4o-mini"              # Explanation model
CHROMA_COLLECTION_NAME = "bns_openai_v1"           # ChromaDB collection name
BATCH_SIZE            = 100                         # Docs per indexing batch
```

---

## 🔧 Troubleshooting

**`AttributeError: 'ChromaOpenAIEmbeddingFunction' object has no attribute 'embed_query'`**
You are running ChromaDB ≥ 0.6.x which changed the embedding function interface. This is already fixed in the current version — update from the repo.

**Results show raw HTML instead of formatted cards**
Legal texts contain special characters (`<`, `>`, `&`) that broke older versions. The current version uses `html.escape()` on all dynamic content — update from the repo.

**No IPC column shown on result cards**
Check that your CSV has an IPC-related column. See the auto-detection list in the Requirements section above. If your column name is different, add it to `IPC_COL_CANDIDATES` in `app_openai.py`.

**Slow indexing on first run**
This is normal — the app is calling the OpenAI Embeddings API for every provision in batches of 100. Subsequent runs reuse the cached ChromaDB collection and start instantly.

**`bns.csv` not found**
Place the CSV in the same directory as `app_openai.py` and restart the app.

---

## 🛡️ Privacy & Data

- User queries are **not stored** anywhere
- Each search makes a live API call to OpenAI
- ChromaDB runs **in-memory** — no data is written to disk
- No user analytics or tracking

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Bharatiya Nyaya Sanhita, 2023](https://legislative.gov.in/) — Ministry of Law and Justice, Government of India
- [OpenAI](https://openai.com/) — Embeddings and language model APIs
- [ChromaDB](https://www.trychroma.com/) — Open-source vector database
- [Sentence Transformers](https://www.sbert.net/) — Cross-encoder re-ranking model
- [Streamlit](https://streamlit.io/) — Web application framework

---

<p align="center">
  Made with ⚖️ for legal accessibility in India
</p>
