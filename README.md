# NLP Research Assistant

An advanced NLP tool for research document analysis with an optimized, aesthetic interface. Upload PDF papers and leverage state-of-the-art NLP models to summarize, search, extract insights, detect trends, and visualize research content.

![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

---

## 🚀 Features

- **📄 Paper Analysis** – Extract metadata, abstract, and full text from PDFs.
- **🤖 AI-Powered Summarization** – Abstractive (BART) and extractive (LexRank/LSA) summaries.
- **🔍 Semantic Search** – Find relevant papers and passages using sentence embeddings.
- **🏷️ Named Entity Recognition** – Identify people, organizations, locations, and more.
- **📊 Trend Detection** – Discover emerging topics across your document collection.
- **🎨 Interactive Visualizations** – Word clouds, knowledge graphs, and trend charts.
- **💡 Research Recommendations** – Get personalized suggestions based on content.
- **⚙️ Configurable Settings** – Choose models, chunk sizes, and performance options.
- **📤 Export Reports** – Generate comprehensive analysis reports in text format.

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment recommended

### Step-by-step

1. **Clone the repository**

```bash
git clone https://github.com/your-username/nlp-research-assistant.git
cd nlp-research-assistant
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download necessary NLTK and spaCy models (the app will attempt to download them automatically on first run, but you can pre‑download):**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
python -m spacy download en_core_web_sm
```

5. **Run the application**
```bash
streamlit run main.py
```
The app will open in your default browser at ```http://localhost:8501```.

---

## 📚 Usage
### Quick Start
1. **Upload PDFs** – Use the sidebar uploader or drag & drop files.
2. **Navigate** – Use the sidebar to switch between Dashboard, Paper Analyzer, Semantic Search, Trends Explorer, Visualizations, and Settings.
3. **Analyze a paper** – Go to Paper Analyzer, select a paper, and generate summaries, insights, or entity lists.
4. **Search semantically** – Enter a query in Semantic Search to find relevant papers and passages.
5. **Discover trends** – In Trends Explorer, analyze term frequencies and emerging topics across all loaded papers.
6. **Visualize** – Create word clouds, knowledge graphs, or comparison charts in the Visualizations tab.

### Configuration
- Adjust models, chunk sizes, and performance settings in the Settings page.
- All data is stored in memory; you can export/import state using the ```save_state/load_state``` functions or the Danger Zone options.

---

## 🧠 Technologies
| Component          | Library/Tool                                                                 |
|--------------------|------------------------------------------------------------------------------|
| Web Framework      | Streamlit                                                                    |
| NLP                | spaCy, NLTK, Transformers (Hugging Face), Sentence-Transformers, Sumy       |
| Embeddings & Search| Sentence‑Transformers, ChromaDB, scikit‑learn                                |
| Visualizations     | Matplotlib, Plotly, NetworkX, WordCloud                                     |
| PDF Processing     | PyPDF2, pdfplumber                                                          |
| Data Handling      | Pandas, NumPy                                                               |

---

## 📁 Project Structure
```
.
├── main.py                     # Main application code
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── research_results/           # Directory for saved results (auto‑created)
└── backup_*.json               # Optional exported state files
```
---
### 🔧 Requirements
A complete requirements.txt is provided in the files – ensure you install all packages.

---

Made with ❤️ for the research community
