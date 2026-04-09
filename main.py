"""
NLP Research Assistant - Professional UI Version
Advanced NLP tool for research document analysis with optimized, aesthetic interface
"""

# ==================== IMPORTS ====================
import os, re, json, warnings, hashlib, sys
import base64, tempfile, time, io
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2, pdfplumber
import spacy, nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from wordcloud import WordCloud
# FIX: Import directly instead of using pipeline for summarization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
import chromadb

# Try to import streamlit
try:
    import streamlit as st
    from streamlit.components.v1 import html
    STREAMLIT_AVAILABLE = True
except ImportError:
    print("Streamlit not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    import streamlit as st
    from streamlit.components.v1 import html
    STREAMLIT_AVAILABLE = True

# Try to import psutil for system info (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ==================== CUSTOM CSS & STYLING ====================
CUSTOM_CSS = """
<style>
    /* ── CSS custom properties: light defaults, overridden by .theme-dark ── */
    :root {
        --card-bg:        #ffffff;
        --card-border:    #e2e8f0;
        --card-text:      #2d3748;
        --card-subtext:   #718096;
        --page-bg:        #f8fafc;
        --page-bg2:       #edf2f7;
        --input-bg:       #f8fafc;
        --summary-bg:     #f8fafc;
        --summary-text:   #2d3748;
        --passage-bg:     #f8fafc;
        --header-border:  #e2e8f0;
        --sub-header-clr: #4a5568;
        --tag-bg:         #e2e8f0;
        --tag-text:       #4a5568;
        --entity-bg:      #ffffff;
        --rec-bg:         #f8fafc;
        --activity-border:#e2e8f0;
        --metric-bg:      #ffffff;
        --metric-text:    #2d3748;
        --metric-sub:     #718096;
        --search-card-bg: #ffffff;
        --trend-card-bg:  #f8fafc;
        --paper-meta-bg:  #f8fafc;
    }

    /* ── Dark theme overrides (injected by Python when user picks Dark) ── */
    .theme-dark, .theme-dark * {
        --card-bg:        #1e2130;
        --card-border:    #3a3f5c;
        --card-text:      #e2e8f0;
        --card-subtext:   #a0aec0;
        --page-bg:        #161824;
        --page-bg2:       #1a1d2e;
        --input-bg:       #252840;
        --summary-bg:     #1e2130;
        --summary-text:   #e2e8f0;
        --passage-bg:     #252840;
        --header-border:  #3a3f5c;
        --sub-header-clr: #a0aec0;
        --tag-bg:         #3a3f5c;
        --tag-text:       #e2e8f0;
        --entity-bg:      #1e2130;
        --rec-bg:         #1a1d2e;
        --activity-border:#3a3f5c;
        --metric-bg:      #1e2130;
        --metric-text:    #e2e8f0;
        --metric-sub:     #a0aec0;
        --search-card-bg: #1e2130;
        --trend-card-bg:  #1a1d2e;
        --paper-meta-bg:  #1a1d2e;
    }

    /* ── Detect system dark mode automatically when theme = System ── */
    @media (prefers-color-scheme: dark) {
        .theme-system, .theme-system * {
            --card-bg:        #1e2130;
            --card-border:    #3a3f5c;
            --card-text:      #e2e8f0;
            --card-subtext:   #a0aec0;
            --page-bg:        #161824;
            --page-bg2:       #1a1d2e;
            --input-bg:       #252840;
            --summary-bg:     #1e2130;
            --summary-text:   #e2e8f0;
            --passage-bg:     #252840;
            --header-border:  #3a3f5c;
            --sub-header-clr: #a0aec0;
            --tag-bg:         #3a3f5c;
            --tag-text:       #e2e8f0;
            --entity-bg:      #1e2130;
            --rec-bg:         #1a1d2e;
            --activity-border:#3a3f5c;
            --metric-bg:      #1e2130;
            --metric-text:    #e2e8f0;
            --metric-sub:     #a0aec0;
            --search-card-bg: #1e2130;
            --trend-card-bg:  #1a1d2e;
            --paper-meta-bg:  #1a1d2e;
        }
    }

    .main { padding: 0rem 1rem; }

    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem; font-weight: 600;
        color: var(--sub-header-clr);
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--header-border);
        padding-bottom: 0.5rem;
    }

    /* Cards */
    .stCard {
        background: var(--card-bg); color: var(--card-text);
        border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid var(--card-border); margin-bottom: 1rem;
    }
    .paper-card {
        background: var(--card-bg); color: var(--card-text);
        border-radius: 10px; padding: 1.2rem;
        border-left: 4px solid #667eea; transition: transform 0.2s;
    }
    .paper-card strong { color: var(--card-text); }
    .paper-card small  { color: var(--card-subtext); }
    .paper-card:hover  { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }

    /* Summary */
    .summary-box {
        background: var(--summary-bg); color: var(--summary-text) !important;
        border-radius: 10px; padding: 1.5rem;
        border: 1px solid var(--card-border); margin: 1rem 0;
        font-size: 1rem; line-height: 1.6;
        max-height: 400px; overflow-y: auto; width: 100%;
    }
    .summary-box h4 { color: var(--card-text) !important; margin-top: 0; }
    .summary-box p, .summary-box span, .summary-box div { color: var(--summary-text) !important; }

    /* Metric cards */
    .metric-card {
        background: var(--metric-bg); color: var(--metric-text);
        border-radius: 10px; padding: 1rem;
        border: 1px solid var(--card-border); text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.875rem; color: var(--metric-sub); text-transform: uppercase; letter-spacing: 0.05em; }

    /* Tags */
    .tag         { display: inline-block; background: var(--tag-bg); color: var(--tag-text); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem; margin: 0.25rem; }
    .tag-primary { background: #667eea; color: #ffffff !important; }
    .tag-success { background: #48bb78; color: #ffffff !important; }
    .tag-warning { background: #ed8936; color: #ffffff !important; }

    /* Buttons */
    .stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.3s; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }

    /* Search */
    .search-result {
        background: var(--search-card-bg); color: var(--card-text);
        border-radius: 10px; padding: 1rem; margin: 1rem 0;
        border-left: 4px solid #4299e1; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Misc */
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
    .dataframe { border-radius: 8px; overflow: hidden; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track  { background: #f1f1f1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb  { background: #c1c1c1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #a1a1a1; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 10px 20px; font-weight: 600; }

    /* Alerts */
    .success-message { background: linear-gradient(135deg,#c6f6d5,#9ae6b4); border:1px solid #48bb78; border-radius:8px; padding:1rem; color:#22543d; }
    .info-message    { background: linear-gradient(135deg,#bee3f8,#90cdf4); border:1px solid #4299e1; border-radius:8px; padding:1rem; color:#2c5282; }
    .warning-message { background: linear-gradient(135deg,#feebc8,#fbd38d); border:1px solid #ed8936; border-radius:8px; padding:1rem; color:#9c4221; }
</style>
"""

# ==================== THEME HELPERS ====================
def get_theme_class() -> str:
    """Return CSS class that activates the correct theme variables."""
    t = st.session_state.get('app_theme', 'Light')
    return {'Light': 'theme-light', 'Dark': 'theme-dark', 'System': 'theme-system'}.get(t, 'theme-light')

def themed(html: str) -> str:
    """Wrap inline HTML in a div carrying the active theme class."""
    return f"<div class='{get_theme_class()}'>{html}</div>"

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    """Configuration settings for the NLP Research Assistant"""
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"
    NER_MODEL: str = "en_core_web_sm"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_SUMMARY_LENGTH: int = 200
    MIN_SENTENCE_LENGTH: int = 20
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    TREND_WINDOW_DAYS: int = 30
    TOP_TREND_TERMS: int = 10
    ENABLE_VISUALIZATIONS: bool = True
    SAVE_RESULTS: bool = True
    RESULTS_DIR: str = "research_results"
    THEME: str = "light"

# ==================== DATA STRUCTURES ====================
@dataclass
class ResearchPaper:
    """Represents a single research paper/document"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    content: str
    source_file: str
    publication_date: Optional[str] = None
    keywords: List[str] = None
    embeddings: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

    def to_dict(self):
        d = asdict(self)
        # embeddings are numpy arrays, not JSON serializable
        if d.get('embeddings') is not None:
            d['embeddings'] = None
        return d

@dataclass
class SearchResult:
    paper_id: str
    title: str
    similarity_score: float
    relevant_passages: List[str]
    metadata: Dict[str, Any]

@dataclass
class AnalysisResult:
    summary: str
    key_terms: List[str]
    entities: Dict[str, List[str]]
    sentiment: Dict[str, float]
    trends: Dict[str, Any]
    recommendations: List[str]

# ==================== CORE NLP CLASS ====================
class NLPResearchAssistant:
    """Main class for NLP Research Assistant functionality"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.papers: Dict[str, ResearchPaper] = {}
        self.vector_db = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self._init_session_state()
        if self.config.SAVE_RESULTS and not os.path.exists(self.config.RESULTS_DIR):
            os.makedirs(self.config.RESULTS_DIR)

    def _init_session_state(self):
        defaults = {
            'papers_loaded': False,
            'current_paper': None,
            'search_results': [],
            'insights': {},
            'trends': {},
            'summary_generated': False,
            'current_summary': "",
            'app_theme': 'Light',
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ==================== FIX: REVISED NLP INITIALIZATION ====================
    def _initialize_nlp_components(self):
        """Initialize all NLP models with robust error handling.
        
        KEY FIX: Newer transformers versions removed 'summarization' from pipeline tasks.
        We now load the model directly via AutoModelForSeq2SeqLM instead of using pipeline().
        """
        with st.spinner("🔄 Initializing NLP components..."):

            # --- NLTK data ---
            self._download_nltk_data()

            # --- Embedding model ---
            st.info("📥 Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            st.success("✅ Embedding model loaded.")

            # --- Summarization: direct model load (no pipeline) ---
            st.info("📝 Loading summarization model (this may take a moment)...")
            self._summarization_model = None
            self._summarization_tokenizer = None

            try:
                self._summarization_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.SUMMARIZATION_MODEL
                )
                self._summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.SUMMARIZATION_MODEL
                )
                st.success("✅ Summarization model loaded.")
            except Exception as e:
                st.warning(
                    f"⚠️ Could not load neural summarization model: {str(e)[:80]}. "
                    "Falling back to extractive summarization (sumy)."
                )
                self._summarization_model = None
                self._summarization_tokenizer = None

            # Attach summarize_text as an instance method
            def _summarize(text: str, max_length: int = 150) -> str:
                # Try neural model first
                if self._summarization_model and self._summarization_tokenizer:
                    try:
                        inputs = self._summarization_tokenizer(
                            text[:4000],
                            return_tensors="pt",
                            truncation=True,
                            max_length=1024,
                        )
                        summary_ids = self._summarization_model.generate(
                            inputs["input_ids"],
                            max_length=max_length,
                            min_length=50,
                            num_beams=4,
                            early_stopping=True,
                        )
                        return self._summarization_tokenizer.decode(
                            summary_ids[0], skip_special_tokens=True
                        )
                    except Exception as e:
                        st.warning(f"Neural summarization failed, using extractive: {e}")

                # Extractive fallback via sumy
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                try:
                    summarizer = LexRankSummarizer()
                except Exception:
                    summarizer = LsaSummarizer()
                sentences = summarizer(parser.document, sentences_count=5)
                return " ".join(str(s) for s in sentences)

            self.summarize_text = _summarize

            # --- spaCy NER ---
            st.info("🏷️ Loading spaCy model...")
            try:
                self.nlp = spacy.load(self.config.NER_MODEL)
            except OSError:
                st.warning(f"Downloading spaCy model: {self.config.NER_MODEL}")
                spacy.cli.download(self.config.NER_MODEL)
                self.nlp = spacy.load(self.config.NER_MODEL)
            st.success("✅ spaCy model loaded.")

            # --- NLTK components ---
            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()

            # --- ChromaDB ---
            self._initialize_vector_db()

            st.success("✅ All NLP components initialized successfully!")

    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        required = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
        progress_bar = st.progress(0)
        status = st.empty()
        for idx, pkg in enumerate(required):
            status.text(f"📥 Checking NLTK data: {pkg}")
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                st.warning(f"Could not download NLTK package '{pkg}': {e}")
            progress_bar.progress((idx + 1) / len(required))
        progress_bar.empty()
        status.empty()

    def _initialize_vector_db(self):
        self.vector_db = chromadb.EphemeralClient()
        self.collection = self.vector_db.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"},
        )

    # ==================== DOCUMENT PROCESSING ====================

    def load_pdf(self, file_path: str) -> Optional[ResearchPaper]:
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return None

        with st.spinner(f"📄 Processing PDF: {os.path.basename(file_path)}..."):
            try:
                text = ""
                metadata = {}

                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if pdf.pages:
                        metadata = self._extract_metadata(pdf.pages[0].extract_text() or "")

                if not text.strip():
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            text += (page.extract_text() or "") + "\n"

                if not text.strip():
                    st.error("❌ No text could be extracted from PDF.")
                    return None

                file_hash = hashlib.md5(text.encode()).hexdigest()[:10]
                paper_id = f"paper_{file_hash}"

                paper = ResearchPaper(
                    id=paper_id,
                    title=metadata.get("title", os.path.basename(file_path)),
                    authors=metadata.get("authors", []),
                    abstract=self._extract_abstract(text),
                    content=text,
                    source_file=file_path,
                    publication_date=metadata.get("publication_date"),
                    keywords=metadata.get("keywords", []),
                )

                self.papers[paper_id] = paper
                self._generate_embeddings(paper)
                self._add_to_vector_db(paper)
                st.session_state.papers_loaded = True
                return paper

            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                return None

    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        metadata = {"title": "", "authors": [], "publication_date": None, "keywords": []}
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            metadata["title"] = lines[0][:200]

        for pattern in [
            r'Authors?:\s*(.+)',
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+)*)',
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                metadata["authors"] = [a.strip() for a in re.split(r',\s*|\s+and\s+', m.group(1)) if a.strip()]
                break

        for pattern in [r'©\s*(\d{4})', r'published\s+(?:on|in)\s+(\w+\s+\d{4})', r'(\d{4})']:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                metadata["publication_date"] = m.group(1)
                break

        m = re.search(r'Keywords?:\s*(.+)', text, re.IGNORECASE)
        if m:
            metadata["keywords"] = [k.strip() for k in re.split(r'[,;\n]', m.group(1)) if k.strip()]

        return metadata

    def _extract_abstract(self, text: str) -> str:
        for pattern in [
            r'Abstract\s*\n(.+?)(?=\n\s*\n|\nIntroduction|\n\d\.)',
            r'ABSTRACT\s*\n(.+?)(?=\n\s*\n|\nINTRODUCTION|\n1\.)',
            r'Summary\s*\n(.+?)(?=\n\s*\n|\nIntroduction)',
        ]:
            m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if m:
                return re.sub(r'\s+', ' ', m.group(1).strip())[:1000]
        sentences = sent_tokenize(text)
        return " ".join(sentences[:3]) if len(sentences) > 3 else text[:500]

    def _generate_embeddings(self, paper: ResearchPaper):
        text = paper.abstract or (self._chunk_text(paper.content) or [""])[0]
        if text:
            emb = self.embedding_model.encode(text)
            paper.embeddings = emb
            self.embeddings_cache[paper.id] = emb

    def _add_to_vector_db(self, paper: ResearchPaper):
        if paper.embeddings is not None:
            self.collection.add(
                embeddings=[paper.embeddings.tolist()],
                documents=[paper.content[:10000]],
                metadatas=[{
                    "title": paper.title,
                    "authors": ", ".join(paper.authors),
                    "source": paper.source_file,
                    "date": paper.publication_date or "",
                }],
                ids=[paper.id],
            )

    # ==================== TEXT PROCESSING UTILITIES ====================

    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or self.config.CHUNK_OVERLAP
        chunks, start = [], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                for ch in ['.', '!', '?', '\n']:
                    pos = text.rfind(ch, start, end)
                    if pos > start + chunk_size // 2:
                        end = pos + 1
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _extract_key_terms(self, text: str, top_n: int = 20) -> List[str]:
        processed = self._preprocess_text(text)
        vec = TfidfVectorizer(stop_words='english', max_features=top_n * 2, ngram_range=(1, 3))
        try:
            matrix = vec.fit_transform([processed])
            names = vec.get_feature_names_out()
            scores = matrix.toarray()[0]
            return [names[i] for i in scores.argsort()[-top_n:][::-1]]
        except Exception:
            return []

    # ==================== CORE FUNCTIONALITIES ====================

    def summarize_paper(self, paper_id: str, method: str = "abstractive") -> str:
        if paper_id not in self.papers:
            return "Paper not found."
        paper = self.papers[paper_id]
        if method == "abstractive":
            return self.summarize_text(paper.content, max_length=self.config.MAX_SUMMARY_LENGTH)
        # Extractive
        parser = PlaintextParser.from_string(paper.content, Tokenizer("english"))
        try:
            summarizer = LexRankSummarizer()
        except Exception:
            summarizer = LsaSummarizer()
        return " ".join(str(s) for s in summarizer(parser.document, sentences_count=5))

    def semantic_search(self, query: str, top_k: int = None) -> List[SearchResult]:
        top_k = top_k or self.config.TOP_K_RESULTS
        if not self.papers:
            return []
        q_emb = self.embedding_model.encode(query)
        results = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
        out = []
        for i, pid in enumerate(results["ids"][0]):
            if pid in self.papers:
                paper = self.papers[pid]
                passages = self._find_relevant_passages(paper.content, query, top_n=3)
                out.append(SearchResult(
                    paper_id=pid,
                    title=paper.title,
                    similarity_score=results["distances"][0][i],
                    relevant_passages=passages,
                    metadata={"authors": paper.authors, "source": paper.source_file, "date": paper.publication_date},
                ))
        return out

    def _find_relevant_passages(self, text: str, query: str, top_n: int = 3) -> List[str]:
        sentences = sent_tokenize(text)
        if len(sentences) <= top_n:
            return [s[:500] for s in sentences]
        sent_embs = self.embedding_model.encode(sentences)
        q_emb = self.embedding_model.encode([query])
        sims = cosine_similarity(q_emb, sent_embs)[0]
        return [sentences[i][:500] for i in sims.argsort()[-top_n:][::-1]]

    def extract_insights(self, paper_id: str) -> Dict[str, Any]:
        if paper_id not in self.papers:
            return {"error": "Paper not found"}
        paper = self.papers[paper_id]
        with st.spinner("🔍 Extracting insights..."):
            doc = self.nlp(paper.content[:10000])
            entities = defaultdict(list)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART']:
                    entities[ent.label_].append(ent.text)
            for k in entities:
                entities[k] = list(set(entities[k]))

            key_terms = self._extract_key_terms(paper.content)
            word_count = len(paper.content.split())
            sentence_count = len(sent_tokenize(paper.content))
            text_lower = paper.content.lower()
            pos_words = ['good', 'excellent', 'effective', 'efficient', 'improved', 'better']
            neg_words = ['bad', 'poor', 'ineffective', 'inefficient', 'worse', 'limitation']
            pos_score = sum(text_lower.count(w) for w in pos_words)
            neg_score = sum(text_lower.count(w) for w in neg_words)
            overall = "Neutral" if pos_score == neg_score else ("Positive" if pos_score > neg_score else "Negative")

            summary = self.summarize_paper(paper_id, method="abstractive")
            insights = {
                "paper_id": paper_id,
                "title": paper.title,
                "summary": summary,
                "key_terms": key_terms[:15],
                "entities": dict(entities),
                "statistics": {
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "reading_time_minutes": round(word_count / 200),
                },
                "sentiment": {"positive_score": pos_score, "negative_score": neg_score, "overall": overall},
                "recommendations": self._generate_recommendations(paper.content),
            }
            st.session_state.insights[paper_id] = insights
            return insights

    def _generate_recommendations(self, text: str) -> List[str]:
        recs = []
        lower = text.lower()
        if any(w in lower for w in ['machine learning', 'neural network', 'deep learning']):
            recs.append("Consider exploring recent advances in transformer architectures.")
        if any(w in lower for w in ['natural language processing', 'nlp', 'text mining']):
            recs.append("Review state-of-the-art in large language models.")
        if any(w in lower for w in ['limitation', 'future work', 'challenge']):
            recs.append("Focus on addressing mentioned limitations in future research.")
        while len(recs) < 3:
            for r in [
                "Compare findings with similar studies in the field.",
                "Consider practical applications of the research.",
                "Explore interdisciplinary connections.",
            ]:
                if r not in recs:
                    recs.append(r)
                if len(recs) >= 3:
                    break
        return recs[:3]

    def detect_trends(self, time_window_days: int = None) -> Dict[str, Any]:
        time_window_days = time_window_days or self.config.TREND_WINDOW_DAYS
        if len(self.papers) < 2:
            return {"error": "Need at least 2 papers for trend analysis."}
        with st.spinner("📊 Analyzing trends..."):
            all_terms = []
            paper_dates = []
            for paper in self.papers.values():
                all_terms.extend(self._extract_key_terms(paper.content, top_n=20))
                yr = 2023
                if paper.publication_date:
                    m = re.search(r'\d{4}', paper.publication_date)
                    if m:
                        yr = int(m.group())
                paper_dates.append(yr)

            counter = Counter(all_terms)
            top_terms = counter.most_common(self.config.TOP_TREND_TERMS)
            trends = {
                "top_terms": [{"term": t, "frequency": f} for t, f in top_terms],
                "total_papers": len(self.papers),
                "time_range": f"{min(paper_dates)} - {max(paper_dates)}",
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "emerging_topics": self._identify_emerging_topics(top_terms),
            }
            st.session_state.trends = trends
            return trends

    def _identify_emerging_topics(self, top_terms):
        patterns = ['ai', 'llm', 'transformer', 'ethical', 'bias', 'fairness',
                    'sustainable', 'quantum', 'blockchain', 'metaverse', 'artificial intelligence',
                    'large language model', 'attention', 'green']
        emerging = []
        for term, _ in top_terms:
            if any(p in term.lower() for p in patterns) and term not in emerging:
                emerging.append(term)
        return emerging[:5]

    # ==================== VISUALIZATION FUNCTIONS ====================

    def create_wordcloud(self, paper_id: str):
        if paper_id not in self.papers:
            st.error("Paper not found")
            return None
        paper = self.papers[paper_id]
        wc = WordCloud(width=1000, height=500, background_color='white', colormap='viridis',
                       max_words=150, contour_width=2, contour_color='steelblue',
                       prefer_horizontal=0.8, scale=2, random_state=42).generate(paper.content)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud: {paper.title[:60]}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig

    def create_knowledge_graph(self, paper_ids: List[str]):
        if not paper_ids:
            paper_ids = list(self.papers.keys())[:5]
        nodes, edges, node_sizes, node_colors = [], [], [], []
        for pid in paper_ids:
            if pid in self.papers:
                nodes.append(self.papers[pid].title[:30])
                node_sizes.append(30)
                node_colors.append('#667eea')
        for pid in paper_ids:
            if pid in self.papers:
                paper = self.papers[pid]
                doc = self.nlp(paper.content[:3000])
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE'] and len(ent.text) > 2:
                        if ent.text not in nodes:
                            nodes.append(ent.text)
                            node_sizes.append(20)
                            node_colors.append(
                                '#48bb78' if ent.label_ == 'PERSON' else
                                '#ed8936' if ent.label_ == 'ORG' else '#4299e1'
                            )
                        edges.append((paper.title[:30], ent.text))
        if not nodes:
            return None
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, k=2, iterations=50)
        ex, ey = [], []
        for e in G.edges():
            x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
            ex += [x0, x1, None]; ey += [y0, y1, None]
        edge_trace = go.Scatter(x=ex, y=ey, line=dict(width=1, color='#cbd5e0'),
                                hoverinfo='none', mode='lines')
        nx_list = list(nodes)
        nx_x = [pos[n][0] for n in nx_list]
        nx_y = [pos[n][1] for n in nx_list]
        node_trace = go.Scatter(x=nx_x, y=nx_y, mode='markers+text', text=nx_list,
                                textposition="top center", hoverinfo='text',
                                marker=dict(size=node_sizes, color=node_colors,
                                            line_width=2, line_color='white'))
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Knowledge Graph', titlefont_size=16, showlegend=False,
                            hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white'))
        return fig

    def create_trend_chart(self, trends_data: Dict[str, Any]):
        if 'top_terms' not in trends_data:
            return None
        items = trends_data['top_terms'][:10]
        terms = [i['term'] for i in items]
        freqs = [i['frequency'] for i in items]
        fig = go.Figure(data=[go.Bar(
            x=freqs, y=terms, orientation='h',
            marker=dict(color=freqs, colorscale='Viridis',
                        line=dict(color='rgb(8,48,107)', width=1)),
            text=freqs, textposition='auto',
        )])
        fig.update_layout(
            title='Top Trending Terms', xaxis_title='Frequency', yaxis_title='Terms',
            yaxis={'categoryorder': 'total ascending'}, template='plotly_white',
            height=500, margin=dict(l=150, r=50, t=50, b=50))
        return fig

    # ==================== STREAMLIT UI COMPONENTS ====================

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("""
            <div style='text-align:center;padding:1rem 0;'>
                <h1 style='font-size:1.8rem;font-weight:700;
                    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;'>
                    🔬 NLP Research Assistant
                </h1>
                <p style='color:var(--card-subtext);font-size:0.9rem;margin:0.5rem 0;'>
                    AI-Powered Document Analysis
                </p>
            </div>""", unsafe_allow_html=True)
            st.markdown("---")

            st.markdown("### 📱 Navigation")
            page = st.radio("Go to",
                ["🏠 Dashboard", "📄 Paper Analyzer", "🔍 Semantic Search",
                 "📊 Trends Explorer", "🎨 Visualizations", "⚙️ Settings"],
                label_visibility="collapsed")
            st.markdown("---")

            st.markdown("### 📈 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Papers", len(self.papers))
            with col2:
                total = sum(len(p.content) for p in self.papers.values()) if self.papers else 0
                st.metric("Chars", f"{total:,}" if total < 1_000_000 else f"{total//1000}K")
            st.markdown("---")

            st.markdown("### 📤 Quick Upload")
            uploaded = st.file_uploader("Drag & drop PDFs", type="pdf",
                                        accept_multiple_files=True,
                                        label_visibility="collapsed")
            if uploaded:
                success = 0
                with st.spinner(f"Processing {len(uploaded)} file(s)..."):
                    for f in uploaded:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(f.getvalue())
                            tmp_path = tmp.name
                        if self.load_pdf(tmp_path):
                            success += 1
                        os.unlink(tmp_path)
                if success:
                    st.success(f"✅ Added {success} paper(s)!")
                    st.rerun()
            st.markdown("---")

            if self.papers:
                st.markdown("### 📚 Recent Papers")
                for pid, paper in list(self.papers.items())[:3]:
                    st.markdown(f"""
                    <div class='paper-card'>
                        <strong>{paper.title[:40]}...</strong><br>
                        <small style='color:var(--card-subtext);'>{', '.join(paper.authors[:1]) if paper.authors else 'Unknown'}</small>
                    </div>""", unsafe_allow_html=True)
            st.markdown("---")

            st.markdown("### 🟢 System Status")
            st.info(f"**Status:** {'🟢 Active' if self.papers else '🟡 Ready'}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", key="sidebar_refresh", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear", key="sidebar_clear", use_container_width=True):
                    self.papers.clear()
                    st.session_state.clear()
                    st.rerun()
            return page

    def render_home_page(self):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("<h1 class='main-header'>📊 Research Analytics Dashboard</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:1.1rem;color:#4a5568;margin-bottom:2rem;'>Transform your research workflow with AI-powered document analysis.</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='text-align:center;padding:1rem;background:linear-gradient(135deg,#f8fafc,#edf2f7);
                border-radius:12px;border:1px solid #e2e8f0;'>
                <h3 style='color:var(--card-text);margin:0;'>{len(self.papers)}</h3>
                <p style='color:var(--card-subtext);margin:0;font-size:0.9rem;'>Papers Loaded</p>
            </div>""", unsafe_allow_html=True)

        if self.papers:
            st.markdown("<h3 class='sub-header'>📈 Quick Statistics</h3>", unsafe_allow_html=True)
            cols = st.columns(4)
            total_words = sum(len(p.content.split()) for p in self.papers.values())
            for idx, (label, value, color) in enumerate([
                ("Total Papers", len(self.papers), "#4299e1"),
                ("Total Words", f"{total_words:,}", "#48bb78"),
                ("Avg Length", f"{total_words // len(self.papers):,}", "#ed8936"),
                ("Latest", datetime.now().strftime("%b %d"), "#9f7aea"),
            ]):
                with cols[idx]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value' style='color:{color};'>{value}</div>
                        <div class='metric-label'>{label}</div>
                    </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("<h3 class='sub-header'>📚 Featured Papers</h3>", unsafe_allow_html=True)
            if self.papers:
                for pid, paper in list(self.papers.items())[:3]:
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(themed(f"""
                        <div style='padding:1rem;background:var(--card-bg);color:var(--card-text);border-radius:8px;
                            border:1px solid var(--card-border);margin-bottom:1rem;'>
                            <h4 style='margin:0 0 0.5rem 0;color:var(--card-text);'>{paper.title[:70]}</h4>
                            <p style='margin:0 0 0.5rem 0;color:var(--card-subtext);font-size:0.9rem;'>
                                {', '.join(paper.authors[:2]) if paper.authors else 'Unknown authors'}
                            </p>
                            <span class='tag'>📄 PDF</span>
                            <span class='tag'>{len(paper.content.split()):,} words</span>
                        </div>"""), unsafe_allow_html=True)
                    with c2:
                        if st.button("Analyze", key=f"analyze_{pid}", use_container_width=True):
                            st.session_state.current_paper = pid
                            st.session_state['page'] = "📄 Paper Analyzer"
                            st.rerun()
            else:
                st.info("📥 No papers loaded yet. Upload some PDFs to get started!")

            st.markdown("<h3 class='sub-header'>⚡ Quick Actions</h3>", unsafe_allow_html=True)
            ac = st.columns(3)
            with ac[0]:
                if st.button("📊 Generate Trends", key="home_trends", use_container_width=True, disabled=len(self.papers) < 2):
                    st.session_state['page'] = "📊 Trends Explorer"; st.rerun()
            with ac[1]:
                if st.button("🔍 Search Papers", key="home_search", use_container_width=True, disabled=not self.papers):
                    st.session_state['page'] = "🔍 Semantic Search"; st.rerun()
            with ac[2]:
                if st.button("🎨 Create Visuals", key="home_visuals", use_container_width=True, disabled=not self.papers):
                    st.session_state['page'] = "🎨 Visualizations"; st.rerun()

        with col2:
            st.markdown("<h3 class='sub-header'>📤 Upload Panel</h3>", unsafe_allow_html=True)
            uploaded = st.file_uploader("Drop PDFs here", type="pdf",
                                        accept_multiple_files=True, label_visibility="collapsed")
            if uploaded:
                bar = st.progress(0)
                status = st.empty()
                for idx, f in enumerate(uploaded):
                    status.text(f"Processing: {f.name[:30]}...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    self.load_pdf(tmp_path)
                    os.unlink(tmp_path)
                    bar.progress((idx + 1) / len(uploaded))
                bar.empty(); status.empty()
                st.success(f"✅ Processed {len(uploaded)} file(s)!")
                st.rerun()

            st.markdown("<h3 class='sub-header'>🕐 Recent Activity</h3>", unsafe_allow_html=True)
            for activity, t in [("System initialized", "Just now"),
                                  ("Models loaded", "Just now"),
                                  (f"{len(self.papers)} papers available", "Ongoing")]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;
                    padding:0.5rem 0;border-bottom:1px solid var(--activity-border);'>
                    <span>{activity}</span>
                    <span style='color:var(--card-subtext);font-size:0.9rem;'>{t}</span>
                </div>""", unsafe_allow_html=True)

    def render_document_analysis_page(self):
        st.markdown("<h1 class='main-header'>📄 Paper Analyzer</h1>", unsafe_allow_html=True)
        if not self.papers:
            st.warning("⚠️ No papers loaded. Please upload PDFs first."); return

        col1, col2 = st.columns([3, 1])
        with col1:
            options = {f"{p.title[:70]}...": pid for pid, p in self.papers.items()}
            selected_opt = st.selectbox("Select a paper:", list(options.keys()), key="paper_selector")
        with col2:
            if st.button("🔄 Refresh Analysis", key="analyzer_refresh", use_container_width=True):
                pid = options[selected_opt]
                st.session_state.insights.pop(pid, None)
                st.rerun()

        pid = options[selected_opt]
        paper = self.papers[pid]
        st.session_state.current_paper = pid

        st.markdown(themed(f"""
        <div style='background:var(--paper-meta-bg);color:var(--card-text);border-radius:12px;
            padding:1.5rem;margin-bottom:2rem;border:1px solid var(--card-border);'>
            <h3 style='margin-top:0;color:var(--card-text);'>{paper.title}</h3>
            <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;'>
                <div><strong style='color:var(--card-subtext);'>👤 Authors</strong><br>
                    <span style='color:var(--card-text);'>{', '.join(paper.authors[:3]) + ('...' if len(paper.authors)>3 else '') if paper.authors else 'Unknown'}</span></div>
                <div><strong style='color:var(--card-subtext);'>📅 Date</strong><br>
                    <span style='color:var(--card-text);'>{paper.publication_date or 'Unknown'}</span></div>
                <div><strong style='color:var(--card-subtext);'>📏 Length</strong><br>
                    <span style='color:var(--card-text);'>{len(paper.content.split()):,} words</span></div>
                <div><strong style='color:var(--card-subtext);'>🔑 Keywords</strong><br>
                    <span style='color:var(--card-text);'>{', '.join(paper.keywords[:5]) if paper.keywords else 'None'}</span></div>
            </div>
        </div>"""), unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📝 Summary", "🔍 Insights", "🏷️ Entities", "💡 Recommendations", "📊 Overview"])

        with tab1:
            st.markdown("<h3 class='sub-header'>AI-Powered Summary</h3>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                method = st.radio("Method:", ["🤖 Abstractive (AI)", "📋 Extractive (Traditional)"], horizontal=True)
            with c2:
                st.select_slider("Length:", ["Short", "Medium", "Long"], value="Medium")
            with c3:
                gen_btn = st.button("✨ Generate Summary", key="gen_summary", type="primary", use_container_width=True)

            if gen_btn:
                with st.spinner("🤖 Generating summary..."):
                    m = "abstractive" if method.startswith("🤖") else "extractive"
                    summary = self.summarize_paper(pid, m)
                    st.session_state.summary_generated = True
                    st.session_state.current_summary = summary
                    st.markdown(f"<div class='summary-box'><h4 style='color:var(--summary-text);margin-top:0;'>📋 Summary</h4>{summary}</div>",
                                unsafe_allow_html=True)
                    ca, cb, cc = st.columns(3)
                    with ca:
                        if st.button("📋 Copy", key="copy_summary", use_container_width=True):
                            st.code(summary, language="text")
                    with cb:
                        if st.button("💾 Save", key="save_summary", use_container_width=True):
                            fn = f"summary_{pid}.txt"
                            with open(fn, 'w') as f:
                                f.write(f"Title: {paper.title}\n\nSummary:\n{summary}")
                            st.success(f"✅ Saved to {fn}")
                    with cc:
                        if st.button("☁️ Word Cloud", key="summary_wc", use_container_width=True):
                            st.session_state['page'] = "🎨 Visualizations"; st.rerun()
            elif st.session_state.summary_generated:
                st.markdown(f"<div class='summary-box'><h4 style='color:var(--summary-text);margin-top:0;'>📋 Previous Summary</h4>{st.session_state.current_summary}</div>",
                            unsafe_allow_html=True)

        with tab2:
            st.markdown("<h3 class='sub-header'>Deep Insights</h3>", unsafe_allow_html=True)
            if st.button("🔍 Extract Insights", key="extract_insights", type="primary", use_container_width=True):
                ins = self.extract_insights(pid)
                if "error" not in ins:
                    cols = st.columns(4)
                    for i, (lbl, val, clr) in enumerate([
                        ("Word Count", ins['statistics']['word_count'], "#4299e1"),
                        ("Sentences", ins['statistics']['sentence_count'], "#48bb78"),
                        ("Reading Time", f"{ins['statistics']['reading_time_minutes']} min", "#ed8936"),
                        ("Key Terms", len(ins['key_terms']), "#9f7aea"),
                    ]):
                        with cols[i]:
                            st.markdown(themed(f"<div class='metric-card'><div class='metric-value' style='color:{clr};'>{val}</div><div class='metric-label'>{lbl}</div></div>"),
                                        unsafe_allow_html=True)

                    st.markdown("#### 😊 Sentiment Analysis")
                    sc = st.columns(3)
                    with sc[0]:
                        st.markdown(themed("<div style='text-align:center;padding:1rem;background:#c6f6d5;border-radius:8px;border:1px solid #9ae6b4;'>"
                            f"<div style='font-size:2rem;color:#22543d;font-weight:700;'>{ins['sentiment']['positive_score']}</div>"
                            "<div style='color:#22543d;font-weight:600;'>Positive</div></div>"), unsafe_allow_html=True)
                    with sc[1]:
                        st.markdown(themed("<div style='text-align:center;padding:1rem;background:#fed7d7;border-radius:8px;border:1px solid #fc8181;'>"
                            f"<div style='font-size:2rem;color:#9b2c2c;font-weight:700;'>{ins['sentiment']['negative_score']}</div>"
                            "<div style='color:#9b2c2c;font-weight:600;'>Negative</div></div>"), unsafe_allow_html=True)
                    with sc[2]:
                        oc = {"Positive": "#48bb78", "Negative": "#f56565", "Neutral": "#a0aec0"}[ins['sentiment']['overall']]
                        st.markdown(themed(f"<div style='text-align:center;padding:1rem;background:{oc}33;border-radius:8px;border:1px solid {oc};'>"
                            f"<div style='font-size:2rem;color:{oc};font-weight:700;'>{ins['sentiment']['overall'][0]}</div>"
                            f"<div style='color:{oc};font-weight:600;'>{ins['sentiment']['overall']}</div></div>"), unsafe_allow_html=True)

                    st.markdown("#### 🔑 Key Terms")
                    html_tags = "".join(f"<span class='tag tag-primary'>{t}</span>" for t in ins['key_terms'][:15])
                    st.markdown(themed(f"<div style='margin:1rem 0;'>{html_tags}</div>"), unsafe_allow_html=True)

        with tab3:
            st.markdown("<h3 class='sub-header'>Named Entities</h3>", unsafe_allow_html=True)
            if pid in st.session_state.insights:
                entities = st.session_state.insights[pid].get('entities', {})
                if entities:
                    ec = st.columns(3)
                    for idx, (etype, (dname, clr)) in enumerate({
                        'PERSON': ('👤 People', '#4299e1'), 'ORG': ('🏢 Organizations', '#48bb78'),
                        'GPE': ('🌍 Locations', '#ed8936'), 'PRODUCT': ('📦 Products', '#9f7aea'),
                        'WORK_OF_ART': ('🎨 Works', '#ed64a6')
                    }.items()):
                        if etype in entities and entities[etype]:
                            with ec[idx % 3]:
                                # Fixed: explicit color on each item + quotes around padding value
                                items_html = "".join(
                                    f"<div style='padding:0.25rem 0;color:var(--card-text);'>• {e}</div>"
                                    for e in entities[etype][:10]
                                )
                                st.markdown(themed(
                                    f"<div style='background:var(--entity-bg);border-radius:8px;padding:1rem;"
                                    f"border:1px solid var(--card-border);margin-bottom:1rem;'>"
                                    f"<h4 style='color:{clr};margin:0 0 0.5rem 0;'>{dname}</h4>"
                                    f"{items_html}</div>"
                                ), unsafe_allow_html=True)
                else:
                    st.info("No named entities found.")
            else:
                st.info("Click 'Extract Insights' first.")

        with tab4:
            st.markdown("<h3 class='sub-header'>Research Recommendations</h3>", unsafe_allow_html=True)
            if pid in st.session_state.insights:
                for i, rec in enumerate(st.session_state.insights[pid].get('recommendations', []), 1):
                    st.markdown(f"""
                    <div style='background:var(--rec-bg);color:var(--card-text);border-radius:8px;
                        padding:1rem;margin-bottom:1rem;border-left:4px solid #667eea;'>
                        <div style='display:flex;align-items:center;gap:1rem;'>
                            <div style='background:#667eea;color:white;width:30px;height:30px;border-radius:50%;
                                display:flex;align-items:center;justify-content:center;font-weight:bold;'>{i}</div>
                            <div>{rec}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Extract insights first to get recommendations.")

        with tab5:
            st.markdown("<h3 class='sub-header'>Paper Overview</h3>", unsafe_allow_html=True)
            with st.expander("📄 View Paper Content (First 1000 chars)", expanded=False):
                st.text_area("Content", paper.content[:1000], height=200, label_visibility="collapsed")
            ac = st.columns(4)
            with ac[0]:
                if st.button("☁️ Word Cloud", key="overview_wc", use_container_width=True):
                    fig = self.create_wordcloud(pid)
                    if fig: st.pyplot(fig)
            with ac[1]:
                if st.button("📊 Export Report", key="export_report", use_container_width=True):
                    report = self.generate_report(pid)
                    st.download_button("Download", data=report,
                                       file_name=f"report_{pid}.txt", mime="text/plain",
                                       use_container_width=True)
            with ac[2]:
                if st.button("🔗 Similar Papers", key="similar_papers", use_container_width=True):
                    st.session_state['page'] = "🔍 Semantic Search"; st.rerun()
            with ac[3]:
                if st.button("📈 Trends", key="overview_trends", use_container_width=True):
                    st.session_state['page'] = "📊 Trends Explorer"; st.rerun()

    def render_search_page(self):
        st.markdown("<h1 class='main-header'>🔍 Semantic Search</h1>", unsafe_allow_html=True)
        if not self.papers:
            st.warning("⚠️ No papers loaded. Please upload PDFs first."); return

        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Search:", placeholder="Enter topic, question, or keyword...",
                                  key="search_query_main", label_visibility="collapsed")
        with col2:
            top_k = st.number_input("Results", min_value=1, max_value=20, value=5,
                                    step=1, label_visibility="collapsed")

        sc = st.columns([1, 1, 6])
        with sc[0]:
            search_btn = st.button("🔍 Search", key="search_btn", type="primary", use_container_width=True)
        with sc[1]:
            if st.button("🔄 Clear", key="search_clear", use_container_width=True):
                st.session_state.search_results = []; st.rerun()

        if search_btn and query:
            with st.spinner(f"Searching {len(self.papers)} papers..."):
                st.session_state.search_results = self.semantic_search(query, top_k)

        if st.session_state.search_results:
            st.markdown(f"<div style='background:linear-gradient(135deg,#bee3f8,#90cdf4);border-radius:8px;padding:1rem;margin:1rem 0;border:1px solid #4299e1;'><h4 style='margin:0;color:#2c5282;'>📊 {len(st.session_state.search_results)} results for: \"{query}\"</h4></div>",
                        unsafe_allow_html=True)
            for i, result in enumerate(st.session_state.search_results):
                sc_clr = "#48bb78" if result.similarity_score > 0.8 else ("#ed8936" if result.similarity_score > 0.6 else "#f56565")
                sc_lbl = "Excellent" if result.similarity_score > 0.8 else ("Good" if result.similarity_score > 0.6 else "Fair")
                st.markdown(f"""
                <div style='background:var(--search-card-bg);color:var(--card-text);border-radius:10px;padding:1.5rem;margin:1rem 0;
                    border:1px solid var(--card-border);border-left:4px solid {sc_clr};'>
                    <div style='display:flex;justify-content:space-between;align-items:start;'>
                        <div><h3 style='margin:0 0 0.5rem 0;color:var(--card-text);'>#{i+1} {result.title}</h3>
                            <div style='color:var(--card-subtext);font-size:0.9rem;margin-bottom:1rem;'>
                                📅 {result.metadata.get('date','Unknown')}</div></div>
                        <div style='text-align:right;'>
                            <div style='font-size:1.5rem;font-weight:bold;color:{sc_clr};'>{result.similarity_score:.3f}</div>
                            <div style='color:{sc_clr};font-size:0.8rem;'>{sc_lbl}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                with st.expander("📖 Relevant passages"):
                    for j, passage in enumerate(result.relevant_passages):
                        st.markdown(f"<div style='background:var(--passage-bg);color:var(--card-text);border-radius:6px;padding:1rem;margin:0.5rem 0;border:1px solid var(--card-border);'><small style='color:var(--card-subtext);'>📌 Passage {j+1}</small><br>{passage}</div>", unsafe_allow_html=True)
                ac = st.columns(4)
                with ac[0]:
                    if st.button("📄 Analyze", key=f"an_{result.paper_id}", use_container_width=True):
                        st.session_state.current_paper = result.paper_id
                        st.session_state['page'] = "📄 Paper Analyzer"; st.rerun()
                with ac[1]:
                    if st.button("📝 Summary", key=f"sm_{result.paper_id}", use_container_width=True):
                        with st.spinner("Generating..."):
                            st.info(self.summarize_paper(result.paper_id))
                with ac[2]:
                    if st.button("💾 Save", key=f"sv_{result.paper_id}", use_container_width=True):
                        fn = f"result_{result.paper_id}.txt"
                        with open(fn, 'w') as f:
                            f.write(f"Query: {query}\nPaper: {result.title}\nScore: {result.similarity_score}\n\n")
                            f.write("\n".join(result.relevant_passages))
                        st.success(f"✅ Saved to {fn}")
                with ac[3]:
                    if st.button("📊 Compare", key=f"cp_{result.paper_id}", use_container_width=True):
                        st.info("Comparison coming soon!")
                st.markdown("---")

    def render_trends_page(self):
        st.markdown("<h1 class='main-header'>📊 Trends Explorer</h1>", unsafe_allow_html=True)
        if len(self.papers) < 2:
            st.warning("⚠️ Need at least 2 papers for trend analysis."); return

        st.markdown("<h3 class='sub-header'>Analysis Configuration</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            window = st.slider("Time Window (days)", 7, 365, 30, 7)
        with c2:
            top_n = st.slider("Terms to Show", 5, 20, 10, 1)
        with c3:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("📈 Analyze Trends", key="analyze_trends_btn", type="primary", use_container_width=True):
                st.session_state.trends = self.detect_trends(window)

        if st.session_state.trends and 'error' not in st.session_state.trends:
            trends = st.session_state.trends
            cols = st.columns(4)
            for i, (lbl, val, clr) in enumerate([
                ("Papers Analyzed", trends['total_papers'], "#4299e1"),
                ("Time Range", trends['time_range'], "#48bb78"),
                ("Analysis Date", trends['analysis_date'], "#ed8936"),
                ("Top Terms", len(trends['top_terms']), "#9f7aea"),
            ]):
                with cols[i]:
                    st.markdown(themed(f"<div class='metric-card'><div class='metric-value' style='color:{clr};'>{val}</div><div class='metric-label'>{lbl}</div></div>"), unsafe_allow_html=True)

            fig = self.create_trend_chart(trends)
            if fig: st.plotly_chart(fig, use_container_width=True)

            if trends.get('emerging_topics'):
                st.markdown("<h3 class='sub-header'>🚀 Emerging Topics</h3>", unsafe_allow_html=True)
                for topic in trends['emerging_topics']:
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"<div style='background:var(--trend-card-bg);color:var(--card-text);border-radius:8px;padding:1rem;margin-bottom:1rem;border-left:4px solid #667eea;'><h4 style='margin:0;'>{topic}</h4></div>", unsafe_allow_html=True)
                    with c2:
                        if st.button("🔍 Search", key=f"s_{topic}", use_container_width=True):
                            st.session_state['page'] = "🔍 Semantic Search"; st.rerun()

            with st.expander("📋 Detailed Data"):
                df = pd.DataFrame(trends['top_terms'])
                st.dataframe(df, use_container_width=True)
                st.download_button("📥 Download CSV", data=df.to_csv(index=False),
                                   file_name=f"trends_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv")

        elif st.session_state.trends and 'error' in st.session_state.trends:
            st.error(st.session_state.trends['error'])

    def render_visualizations_page(self):
        st.markdown("<h1 class='main-header'>🎨 Visualizations</h1>", unsafe_allow_html=True)
        if not self.papers:
            st.warning("⚠️ No papers loaded."); return

        viz_type = st.selectbox("Choose Visualization:", ["☁️ Word Cloud", "🕸️ Knowledge Graph",
                                                           "📊 Trend Chart", "📈 Comparison View"])

        if viz_type == "☁️ Word Cloud":
            opts = {f"{p.title[:60]}...": pid for pid, p in self.papers.items()}
            c1, c2 = st.columns([3, 1])
            with c1:
                sel = st.selectbox("Select paper:", list(opts.keys()))
            with c2:
                if st.button("✨ Generate", key="gen_wc", type="primary", use_container_width=True):
                    fig = self.create_wordcloud(opts[sel])
                    if fig:
                        st.pyplot(fig)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button("📥 Download PNG", data=buf,
                                           file_name=f"wordcloud_{opts[sel]}.png",
                                           mime="image/png")

        elif viz_type == "🕸️ Knowledge Graph":
            opts = {f"{p.title[:40]}...": pid for pid, p in self.papers.items()}
            sels = st.multiselect("Select papers:", list(opts.keys()),
                                  default=list(opts.keys())[:min(3, len(opts))])
            if sels and st.button("🕸️ Generate Graph", key="gen_graph", type="primary", use_container_width=True):
                fig = self.create_knowledge_graph([opts[s] for s in sels])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "📊 Trend Chart":
            if len(self.papers) >= 2:
                fig = self.create_trend_chart(self.detect_trends())
                if fig: st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 papers.")

        elif viz_type == "📈 Comparison View":
            opts = {f"{p.title[:40]}...": pid for pid, p in self.papers.items()}
            sels = st.multiselect("Select 2-3 papers:", list(opts.keys()), max_selections=3)
            if len(sels) >= 2:
                rows = []
                for name in sels:
                    pid = opts[name]
                    ins = self.extract_insights(pid)
                    rows.append({
                        "Paper": self.papers[pid].title[:40] + "...",
                        "Word Count": ins['statistics']['word_count'],
                        "Key Terms": len(ins['key_terms']),
                        "Entities": sum(len(v) for v in ins['entities'].values()),
                        "Sentiment": ins['sentiment']['overall'],
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

    def render_settings_page(self):
        st.markdown("<h1 class='main-header'>⚙️ Settings & Configuration</h1>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["🔧 General", "🤖 AI Models", "⚡ Performance"])

        with tab1:
            with st.form("gen"):
                c1, c2 = st.columns(2)
                with c1:
                    theme_choice = st.selectbox(
                        "Theme",
                        ["Light", "Dark", "System"],
                        index=["Light", "Dark", "System"].index(st.session_state.get('app_theme', 'Light'))
                    )
                    st.slider("Results per page", 5, 50, 10, 5)
                with c2:
                    st.toggle("Auto-save", value=True)
                    st.toggle("Show tips", value=True)
                save_loc = st.text_input("Save location", value=self.config.RESULTS_DIR)
                if st.form_submit_button("💾 Save", type="primary"):
                    self.config.RESULTS_DIR = save_loc
                    st.session_state['app_theme'] = theme_choice
                    st.success(f"✅ Saved! Theme set to: {theme_choice}")
                    st.rerun()

        with tab2:
            with st.form("models"):
                c1, c2 = st.columns(2)
                with c1:
                    emb = st.selectbox("Embedding Model",
                        ["all-MiniLM-L6-v2 (Fast)", "paraphrase-MiniLM-L3-v2 (Balanced)", "all-mpnet-base-v2 (Accurate)"])
                    summ = st.selectbox("Summarization Model",
                        ["facebook/bart-large-cnn (Default)", "t5-small (Lightweight)", "google/pegasus-xsum (Abstractive)"])
                with c2:
                    ner = st.selectbox("NER Model",
                        ["en_core_web_sm (Small)", "en_core_web_md (Medium)", "en_core_web_lg (Large)"])
                    chunk = st.slider("Chunk size", 500, 2000, 1000, 100)
                if st.form_submit_button("💾 Save", type="primary"):
                    self.config.EMBEDDING_MODEL = emb.split()[0]
                    self.config.SUMMARIZATION_MODEL = summ.split()[0]
                    self.config.NER_MODEL = ner.split()[0]
                    self.config.CHUNK_SIZE = chunk
                    st.success("✅ Saved! Reinitialize to apply model changes.")

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Python", sys.version.split()[0])
                st.metric("Streamlit", st.__version__)
                if PSUTIL_AVAILABLE:
                    st.metric("RAM Available", f"{psutil.virtual_memory().available/1e9:.1f} GB")
            with c2:
                st.metric("Papers", len(self.papers))
                st.metric("Cache", f"{len(self.embeddings_cache)} embeddings")

            with st.form("perf"):
                c1, c2 = st.columns(2)
                with c1:
                    st.slider("Max workers", 1, 8, 4)
                    st.slider("Cache size (MB)", 100, 1000, 500, 50)
                with c2:
                    st.toggle("Enable caching", value=True)
                    st.toggle("Auto-cleanup", value=True)
                if st.form_submit_button("💾 Save", type="primary"):
                    st.success("✅ Saved!")

            with st.expander("⚠️ Danger Zone"):
                st.warning("These actions cannot be undone!")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("🗑️ Clear All Data", key="danger_clear", use_container_width=True):
                        self.papers.clear(); st.session_state.clear(); st.rerun()
                with c2:
                    if st.button("🔄 Reset Settings", key="danger_reset", use_container_width=True):
                        self.config = Config(); st.rerun()
                with c3:
                    if st.button("📤 Export Data", key="danger_export", use_container_width=True):
                        self.save_state(f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    def generate_report(self, paper_id: str) -> str:
        if paper_id not in self.papers:
            return "Paper not found."
        paper = self.papers[paper_id]
        ins = self.extract_insights(paper_id)
        lines = [
            "=" * 60, "  RESEARCH PAPER REPORT", "=" * 60,
            f"Title:   {paper.title}",
            f"Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}",
            f"Source:  {paper.source_file}",
            f"Date:    {paper.publication_date or 'Unknown'}",
            "", "-" * 40, "EXECUTIVE SUMMARY", "-" * 40,
            ins['summary'], "",
            "-" * 40, "KEY INSIGHTS", "-" * 40,
            f"Key Terms:    {', '.join(ins['key_terms'][:10])}",
            f"Word Count:   {ins['statistics']['word_count']}",
            f"Reading Time: {ins['statistics']['reading_time_minutes']} min",
            f"Sentiment:    {ins['sentiment']['overall']}",
            "", "-" * 40, "NAMED ENTITIES", "-" * 40,
        ]
        for etype, ents in ins['entities'].items():
            if ents:
                lines.append(f"{etype}: {', '.join(ents[:5])}")
        lines += ["", "-" * 40, "RECOMMENDATIONS", "-" * 40]
        for i, r in enumerate(ins['recommendations'], 1):
            lines.append(f"{i}. {r}")
        lines += ["", "-" * 40, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 60]
        return "\n".join(lines)

    def save_state(self, file_path: str = "state.json"):
        state = {"config": asdict(self.config),
                 "papers": {pid: p.to_dict() for pid, p in self.papers.items()}}
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        st.success(f"✅ Saved to: {file_path}")

    def load_state(self, file_path: str = "state.json"):
        if not os.path.exists(file_path):
            st.error(f"❌ Not found: {file_path}"); return False
        try:
            with open(file_path) as f:
                state = json.load(f)
            self.config = Config(**state['config'])
            self.papers = {}
            for pid, d in state['papers'].items():
                d.pop('embeddings', None)
                self.papers[pid] = ResearchPaper(**d)
                self._generate_embeddings(self.papers[pid])
            self._initialize_vector_db()
            for p in self.papers.values():
                self._add_to_vector_db(p)
            st.success(f"✅ Loaded {len(self.papers)} papers.")
            return True
        except Exception as e:
            st.error(f"❌ Error: {e}"); return False


# ==================== MAIN STREAMLIT APP ====================
def main():
    st.set_page_config(
        page_title="NLP Research Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': 'https://github.com/your-repo/issues',
            'About': "# NLP Research Assistant\nAI-powered document analysis tool.",
        }
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if 'assistant' not in st.session_state:
        with st.spinner("🚀 Initializing NLP Research Assistant..."):
            st.session_state.assistant = NLPResearchAssistant()
            st.session_state.assistant._initialize_nlp_components()

    assistant = st.session_state.assistant

    if 'page' not in st.session_state:
        st.session_state['page'] = "🏠 Dashboard"

    selected = assistant.render_sidebar()
    if selected != st.session_state['page']:
        st.session_state['page'] = selected

    {
        "🏠 Dashboard": assistant.render_home_page,
        "📄 Paper Analyzer": assistant.render_document_analysis_page,
        "🔍 Semantic Search": assistant.render_search_page,
        "📊 Trends Explorer": assistant.render_trends_page,
        "🎨 Visualizations": assistant.render_visualizations_page,
        "⚙️ Settings": assistant.render_settings_page,
    }.get(st.session_state['page'], assistant.render_home_page)()


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    if STREAMLIT_AVAILABLE:
        main()
    else:
        print("Install streamlit: pip install streamlit")
        print("Then run: streamlit run main.py")
