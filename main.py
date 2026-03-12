"""
NLP Research Assistant - Semester Project with Interactive HTML Frontend
Inspired by LexAna: Advanced NLP tool for research document analysis
Features: PDF extraction, summarization, semantic search, trend detection
Interactive Web Interface using Streamlit
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
import networkx as nx
from wordcloud import WordCloud
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
import chromadb

# Try to import streamlit, install if not available
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

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    """Configuration settings for the NLP Research Assistant"""
    # Model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Lightweight embedding model
    SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"  # Summarization model
    NER_MODEL: str = "en_core_web_sm"  # spaCy NER model
    
    # Processing settings
    CHUNK_SIZE: int = 1000  # Characters per chunk for processing
    CHUNK_OVERLAP: int = 200  # Overlap between chunks
    MAX_SUMMARY_LENGTH: int = 150  # Max words in summary
    MIN_SENTENCE_LENGTH: int = 20  # Min characters in sentence
    
    # Search settings
    TOP_K_RESULTS: int = 5  # Number of search results to return
    SIMILARITY_THRESHOLD: float = 0.7  # Minimum similarity score
    
    # Trend detection
    TREND_WINDOW_DAYS: int = 30  # Days for trend analysis
    TOP_TREND_TERMS: int = 10  # Number of trending terms to show
    
    # UI/Display
    ENABLE_VISUALIZATIONS: bool = True
    SAVE_RESULTS: bool = True
    RESULTS_DIR: str = "research_results"

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
        return asdict(self)

@dataclass
class SearchResult:
    """Represents a search result"""
    paper_id: str
    title: str
    similarity_score: float
    relevant_passages: List[str]
    metadata: Dict[str, Any]

@dataclass
class AnalysisResult:
    """Container for analysis results"""
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
        """Initialize the NLP Research Assistant"""
        self.config = config or Config()
        self.papers: Dict[str, ResearchPaper] = {}
        self.vector_db = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Initialize session state for Streamlit
        self._init_session_state()
        
        # Create results directory
        if self.config.SAVE_RESULTS and not os.path.exists(self.config.RESULTS_DIR):
            os.makedirs(self.config.RESULTS_DIR)
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'papers_loaded' not in st.session_state:
            st.session_state.papers_loaded = False
        if 'current_paper' not in st.session_state:
            st.session_state.current_paper = None
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'insights' not in st.session_state:
            st.session_state.insights = {}
        if 'trends' not in st.session_state:
            st.session_state.trends = {}
    
    def _initialize_nlp_components(self):
        """Initialize all NLP models and components"""
        with st.spinner("Initializing NLP components..."):
            
            # Download NLTK data if needed
            self._download_nltk_data()
            
            # Initialize embedding model
            st.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            
            # Initialize summarization pipeline
            st.info("Loading summarization model...")
            self.summarizer = pipeline(
                "summarization",
                model=self.config.SUMMARIZATION_MODEL,
                tokenizer=self.config.SUMMARIZATION_MODEL
            )
            
            # Initialize spaCy for NER
            st.info("Loading spaCy model...")
            try:
                self.nlp = spacy.load(self.config.NER_MODEL)
            except:
                st.warning(f"Downloading spaCy model: {self.config.NER_MODEL}")
                spacy.cli.download(self.config.NER_MODEL)
                self.nlp = spacy.load(self.config.NER_MODEL)
            
            # Initialize NLTK components
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Initialize ChromaDB for vector storage
            self._initialize_vector_db()
            
            st.success("NLP components initialized successfully!")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        required_data = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'punkt_tab',
            'omw-eng',
            'brown',
        ]
        
        progress_bar = st.progress(0)
        for idx, data in enumerate(required_data):
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                st.info(f"Downloading NLTK data: {data}")
                try:
                    nltk.download(data, quiet=True)
                except Exception as e:
                    st.warning(f"Could not download {data}: {e}")
            progress_bar.progress((idx + 1) / len(required_data))
        progress_bar.empty()
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        self.vector_db = chromadb.EphemeralClient()
        
        # Create or get collection
        self.collection = self.vector_db.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )
    
    # ==================== DOCUMENT PROCESSING ====================
    
    def load_pdf(self, file_path: str) -> Optional[ResearchPaper]:
        """
        Extract text from PDF and create ResearchPaper object
        """
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
        
        with st.spinner(f"Processing PDF: {os.path.basename(file_path)}..."):
            try:
                text = ""
                metadata = {}
                
                # Extract text using pdfplumber (more accurate)
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    
                    # Try to extract metadata from first page
                    if pdf.pages:
                        first_page = pdf.pages[0]
                        first_page_text = first_page.extract_text()
                        metadata = self._extract_metadata(first_page_text)
                
                # Fallback to PyPDF2 if pdfplumber fails
                if not text.strip():
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                
                if not text.strip():
                    st.error("No text could be extracted from PDF")
                    return None
                
                # Generate unique ID
                file_hash = hashlib.md5(text.encode()).hexdigest()[:10]
                paper_id = f"paper_{file_hash}"
                
                # Extract title and abstract
                title = metadata.get('title', os.path.basename(file_path))
                abstract = self._extract_abstract(text)
                
                paper = ResearchPaper(
                    id=paper_id,
                    title=title,
                    authors=metadata.get('authors', []),
                    abstract=abstract,
                    content=text,
                    source_file=file_path,
                    publication_date=metadata.get('publication_date'),
                    keywords=metadata.get('keywords', [])
                )
                
                # Store paper
                self.papers[paper_id] = paper
                
                # Generate and store embeddings
                self._generate_embeddings(paper)
                
                # Add to vector database
                self._add_to_vector_db(paper)
                
                st.session_state.papers_loaded = True
                st.success(f"Successfully loaded paper: {title}")
                return paper
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                return None
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from document text"""
        metadata = {
            'title': '',
            'authors': [],
            'publication_date': None,
            'keywords': []
        }
        
        # Try to find title (usually first line or after abstract marker)
        lines = text.split('\n')
        if lines:
            metadata['title'] = lines[0].strip()[:200]  # Limit title length
        
        # Look for author patterns
        author_patterns = [
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+)*)',
            r'Authors?:\s*(.+)',
            r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                authors_text = match.group(1)
                # Split authors by commas or 'and'
                authors = re.split(r',\s*|\s+and\s+', authors_text)
                metadata['authors'] = [a.strip() for a in authors if a.strip()]
                break
        
        # Look for publication date
        date_patterns = [
            r'(\d{4})',
            r'published\s+(?:on|in)\s+(\w+\s+\d{4})',
            r'©\s*(\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['publication_date'] = match.group(1)
                break
        
        # Extract potential keywords (uppercase words or after 'Keywords:')
        keyword_match = re.search(r'Keywords?:\s*(.+)', text, re.IGNORECASE)
        if keyword_match:
            keywords_text = keyword_match.group(1)
            # Split by commas, semicolons, or newlines
            keywords = re.split(r'[,;\n]', keywords_text)
            metadata['keywords'] = [k.strip() for k in keywords if k.strip()]
        
        return metadata
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from research paper"""
        abstract_patterns = [
            r'Abstract\s*\n(.+?)(?=\n\s*\n|\nIntroduction|\n\d\.)',
            r'ABSTRACT\s*\n(.+?)(?=\n\s*\n|\nINTRODUCTION|\n1\.)',
            r'Summary\s*\n(.+?)(?=\n\s*\n|\nIntroduction)'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                # Clean up abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract[:1000]  # Limit abstract length
        
        # If no abstract found, use first few sentences
        sentences = sent_tokenize(text)
        if len(sentences) > 3:
            return ' '.join(sentences[:3])
        return text[:500]
    
    def _generate_embeddings(self, paper: ResearchPaper):
        """Generate embeddings for paper content"""
        # Use abstract for embedding if available, otherwise use first chunk
        text_to_embed = paper.abstract if paper.abstract else self._chunk_text(paper.content)[0]
        
        if text_to_embed:
            embedding = self.embedding_model.encode(text_to_embed)
            paper.embeddings = embedding
            self.embeddings_cache[paper.id] = embedding
    
    def _add_to_vector_db(self, paper: ResearchPaper):
        """Add paper to vector database"""
        if paper.embeddings is not None:
            self.collection.add(
                embeddings=[paper.embeddings.tolist()],
                documents=[paper.content[:10000]],  # Limit document size
                metadatas=[{
                    "title": paper.title,
                    "authors": ", ".join(paper.authors),
                    "source": paper.source_file,
                    "date": paper.publication_date or ""
                }],
                ids=[paper.id]
            )
    
    # ==================== TEXT PROCESSING UTILITIES ====================
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or self.config.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Try to end at sentence boundary
            if end < text_length:
                # Look for sentence-ending punctuation
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                    text.rfind('\n', start, end)
                )
                if sentence_end > start + chunk_size // 2:  # Ensure reasonable chunk size
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap  # Apply overlap
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers (keep basic punctuation)
        text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_key_terms(self, text: str, top_n: int = 20) -> List[str]:
        """Extract key terms using TF-IDF"""
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=top_n * 2,
            ngram_range=(1, 3)  # Include unigrams, bigrams, and trigrams
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        
        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Get top terms
        top_indices = scores.argsort()[-top_n:][::-1]
        key_terms = [feature_names[i] for i in top_indices]
        
        return key_terms
    
    # ==================== CORE FUNCTIONALITIES ====================
    
    def summarize_paper(self, paper_id: str, method: str = "abstractive") -> str:
        """
        Generate summary of a research paper
        Methods: "abstractive" (neural) or "extractive" (traditional)
        """
        if paper_id not in self.papers:
            return "Paper not found"
        
        paper = self.papers[paper_id]
        
        if method == "abstractive":
            # Use transformer-based summarization
            try:
                with st.spinner("Generating abstractive summary..."):
                    summary = self.summarizer(
                        paper.content[:4000],  # Limit input length
                        max_length=self.config.MAX_SUMMARY_LENGTH,
                        min_length=50,
                        do_sample=False
                    )[0]['summary_text']
                return summary
            except:
                # Fall back to extractive summarization
                method = "extractive"
        
        if method == "extractive":
            # Use traditional extractive summarization
            with st.spinner("Generating extractive summary..."):
                parser = PlaintextParser.from_string(paper.content, Tokenizer("english"))
                
                # Try LexRank first, fall back to LSA
                try:
                    summarizer = LexRankSummarizer()
                except:
                    summarizer = LsaSummarizer()
                
                # Generate summary sentences
                summary_sentences = summarizer(parser.document, sentences_count=5)
                summary = " ".join(str(sentence) for sentence in summary_sentences)
                
                return summary
        
        return "Invalid summarization method"
    
    def semantic_search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """
        Perform semantic search across all papers
        """
        top_k = top_k or self.config.TOP_K_RESULTS
        
        if not self.papers:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        search_results = []
        
        if results['ids'][0]:
            for i, paper_id in enumerate(results['ids'][0]):
                if paper_id in self.papers:
                    paper = self.papers[paper_id]
                    
                    # Find relevant passages
                    relevant_passages = self._find_relevant_passages(
                        paper.content,
                        query,
                        top_n=3
                    )
                    
                    result = SearchResult(
                        paper_id=paper.id,
                        title=paper.title,
                        similarity_score=results['distances'][0][i],
                        relevant_passages=relevant_passages,
                        metadata={
                            'authors': paper.authors,
                            'source': paper.source_file,
                            'date': paper.publication_date
                        }
                    )
                    search_results.append(result)
        
        return search_results
    
    def _find_relevant_passages(self, text: str, query: str, top_n: int = 3) -> List[str]:
        """Find most relevant passages in text for a query"""
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= top_n:
            return [s[:500] for s in sentences]  # Limit passage length
        
        # Encode sentences and query
        sentence_embeddings = self.embedding_model.encode(sentences)
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
        
        # Get top sentences
        top_indices = similarities.argsort()[-top_n:][::-1]
        relevant_passages = [sentences[i][:500] for i in top_indices]
        
        return relevant_passages
    
    def extract_insights(self, paper_id: str) -> Dict[str, Any]:
        """
        Extract comprehensive insights from a paper
        """
        if paper_id not in self.papers:
            return {"error": "Paper not found"}
        
        paper = self.papers[paper_id]
        
        with st.spinner("Extracting insights..."):
            # Process with spaCy
            doc = self.nlp(paper.content[:10000])  # Limit for performance
            
            # Extract named entities
            entities = defaultdict(list)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART']:
                    entities[ent.label_].append(ent.text)
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            # Extract key terms
            key_terms = self._extract_key_terms(paper.content)
            
            # Calculate basic statistics
            word_count = len(paper.content.split())
            sentence_count = len(sent_tokenize(paper.content))
            
            # Analyze sentiment (basic implementation)
            positive_words = ['good', 'excellent', 'effective', 'efficient', 'improved', 'better']
            negative_words = ['bad', 'poor', 'ineffective', 'inefficient', 'worse', 'limitation']
            
            text_lower = paper.content.lower()
            positive_score = sum(text_lower.count(word) for word in positive_words)
            negative_score = sum(text_lower.count(word) for word in negative_words)
            
            # Generate summary
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
                    "reading_time_minutes": round(word_count / 200)  # Average reading speed
                },
                "sentiment": {
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "overall": "Neutral" if positive_score == negative_score else 
                              "Positive" if positive_score > negative_score else "Negative"
                },
                "recommendations": self._generate_recommendations(paper.content)
            }
            
            # Store in session state
            st.session_state.insights[paper_id] = insights
            
            return insights
    
    def _generate_recommendations(self, text: str) -> List[str]:
        """Generate reading recommendations based on content"""
        recommendations = []
        
        # Simple rule-based recommendations
        if any(word in text.lower() for word in ['machine learning', 'neural network', 'deep learning']):
            recommendations.append("Consider exploring recent advances in transformer architectures")
        
        if any(word in text.lower() for word in ['natural language processing', 'nlp', 'text mining']):
            recommendations.append("Review state-of-the-art in large language models")
        
        if any(word in text.lower() for word in ['limitation', 'future work', 'challenge']):
            recommendations.append("Focus on addressing mentioned limitations in future research")
        
        if len(recommendations) < 3:
            recommendations.extend([
                "Compare findings with similar studies in the field",
                "Consider practical applications of the research",
                "Explore interdisciplinary connections"
            ])
        
        return recommendations[:3]
    
    def detect_trends(self, time_window_days: int = None) -> Dict[str, Any]:
        """
        Detect trends across all loaded papers
        """
        time_window_days = time_window_days or self.config.TREND_WINDOW_DAYS
        
        if len(self.papers) < 2:
            return {"error": "Need at least 2 papers for trend analysis"}
        
        with st.spinner("Analyzing trends..."):
            # Collect all terms across papers
            all_terms = []
            paper_dates = []
            
            for paper in self.papers.values():
                # Extract key terms from each paper
                terms = self._extract_key_terms(paper.content, top_n=20)
                all_terms.extend(terms)
                
                # Extract or estimate publication date
                if paper.publication_date:
                    try:
                        # Try to parse date (simplified)
                        year_match = re.search(r'\d{4}', paper.publication_date)
                        if year_match:
                            paper_dates.append(int(year_match.group()))
                    except:
                        paper_dates.append(2023)  # Default year
                else:
                    paper_dates.append(2023)
            
            # Analyze term frequency
            term_counter = Counter(all_terms)
            top_terms = term_counter.most_common(self.config.TOP_TREND_TERMS)
            
            # Calculate trend metrics
            trends = {
                "top_terms": [{"term": term, "frequency": freq} for term, freq in top_terms],
                "total_papers": len(self.papers),
                "time_range": f"{min(paper_dates)} - {max(paper_dates)}",
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "emerging_topics": self._identify_emerging_topics(top_terms)
            }
            
            # Store in session state
            st.session_state.trends = trends
            
            return trends
    
    def _identify_emerging_topics(self, top_terms: List[Tuple[str, int]]) -> List[str]:
        """Identify potentially emerging topics"""
        emerging_patterns = [
            'ai', 'artificial intelligence',
            'llm', 'large language model',
            'transformer', 'attention',
            'ethical', 'bias', 'fairness',
            'sustainable', 'green',
            'quantum', 'blockchain', 'metaverse'
        ]
        
        emerging = []
        for term, _ in top_terms:
            term_lower = term.lower()
            for pattern in emerging_patterns:
                if pattern in term_lower and term not in emerging:
                    emerging.append(term)
        
        return emerging[:5]
    
    # ==================== VISUALIZATION FUNCTIONS ====================
    
    def create_wordcloud(self, paper_id: str):
        """Create word cloud visualization for a paper"""
        if paper_id not in self.papers:
            st.error("Paper not found")
            return None
        
        paper = self.papers[paper_id]
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        ).generate(paper.content)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud: {paper.title[:50]}...', fontsize=14)
        
        return fig
    
    def create_knowledge_graph(self, paper_ids: List[str]):
        """Create knowledge graph visualization"""
        if not paper_ids:
            paper_ids = list(self.papers.keys())[:5]  # Limit to first 5 papers
        
        # Extract entities and relationships
        nodes = set()
        edges = []
        
        for paper_id in paper_ids:
            if paper_id in self.papers:
                paper = self.papers[paper_id]
                doc = self.nlp(paper.content[:5000])  # Limit for performance
                
                # Add paper as node
                nodes.add(paper.title[:30])
                
                # Extract entities and relationships
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                        nodes.add(ent.text)
                        edges.append((paper.title[:30], ent.text, ent.label_))
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(src, dst) for src, dst, _ in edges])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color='gray', ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        ax.set_title("Knowledge Graph of Research Papers")
        ax.axis('off')
        
        return fig
    
    def create_trend_chart(self, trends_data: Dict[str, Any]):
        """Create bar chart for trending terms"""
        if 'top_terms' not in trends_data:
            return None
        
        terms = [item['term'] for item in trends_data['top_terms'][:10]]
        frequencies = [item['frequency'] for item in trends_data['top_terms'][:10]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(terms, frequencies, color='steelblue')
        ax.set_xlabel('Frequency')
        ax.set_title('Top Trending Terms')
        ax.invert_yaxis()  # Highest frequency at top
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {width}', va='center')
        
        return fig
    
    # ==================== STREAMLIT UI COMPONENTS ====================
    
    def render_sidebar(self):
        """Render the sidebar with navigation and info"""
        with st.sidebar:
            st.title("🔬 NLP Research Assistant")
            st.markdown("---")
            
            st.markdown("### 📚 Navigation")
            page = st.radio(
                "Go to",
                ["🏠 Home", "📄 Document Analysis", "🔍 Semantic Search", 
                 "📊 Trend Analysis", "📈 Visualizations", "⚙️ Settings"]
            )
            
            st.markdown("---")
            st.markdown("### 📊 Statistics")
            st.metric("Papers Loaded", len(self.papers))
            
            if self.papers:
                st.metric("Total Content", f"{sum(len(p.content) for p in self.papers.values()):,} chars")
            
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.info("""
            This NLP Research Assistant helps analyze research papers with:
            - PDF text extraction
            - Automatic summarization
            - Semantic search
            - Trend detection
            - Interactive visualizations
            """)
            
            # Quick actions
            st.markdown("---")
            st.markdown("### ⚡ Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Clear All", use_container_width=True):
                    self.papers.clear()
                    st.session_state.clear()
                    st.rerun()
            
            with col2:
                if st.button("💾 Save State", use_container_width=True):
                    self.save_state()
                    st.success("State saved!")
            
            return page
    
    def render_home_page(self):
        """Render the home page"""
        st.title("🏠 Welcome to NLP Research Assistant")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Transform Your Research Workflow
            
            An intelligent assistant for analyzing academic papers using cutting-edge NLP.
            
            **Key Features:**
            - 📥 **Upload & Process PDFs** - Extract text from research papers
            - 📝 **Automatic Summarization** - Get concise summaries of complex papers
            - 🔍 **Semantic Search** - Find relevant content across all papers
            - 📊 **Trend Analysis** - Identify emerging topics and patterns
            - 🎨 **Interactive Visualizations** - Word clouds, knowledge graphs, and more
            - 🤖 **AI-Powered Insights** - Key terms, entities, and recommendations
            
            **Get Started:**
            1. Upload PDFs in the **Document Analysis** section
            2. Explore papers with **Semantic Search**
            3. Generate insights and visualizations
            """)
        
        with col2:
            st.image("https://th.bing.com/th/id/R.26fdcdd198ee8c489f37343769657d5a?rik=4yvj2ukyHDLfGQ&pid=ImgRaw&r=0", 
                    use_column_width=True, caption="AI-Powered Research")
            
            # Quick upload
            st.markdown("### Quick Upload")
            uploaded_files = st.file_uploader(
                "Drag & drop PDFs here",
                type="pdf",
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                    for uploaded_file in uploaded_files:
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Load the PDF
                        paper = self.load_pdf(tmp_path)
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                
                st.success(f"Successfully loaded {len(uploaded_files)} paper(s)!")
                st.rerun()
        
        # Recent activity
        if self.papers:
            st.markdown("---")
            st.markdown("### 📋 Recent Papers")
            
            # Display papers in a grid
            cols = st.columns(3)
            for idx, (paper_id, paper) in enumerate(list(self.papers.items())[:6]):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{paper.title[:50]}...**")
                        st.caption(f"Authors: {', '.join(paper.authors[:2]) if paper.authors else 'Unknown'}")
                        st.caption(f"ID: {paper_id[:8]}...")
                        
                        if st.button("Analyze", key=f"analyze_{paper_id}", use_container_width=True):
                            st.session_state.current_paper = paper_id
                            st.switch_page("Document Analysis")
    
    def render_document_analysis_page(self):
        """Render the document analysis page"""
        st.title("📄 Document Analysis")
        
        if not self.papers:
            st.warning("No papers loaded. Please upload PDFs first.")
            return
        
        # Paper selection
        paper_options = {f"{p.title[:50]}... (ID: {pid})": pid for pid, p in self.papers.items()}
        selected_option = st.selectbox(
            "Select a paper to analyze:",
            options=list(paper_options.keys()),
            index=0
        )
        
        selected_paper_id = paper_options[selected_option]
        paper = self.papers[selected_paper_id]
        
        # Store current paper in session state
        st.session_state.current_paper = selected_paper_id
        
        # Paper metadata
        with st.expander("📋 Paper Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Title", paper.title)
                st.metric("Authors", ", ".join(paper.authors) if paper.authors else "Unknown")
            
            with col2:
                st.metric("Publication Date", paper.publication_date or "Unknown")
                st.metric("Keywords", ", ".join(paper.keywords[:5]) if paper.keywords else "None")
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔑 Key Insights", "🏷️ Named Entities", "💡 Recommendations"])
        
        with tab1:
            st.subheader("Paper Summary")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                summary_method = st.radio(
                    "Summary Method:",
                    ["Abstractive (AI)", "Extractive (Traditional)"],
                    horizontal=True
                )
            
            with col2:
                if st.button("Generate Summary", type="primary"):
                    method = "abstractive" if summary_method == "Abstractive (AI)" else "extractive"
                    with st.spinner("Generating summary..."):
                        summary = self.summarize_paper(selected_paper_id, method)
                        
                        st.markdown("### 📋 Summary")
                        st.write(summary)
                        
                        # Save summary
                        if st.button("💾 Save Summary"):
                            filename = f"summary_{selected_paper_id}.txt"
                            with open(filename, 'w') as f:
                                f.write(f"Title: {paper.title}\n\n")
                                f.write(f"Summary:\n{summary}")
                            st.success(f"Summary saved to {filename}")
        
        with tab2:
            st.subheader("Key Insights")
            
            if st.button("Extract Insights", type="primary"):
                insights = self.extract_insights(selected_paper_id)
                
                if "error" not in insights:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Word Count", insights['statistics']['word_count'])
                    
                    with col2:
                        st.metric("Sentences", insights['statistics']['sentence_count'])
                    
                    with col3:
                        st.metric("Reading Time", f"{insights['statistics']['reading_time_minutes']} min")
                    
                    # Key terms
                    st.markdown("### 🔑 Key Terms")
                    tags = " ".join([f"`{term}`" for term in insights['key_terms'][:10]])
                    st.markdown(tags)
                    
                    # Sentiment
                    st.markdown("### 😊 Sentiment Analysis")
                    sentiment_cols = st.columns(3)
                    with sentiment_cols[0]:
                        st.metric("Positive Score", insights['sentiment']['positive_score'])
                    with sentiment_cols[1]:
                        st.metric("Negative Score", insights['sentiment']['negative_score'])
                    with sentiment_cols[2]:
                        st.metric("Overall", insights['sentiment']['overall'])
        
        with tab3:
            st.subheader("Named Entities")
            
            if selected_paper_id in st.session_state.insights:
                insights = st.session_state.insights[selected_paper_id]
                entities = insights.get('entities', {})
                
                if entities:
                    for entity_type, entity_list in entities.items():
                        with st.expander(f"🏷️ {entity_type} ({len(entity_list)})"):
                            cols = st.columns(3)
                            for i, entity in enumerate(entity_list):
                                with cols[i % 3]:
                                    st.info(entity)
                else:
                    st.info("No named entities found.")
            else:
                st.info("Click 'Extract Insights' to analyze named entities.")
        
        with tab4:
            st.subheader("Recommendations")
            
            if selected_paper_id in st.session_state.insights:
                insights = st.session_state.insights[selected_paper_id]
                recommendations = insights.get('recommendations', [])
                
                for i, rec in enumerate(recommendations, 1):
                    with st.container(border=True):
                        st.markdown(f"**{i}. {rec}**")
                        st.progress(min((i * 20), 100))
            else:
                st.info("Click 'Extract Insights' to get recommendations.")
        
        # Quick actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Word Cloud", use_container_width=True):
                fig = self.create_wordcloud(selected_paper_id)
                if fig:
                    st.pyplot(fig)
        
        with col2:
            if st.button("📄 View Full Text", use_container_width=True):
                with st.expander("Full Paper Content", expanded=True):
                    st.text_area("Content", paper.content[:5000], height=300)
        
        with col3:
            if st.button("💾 Export Report", use_container_width=True):
                report = self.generate_report(selected_paper_id)
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"report_{selected_paper_id}.txt",
                    mime="text/plain"
                )
    
    def render_search_page(self):
        """Render the semantic search page"""
        st.title("🔍 Semantic Search")
        
        if not self.papers:
            st.warning("No papers loaded. Please upload PDFs first.")
            return
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Search across all papers:",
                placeholder="Enter your research query...",
                key="search_query"
            )
        
        with col2:
            top_k = st.number_input(
                "Results",
                min_value=1,
                max_value=20,
                value=self.config.TOP_K_RESULTS,
                step=1
            )
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query:
                with st.spinner(f"Searching across {len(self.papers)} papers..."):
                    results = self.semantic_search(query, top_k)
                    st.session_state.search_results = results
            else:
                st.warning("Please enter a search query.")
        
        # Display results
        if st.session_state.search_results:
            st.markdown(f"### 📊 Found {len(st.session_state.search_results)} results")
            
            for i, result in enumerate(st.session_state.search_results):
                with st.container(border=True):
                    # Header with similarity score
                    similarity_color = "green" if result.similarity_score > 0.8 else "orange" if result.similarity_score > 0.6 else "red"
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{i+1}. {result.title}**")
                        st.caption(f"Authors: {result.metadata.get('authors', 'Unknown')}")
                    
                    with col2:
                        st.metric(
                            "Similarity",
                            f"{result.similarity_score:.3f}",
                            delta_color="off"
                        )
                    
                    # Relevant passages
                    with st.expander("View relevant passages"):
                        for j, passage in enumerate(result.relevant_passages):
                            st.markdown(f"**Passage {j+1}:**")
                            st.info(passage)
                    
                    # Actions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📄 View Paper", key=f"view_{result.paper_id}"):
                            st.session_state.current_paper = result.paper_id
                            st.switch_page("Document Analysis")
                    
                    with col2:
                        if st.button("📝 Summary", key=f"summary_{result.paper_id}"):
                            with st.spinner("Generating summary..."):
                                summary = self.summarize_paper(result.paper_id)
                                st.info(summary)
                    
                    with col3:
                        if st.button("💾 Save Result", key=f"save_{result.paper_id}"):
                            # Save result to file
                            filename = f"search_result_{result.paper_id}.txt"
                            with open(filename, 'w') as f:
                                f.write(f"Query: {query}\n")
                                f.write(f"Paper: {result.title}\n")
                                f.write(f"Similarity: {result.similarity_score}\n\n")
                                f.write("Relevant Passages:\n")
                                for passage in result.relevant_passages:
                                    f.write(f"- {passage}\n")
                            st.success(f"Saved to {filename}")
        elif query and not st.session_state.search_results:
            st.info("No results found. Try a different query.")
    
    def render_trends_page(self):
        """Render the trend analysis page"""
        st.title("📊 Trend Analysis")
        
        if len(self.papers) < 2:
            st.warning("Need at least 2 papers for trend analysis.")
            return
        
        # Analysis controls
        col1, col2 = st.columns(2)
        with col1:
            window_days = st.slider(
                "Analysis Window (days)",
                min_value=7,
                max_value=365,
                value=self.config.TREND_WINDOW_DAYS,
                step=7
            )
        
        with col2:
            top_terms = st.slider(
                "Top Terms to Show",
                min_value=5,
                max_value=20,
                value=self.config.TOP_TREND_TERMS,
                step=1
            )
        
        if st.button("📈 Analyze Trends", type="primary", use_container_width=True):
            with st.spinner("Analyzing trends across all papers..."):
                trends = self.detect_trends(window_days)
                st.session_state.trends = trends
        
        # Display trends
        if st.session_state.trends and 'error' not in st.session_state.trends:
            trends = st.session_state.trends
            
            # Overview metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Papers Analyzed", trends['total_papers'])
            
            with col2:
                st.metric("Time Range", trends['time_range'])
            
            with col3:
                st.metric("Analysis Date", trends['analysis_date'])
            
            # Top terms chart
            st.markdown("### 📈 Trending Terms")
            fig = self.create_trend_chart(trends)
            if fig:
                st.pyplot(fig)
            
            # Emerging topics
            if trends.get('emerging_topics'):
                st.markdown("### 🚀 Emerging Topics")
                for topic in trends['emerging_topics']:
                    with st.container(border=True):
                        st.markdown(f"**{topic}**")
                        st.progress(75)
            
            # Detailed data
            with st.expander("📋 View Detailed Data"):
                df = pd.DataFrame(trends['top_terms'])
                st.dataframe(df, use_container_width=True)
                
                # Export options
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"trends_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        elif st.session_state.trends and 'error' in st.session_state.trends:
            st.error(st.session_state.trends['error'])
    
    def render_visualizations_page(self):
        """Render the visualizations page"""
        st.title("📈 Visualizations")
        
        if not self.papers:
            st.warning("No papers loaded. Please upload PDFs first.")
            return
        
        # Visualization selection
        viz_type = st.selectbox(
            "Choose Visualization Type:",
            ["Word Cloud", "Knowledge Graph", "Comparison Chart"]
        )
        
        if viz_type == "Word Cloud":
            st.markdown("### ☁️ Word Cloud Generator")
            
            # Paper selection for word cloud
            paper_options = {f"{p.title[:50]}...": pid for pid, p in self.papers.items()}
            selected_option = st.selectbox(
                "Select a paper:",
                options=list(paper_options.keys()),
                index=0
            )
            
            selected_paper_id = paper_options[selected_option]
            
            # Generate word cloud
            if st.button("Generate Word Cloud", type="primary"):
                fig = self.create_wordcloud(selected_paper_id)
                if fig:
                    st.pyplot(fig)
                    
                    # Save option
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 Download Word Cloud",
                        data=buf,
                        file_name=f"wordcloud_{selected_paper_id}.png",
                        mime="image/png"
                    )
        
        elif viz_type == "Knowledge Graph":
            st.markdown("### 🕸️ Knowledge Graph")
            
            # Multi-select papers for graph
            paper_options = {f"{p.title[:30]}...": pid for pid, p in self.papers.items()}
            selected_papers = st.multiselect(
                "Select papers to include:",
                options=list(paper_options.keys()),
                default=list(paper_options.keys())[:3]
            )
            
            selected_paper_ids = [paper_options[opt] for opt in selected_papers]
            
            if st.button("Generate Knowledge Graph", type="primary"):
                if selected_paper_ids:
                    fig = self.create_knowledge_graph(selected_paper_ids)
                    if fig:
                        st.pyplot(fig)
                        
                        # Save option
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download Graph",
                            data=buf,
                            file_name=f"knowledge_graph_{datetime.now().strftime('%H%M%S')}.png",
                            mime="image/png"
                        )
                else:
                    st.warning("Please select at least one paper.")
    
    def render_settings_page(self):
        """Render the settings page"""
        st.title("⚙️ Settings")
        
        with st.form("settings_form"):
            st.markdown("### Model Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2", "all-mpnet-base-v2"],
                    index=0
                )
            
            with col2:
                summarization_model = st.selectbox(
                    "Summarization Model",
                    ["facebook/bart-large-cnn", "t5-small", "google/pegasus-xsum"],
                    index=0
                )
            
            st.markdown("### Processing Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider(
                    "Chunk Size (characters)",
                    min_value=500,
                    max_value=2000,
                    value=self.config.CHUNK_SIZE,
                    step=100
                )
                
                max_summary_length = st.slider(
                    "Max Summary Length",
                    min_value=50,
                    max_value=300,
                    value=self.config.MAX_SUMMARY_LENGTH,
                    step=10
                )
            
            with col2:
                chunk_overlap = st.slider(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=500,
                    value=self.config.CHUNK_OVERLAP,
                    step=50
                )
                
                top_k_results = st.slider(
                    "Search Results",
                    min_value=1,
                    max_value=20,
                    value=self.config.TOP_K_RESULTS,
                    step=1
                )
            
            st.markdown("### Display Settings")
            enable_viz = st.toggle("Enable Visualizations", value=self.config.ENABLE_VISUALIZATIONS)
            save_results = st.toggle("Auto-save Results", value=self.config.SAVE_RESULTS)
            
            # Submit button
            if st.form_submit_button("💾 Save Settings", type="primary"):
                # Update config
                self.config.EMBEDDING_MODEL = embedding_model
                self.config.SUMMARIZATION_MODEL = summarization_model
                self.config.CHUNK_SIZE = chunk_size
                self.config.CHUNK_OVERLAP = chunk_overlap
                self.config.MAX_SUMMARY_LENGTH = max_summary_length
                self.config.TOP_K_RESULTS = top_k_results
                self.config.ENABLE_VISUALIZATIONS = enable_viz
                self.config.SAVE_RESULTS = save_results
                
                st.success("Settings saved! Please reinitialize the assistant for changes to take effect.")
        
        # System information
        st.markdown("---")
        st.markdown("### ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Python Version", sys.version.split()[0])
            st.metric("Streamlit Version", st.__version__)
        
        with col2:
            st.metric("Papers Loaded", len(self.papers))
            st.metric("Memory Usage", f"{sys.getsizeof(self.papers) / 1024 / 1024:.2f} MB")
        
        # Danger zone
        with st.expander("⚠️ Danger Zone", expanded=False):
            st.warning("These actions cannot be undone!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All Data", type="secondary"):
                    self.papers.clear()
                    st.session_state.clear()
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset Settings", type="secondary"):
                    self.config = Config()
                    st.rerun()
    
    def generate_report(self, paper_id: str) -> str:
        """Generate comprehensive report for a paper"""
        if paper_id not in self.papers:
            return "Paper not found"
        
        paper = self.papers[paper_id]
        insights = self.extract_insights(paper_id)
        
        report = f"""
        ==================== RESEARCH PAPER REPORT ====================
        
        Title: {paper.title}
        Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}
        Source: {paper.source_file}
        Date: {paper.publication_date or 'Unknown'}
        
        -------------------- EXECUTIVE SUMMARY --------------------
        {insights['summary']}
        
        -------------------- KEY INSIGHTS --------------------
        • Key Terms: {', '.join(insights['key_terms'][:10])}
        • Word Count: {insights['statistics']['word_count']}
        • Estimated Reading Time: {insights['statistics']['reading_time_minutes']} minutes
        • Overall Sentiment: {insights['sentiment']['overall']}
        
        -------------------- NAMED ENTITIES --------------------
        """
        
        for entity_type, entities_list in insights['entities'].items():
            if entities_list:
                report += f"\n{entity_type}: {', '.join(entities_list[:5])}"
        
        report += f"""
        
        -------------------- RECOMMENDATIONS --------------------
        """
        
        for i, rec in enumerate(insights['recommendations'], 1):
            report += f"\n{i}. {rec}"
        
        report += f"""
        
        -------------------- ANALYSIS METADATA --------------------
        Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Paper ID: {paper_id}
        Total papers in database: {len(self.papers)}
        
        ==================== END OF REPORT ====================
        """
        
        return report
    
    def save_state(self, file_path: str = "research_assistant_state.json"):
        """Save current state to file"""
        state = {
            "config": asdict(self.config),
            "papers": {pid: paper.to_dict() for pid, paper in self.papers.items()}
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        st.success(f"State saved to: {file_path}")
    
    def load_state(self, file_path: str = "research_assistant_state.json"):
        """Load state from file"""
        if not os.path.exists(file_path):
            st.error(f"State file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Load config
            self.config = Config(**state['config'])
            
            # Load papers
            self.papers = {}
            for pid, paper_data in state['papers'].items():
                paper = ResearchPaper(**paper_data)
                self.papers[pid] = paper
                # Regenerate embeddings
                self._generate_embeddings(paper)
            
            # Reinitialize vector DB
            self._initialize_vector_db()
            
            # Add papers to vector DB
            for paper in self.papers.values():
                self._add_to_vector_db(paper)
            
            st.success(f"Loaded {len(self.papers)} papers from state file")
            return True
            
        except Exception as e:
            st.error(f"Error loading state: {e}")
            return False

# ==================== STREAMLIT APP ====================
def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="NLP Research Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
    }
    .stDownloadButton > button {
        width: 100%;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        with st.spinner("🚀 Initializing NLP Research Assistant..."):
            st.session_state.assistant = NLPResearchAssistant()
            st.session_state.assistant._initialize_nlp_components()
    
    assistant = st.session_state.assistant
    
    # Render sidebar and get page selection
    page = assistant.render_sidebar()
    
    # Map page names to render functions
    page_map = {
        "🏠 Home": assistant.render_home_page,
        "📄 Document Analysis": assistant.render_document_analysis_page,
        "🔍 Semantic Search": assistant.render_search_page,
        "📊 Trend Analysis": assistant.render_trends_page,
        "📈 Visualizations": assistant.render_visualizations_page,
        "⚙️ Settings": assistant.render_settings_page
    }
    
    # Render selected page
    if page in page_map:
        page_map[page]()
    else:
        assistant.render_home_page()

# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Check if running in Streamlit
    if STREAMLIT_AVAILABLE:
        main()
    else:
        # Fall back to command line interface
        print("\n" + "="*60)
        print("      NLP RESEARCH ASSISTANT")
        print("="*60)
        print("Streamlit not available. Running in command line mode...")
        print("="*60 + "\n")
        
        # You could add the CLI interface here if needed
        print("Please install Streamlit for the web interface:")
        print("pip install streamlit")
        print("\nThen run: streamlit run your_script.py")