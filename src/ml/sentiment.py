"""
Financial Sentiment Analysis & NLP Module
Advanced sentiment analysis with financial context and entity recognition
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
from collections import defaultdict, Counter

# NLP Libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from textblob import TextBlob

# Transformers for FinBERT
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    BertTokenizer,
    BertForSequenceClassification
)
import torch

# Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim import corpora, models
from gensim.models import CoherenceModel, Word2Vec

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    text: str
    overall_sentiment: str  # positive, negative, neutral
    sentiment_score: float  # -1 to 1
    confidence: float
    
    # Detailed scores
    positive_score: float
    negative_score: float
    neutral_score: float
    
    # Financial sentiment
    bullish_score: float
    bearish_score: float
    uncertainty_score: float
    
    # Entity sentiment
    entity_sentiments: Dict[str, float]
    
    # Topic sentiment
    topic_sentiments: Dict[str, float]
    
    # Temporal sentiment
    sentence_sentiments: List[float]
    
    # Risk indicators
    risk_mentions: int
    opportunity_mentions: int
    forward_looking_count: int


@dataclass
class DocumentAnalysis:
    """Complete document analysis results"""
    document_id: str
    sections: Dict[str, SentimentResult]
    
    # Aggregate metrics
    overall_sentiment: float
    sentiment_volatility: float
    sentiment_trend: float
    
    # Key insights
    key_topics: List[Tuple[str, float]]
    key_entities: List[Tuple[str, float]]
    key_risks: List[str]
    key_opportunities: List[str]
    
    # Comparative metrics
    yoy_sentiment_change: Optional[float] = None
    peer_sentiment_delta: Optional[float] = None
    
    # Time series
    sentiment_timeline: Optional[pd.DataFrame] = None


class FinancialSentimentAnalyzer:
    """
    Advanced sentiment analysis system for financial documents
    """
    
    # Financial lexicons
    FINANCIAL_POSITIVE = [
        'profit', 'growth', 'increase', 'improve', 'gain', 'positive', 'strong',
        'outperform', 'exceed', 'success', 'opportunity', 'efficient', 'innovative',
        'breakthrough', 'expansion', 'recovery', 'optimistic', 'favorable', 'robust'
    ]
    
    FINANCIAL_NEGATIVE = [
        'loss', 'decline', 'decrease', 'risk', 'concern', 'challenge', 'weak',
        'underperform', 'miss', 'failure', 'threat', 'inefficient', 'obsolete',
        'recession', 'downturn', 'pessimistic', 'unfavorable', 'volatile'
    ]
    
    UNCERTAINTY_WORDS = [
        'may', 'might', 'could', 'possibly', 'uncertain', 'approximately',
        'believe', 'anticipate', 'expect', 'intend', 'seek', 'potential',
        'contingent', 'depend', 'fluctuate', 'variable'
    ]
    
    RISK_INDICATORS = [
        'risk', 'exposure', 'liability', 'contingency', 'vulnerability',
        'threat', 'adverse', 'material weakness', 'going concern', 'impairment'
    ]
    
    def __init__(self,
                 use_finbert: bool = True,
                 use_topic_modeling: bool = True,
                 device: str = 'cpu'):
        """
        Initialize sentiment analyzer
        
        Args:
            use_finbert: Use FinBERT for financial sentiment
            use_topic_modeling: Enable topic modeling
            device: Device for models ('cpu' or 'cuda')
        """
        self.use_finbert = use_finbert
        self.use_topic_modeling = use_topic_modeling
        self.device = device
        
        # Initialize models
        self._initialize_models()
        
        logger.info("FinancialSentimentAnalyzer initialized")
    
    def _initialize_models(self):
        """Initialize NLP models"""
        
        # Download NLTK data
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize FinBERT
        if self.use_finbert:
            try:
                self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
                self.finbert_model.to(self.device)
                self.finbert_model.eval()
                
                # Create pipeline
                self.finbert_pipeline = pipeline(
                    'sentiment-analysis',
                    model=self.finbert_model,
                    tokenizer=self.finbert_tokenizer,
                    device=0 if self.device == 'cuda' else -1
                )
            except Exception as e:
                logger.warning(f"Could not load FinBERT: {e}")
                self.use_finbert = False
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_document(self,
                        text: str,
                        document_id: str = None,
                        sections: Optional[Dict[str, str]] = None) -> DocumentAnalysis:
        """
        Comprehensive document analysis
        
        Args:
            text: Full document text
            document_id: Document identifier
            sections: Optional pre-split sections
            
        Returns:
            DocumentAnalysis object
        """
        # Split into sections if not provided
        if sections is None:
            sections = self._split_sections(text)
        
        # Analyze each section
        section_results = {}
        for section_name, section_text in sections.items():
            if section_text:
                section_results[section_name] = self.analyze_text(section_text)
        
        # Calculate aggregate metrics
        sentiments = [r.sentiment_score for r in section_results.values()]
        overall_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_volatility = np.std(sentiments) if len(sentiments) > 1 else 0
        
        # Calculate trend (slope of sentiment across sections)
        if len(sentiments) > 1:
            x = np.arange(len(sentiments))
            trend = np.polyfit(x, sentiments, 1)[0]
        else:
            trend = 0
        
        # Extract key insights
        key_topics = self._extract_key_topics(text) if self.use_topic_modeling else []
        key_entities = self._extract_key_entities(text) if self.nlp else []
        key_risks = self._extract_risks(text)
        key_opportunities = self._extract_opportunities(text)
        
        # Create timeline if multiple sections
        if len(section_results) > 1:
            timeline = pd.DataFrame({
                'section': list(section_results.keys()),
                'sentiment': [r.sentiment_score for r in section_results.values()],
                'confidence': [r.confidence for r in section_results.values()]
            })
        else:
            timeline = None
        
        return DocumentAnalysis(
            document_id=document_id or 'unknown',
            sections=section_results,
            overall_sentiment=overall_sentiment,
            sentiment_volatility=sentiment_volatility,
            sentiment_trend=trend,
            key_topics=key_topics,
            key_entities=key_entities,
            key_risks=key_risks,
            key_opportunities=key_opportunities,
            sentiment_timeline=timeline
        )
    
    def analyze_text(self, text: str, chunk_size: int = 512) -> SentimentResult:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            chunk_size: Size of chunks for long texts
            
        Returns:
            SentimentResult object
        """
        # Clean text
        text_clean = self._clean_text(text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Get different sentiment scores
        vader_scores = self._get_vader_sentiment(text_clean)
        
        if self.use_finbert:
            finbert_scores = self._get_finbert_sentiment(text_clean, chunk_size)
        else:
            finbert_scores = vader_scores  # Fallback
        
        textblob_scores = self._get_textblob_sentiment(text_clean)
        
        # Combine scores (weighted average)
        weights = {'vader': 0.2, 'finbert': 0.6, 'textblob': 0.2}
        
        if self.use_finbert:
            combined_positive = (
                weights['vader'] * vader_scores['positive'] +
                weights['finbert'] * finbert_scores['positive'] +
                weights['textblob'] * textblob_scores['positive']
            )
            combined_negative = (
                weights['vader'] * vader_scores['negative'] +
                weights['finbert'] * finbert_scores['negative'] +
                weights['textblob'] * textblob_scores['negative']
            )
            combined_neutral = (
                weights['vader'] * vader_scores['neutral'] +
                weights['finbert'] * finbert_scores.get('neutral', 0) +
                weights['textblob'] * textblob_scores.get('neutral', 0)
            )
        else:
            combined_positive = vader_scores['positive']
            combined_negative = vader_scores['negative']
            combined_neutral = vader_scores['neutral']
        
        # Calculate overall sentiment
        sentiment_score = combined_positive - combined_negative
        
        # Determine sentiment label
        if sentiment_score > 0.1:
            overall_sentiment = 'positive'
        elif sentiment_score < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Calculate confidence
        confidence = max(combined_positive, combined_negative, combined_neutral)
        
        # Financial specific sentiment
        bullish_score, bearish_score = self._get_financial_sentiment(text_clean)
        uncertainty_score = self._calculate_uncertainty(text_clean)
        
        # Entity sentiment
        entity_sentiments = self._get_entity_sentiments(text) if self.nlp else {}
        
        # Topic sentiment
        topic_sentiments = self._get_topic_sentiments(text) if self.use_topic_modeling else {}
        
        # Sentence-level sentiment
        sentence_sentiments = []
        for sentence in sentences[:50]:  # Limit to 50 sentences for performance
            sent_score = self._get_sentence_sentiment(sentence)
            sentence_sentiments.append(sent_score)
        
        # Count risk and opportunity mentions
        risk_mentions = sum(1 for word in self.RISK_INDICATORS if word.lower() in text_clean.lower())
        opportunity_mentions = text_clean.lower().count('opportunity')
        
        # Count forward-looking statements
        forward_looking = len([s for s in sentences if self._is_forward_looking(s)])
        
        return SentimentResult(
            text=text[:500],  # Store preview
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            confidence=confidence,
            positive_score=combined_positive,
            negative_score=combined_negative,
            neutral_score=combined_neutral,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            uncertainty_score=uncertainty_score,
            entity_sentiments=entity_sentiments,
            topic_sentiments=topic_sentiments,
            sentence_sentiments=sentence_sentiments,
            risk_mentions=risk_mentions,
            opportunity_mentions=opportunity_mentions,
            forward_looking_count=forward_looking
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis"""
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers (optional)
        # text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def _get_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get VADER sentiment scores"""
        scores = self.vader.polarity_scores(text)
        
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    
    def _get_finbert_sentiment(self, text: str, chunk_size: int = 512) -> Dict[str, float]:
        """Get FinBERT sentiment scores"""
        
        # Split text into chunks if too long
        chunks = self._split_into_chunks(text, chunk_size)
        
        all_scores = {'positive': [], 'negative': [], 'neutral': []}
        
        for chunk in chunks:
            if len(chunk.strip()) < 10:
                continue
            
            try:
                # Get predictions
                results = self.finbert_pipeline(chunk)
                
                # Aggregate scores
                for result in results:
                    label = result['label'].lower()
                    score = result['score']
                    
                    if label in all_scores:
                        all_scores[label].append(score)
            except:
                continue
        
        # Average scores
        final_scores = {}
        for label, scores in all_scores.items():
            final_scores[label] = np.mean(scores) if scores else 0
        
        # Ensure all labels present
        for label in ['positive', 'negative', 'neutral']:
            if label not in final_scores:
                final_scores[label] = 0
        
        return final_scores
    
    def _get_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Get TextBlob sentiment scores"""
        blob = TextBlob(text[:5000])  # Limit text length
        
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Convert to positive/negative scores
        if polarity > 0:
            positive = polarity
            negative = 0
        else:
            positive = 0
            negative = -polarity
        
        neutral = 1 - abs(polarity)
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'subjectivity': subjectivity
        }
    
    def _get_financial_sentiment(self, text: str) -> Tuple[float, float]:
        """Calculate financial-specific sentiment"""
        text_lower = text.lower()
        
        # Count positive and negative financial words
        positive_count = sum(1 for word in self.FINANCIAL_POSITIVE if word in text_lower)
        negative_count = sum(1 for word in self.FINANCIAL_NEGATIVE if word in text_lower)
        
        # Normalize by text length
        text_length = len(text.split())
        
        bullish_score = positive_count / max(text_length, 1) * 100
        bearish_score = negative_count / max(text_length, 1) * 100
        
        return bullish_score, bearish_score
    
    def _calculate_uncertainty(self, text: str) -> float:
        """Calculate uncertainty score"""
        text_lower = text.lower()
        
        uncertainty_count = sum(1 for word in self.UNCERTAINTY_WORDS if word in text_lower)
        text_length = len(text.split())
        
        return uncertainty_count / max(text_length, 1) * 100
    
    def _get_entity_sentiments(self, text: str) -> Dict[str, float]:
        """Get sentiment for named entities"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text[:100000])  # Limit text length
        
        entity_sentiments = {}
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT']:
                # Get context around entity
                start = max(0, ent.start - 10)
                end = min(len(doc), ent.end + 10)
                context = doc[start:end].text
                
                # Get sentiment of context
                sentiment = self._get_sentence_sentiment(context)
                entity_sentiments[ent.text] = sentiment
        
        return entity_sentiments
    
    def _get_topic_sentiments(self, text: str) -> Dict[str, float]:
        """Get sentiment for different topics"""
        # Extract topics
        topics = self._extract_topics(text, n_topics=5)
        
        topic_sentiments = {}
        
        for topic_words, weight in topics:
            # Create topic text from words
            topic_text = ' '.join(topic_words.split()[:5])
            
            # Find sentences containing topic words
            sentences = sent_tokenize(text)
            topic_sentences = [s for s in sentences if any(word in s.lower() for word in topic_words.split())]
            
            if topic_sentences:
                # Get average sentiment of topic sentences
                sentiments = [self._get_sentence_sentiment(s) for s in topic_sentences[:10]]
                topic_sentiments[topic_text] = np.mean(sentiments)
        
        return topic_sentiments
    
    def _get_sentence_sentiment(self, sentence: str) -> float:
        """Get sentiment score for a single sentence"""
        scores = self.vader.polarity_scores(sentence)
        return scores['compound']
    
    def _split_into_chunks(self, text: str, chunk_size: int = 512) -> List[str]:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _split_sections(self, text: str) -> Dict[str, str]:
        """Split document into sections"""
        sections = {}
        
        # Common section patterns
        section_patterns = [
            (r'(?i)business overview', 'business'),
            (r'(?i)risk factors', 'risks'),
            (r'(?i)management.{0,20}discussion', 'mda'),
            (r'(?i)financial statements', 'financials'),
            (r'(?i)executive summary', 'summary')
        ]
        
        for pattern, name in section_patterns:
            match = re.search(pattern, text)
            if match:
                start = match.start()
                # Find next section or end
                end = len(text)
                for other_pattern, _ in section_patterns:
                    if other_pattern != pattern:
                        other_match = re.search(other_pattern, text[start+100:])
                        if other_match:
                            end = min(end, start + 100 + other_match.start())
                
                sections[name] = text[start:end][:50000]  # Limit section size
        
        # If no sections found, treat as single section
        if not sections:
            sections['full_document'] = text[:100000]
        
        return sections
    
    def _extract_topics(self, text: str, n_topics: int = 10) -> List[Tuple[str, float]]:
        """Extract topics using LDA"""
        # Tokenize and vectorize
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2)
        )
        
        try:
            doc_term_matrix = vectorizer.fit_transform([text])
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, doc_term_matrix.shape[1]),
                random_state=42
            )
            lda.fit(doc_term_matrix)
            
            # Get topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = ' '.join([feature_names[i] for i in top_indices[:5]])
                topic_weight = topic[top_indices].mean()
                topics.append((top_words, topic_weight))
            
            return topics
        except:
            return []
    
    def _extract_key_topics(self, text: str) -> List[Tuple[str, float]]:
        """Extract key topics with weights"""
        topics = self._extract_topics(text, n_topics=5)
        return topics[:5]  # Top 5 topics
    
    def _extract_key_entities(self, text: str) -> List[Tuple[str, float]]:
        """Extract key entities with importance scores"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text[:50000])
        
        entity_counts = Counter()
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'PRODUCT', 'MONEY']:
                entity_counts[ent.text] = entity_counts.get(ent.text, 0) + 1
        
        # Normalize counts
        total = sum(entity_counts.values())
        if total == 0:
            return []
        
        entities = [(ent, count/total) for ent, count in entity_counts.most_common(10)]
        
        return entities
    
    def _extract_risks(self, text: str) -> List[str]:
        """Extract risk mentions"""
        sentences = sent_tokenize(text)
        risk_sentences = []
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in self.RISK_INDICATORS):
                # Clean and add
                clean_sent = sentence.strip()
                if 20 < len(clean_sent) < 500:  # Reasonable length
                    risk_sentences.append(clean_sent)
        
        return risk_sentences[:10]  # Top 10 risks
    
    def _extract_opportunities(self, text: str) -> List[str]:
        """Extract opportunity mentions"""
        sentences = sent_tokenize(text)
        opportunity_sentences = []
        
        opportunity_words = ['opportunity', 'potential', 'growth', 'expansion', 'innovation']
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in opportunity_words):
                clean_sent = sentence.strip()
                if 20 < len(clean_sent) < 500:
                    opportunity_sentences.append(clean_sent)
        
        return opportunity_sentences[:10]
    
    def _is_forward_looking(self, sentence: str) -> bool:
        """Check if sentence is forward-looking"""
        forward_words = ['will', 'expect', 'anticipate', 'forecast', 'project', 'intend', 'plan', 'believe']
        return any(word in sentence.lower() for word in forward_words)
    
    def compare_documents(self,
                         current_doc: str,
                         previous_doc: str) -> Dict[str, Any]:
        """
        Compare sentiment between two documents (e.g., YoY)
        
        Args:
            current_doc: Current document text
            previous_doc: Previous document text
            
        Returns:
            Comparison results
        """
        current_analysis = self.analyze_document(current_doc, 'current')
        previous_analysis = self.analyze_document(previous_doc, 'previous')
        
        comparison = {
            'sentiment_change': current_analysis.overall_sentiment - previous_analysis.overall_sentiment,
            'volatility_change': current_analysis.sentiment_volatility - previous_analysis.sentiment_volatility,
            'trend_change': current_analysis.sentiment_trend - previous_analysis.sentiment_trend,
            
            'current_sentiment': current_analysis.overall_sentiment,
            'previous_sentiment': previous_analysis.overall_sentiment,
            
            'section_changes': {}
        }
        
        # Compare sections
        for section in current_analysis.sections:
            if section in previous_analysis.sections:
                current_score = current_analysis.sections[section].sentiment_score
                previous_score = previous_analysis.sections[section].sentiment_score
                comparison['section_changes'][section] = current_score - previous_score
        
        return comparison
    
    def create_sentiment_report(self, analysis: DocumentAnalysis) -> pd.DataFrame:
        """
        Create sentiment analysis report
        
        Args:
            analysis: DocumentAnalysis object
            
        Returns:
            DataFrame with report
        """
        report_data = []
        
        # Overall metrics
        report_data.append({
            'Category': 'Overall',
            'Metric': 'Document Sentiment',
            'Value': f"{analysis.overall_sentiment:.3f}",
            'Interpretation': 'Positive' if analysis.overall_sentiment > 0 else 'Negative'
        })
        
        report_data.append({
            'Category': 'Overall',
            'Metric': 'Sentiment Volatility',
            'Value': f"{analysis.sentiment_volatility:.3f}",
            'Interpretation': 'High' if analysis.sentiment_volatility > 0.2 else 'Low'
        })
        
        report_data.append({
            'Category': 'Overall',
            'Metric': 'Sentiment Trend',
            'Value': f"{analysis.sentiment_trend:.3f}",
            'Interpretation': 'Improving' if analysis.sentiment_trend > 0 else 'Declining'
        })
        
        # Section sentiments
        for section, result in analysis.sections.items():
            report_data.append({
                'Category': 'Section',
                'Metric': f"{section} Sentiment",
                'Value': f"{result.sentiment_score:.3f}",
                'Interpretation': result.overall_sentiment
            })
        
        # Key topics
        for i, (topic, weight) in enumerate(analysis.key_topics[:5], 1):
            report_data.append({
                'Category': 'Topics',
                'Metric': f"Topic {i}",
                'Value': topic,
                'Interpretation': f"Weight: {weight:.3f}"
            })
        
        return pd.DataFrame(report_data)
    
    def visualize_sentiment(self, analysis: DocumentAnalysis) -> go.Figure:
        """
        Create sentiment visualization
        
        Args:
            analysis: DocumentAnalysis object
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Section sentiments
        sections = list(analysis.sections.keys())
        sentiments = [analysis.sections[s].sentiment_score for s in sections]
        colors = ['green' if s > 0 else 'red' for s in sentiments]
        
        fig.add_trace(go.Bar(
            x=sections,
            y=sentiments,
            marker_color=colors,
            name='Section Sentiment'
        ))
        
        # Add overall line
        fig.add_hline(
            y=analysis.overall_sentiment,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Overall: {analysis.overall_sentiment:.3f}"
        )
        
        fig.update_layout(
            title="Document Sentiment Analysis",
            xaxis_title="Section",
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1, 1]),
            template="plotly_white",
            height=500
        )
        
        return fig