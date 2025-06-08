"""
Advanced NLP-based highlight extraction for meeting transcripts
"""

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import yake
from keybert import KeyBERT
from transformers import pipeline
import textstat
from collections import Counter, defaultdict
import re
from typing import List, Dict, Tuple, Optional
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

class IntelligentHighlightExtractor:
    """
    Advanced NLP-based highlight extraction for any type of meeting/discussion
    """
    
    def __init__(self):
        self.logger = logging.getLogger("HighlightExtractor")
        
        # Initialize NLP models
        self._initialize_models()
        
        # Initialize stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Add meeting-specific stop words
        meeting_stopwords = {
            'um', 'uh', 'like', 'you', 'know', 'think', 'going', 'get', 'got',
            'would', 'could', 'should', 'really', 'actually', 'basically',
            'okay', 'right', 'yeah', 'yes', 'no', 'well', 'so', 'and', 'but'
        }
        self.stop_words.update(meeting_stopwords)
    
    def _initialize_models(self):
        """Initialize all NLP models with error handling"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("SpaCy model loaded successfully")
        except OSError:
            self.logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        try:
            # Initialize KeyBERT for semantic keyword extraction
            self.keybert = KeyBERT()
            self.logger.info("KeyBERT model loaded successfully")
        except Exception as e:
            self.logger.warning(f"KeyBERT initialization failed: {e}")
            self.keybert = None
        
        try:
            # Initialize YAKE for unsupervised keyword extraction
            self.yake_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # Extract up to 3-word phrases
                dedupLim=0.7,
                top=20
            )
            self.logger.info("YAKE extractor initialized")
        except Exception as e:
            self.logger.warning(f"YAKE initialization failed: {e}")
            self.yake_extractor = None
        
        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            self.logger.info("Sentiment analyzer loaded")
        except Exception as e:
            self.logger.warning(f"Sentiment analyzer initialization failed: {e}")
            self.sentiment_analyzer = None
    
    def extract_highlights(self, transcription_results: List[Dict]) -> List[str]:
        """
        Extract intelligent highlights from meeting transcription
        
        Args:
            transcription_results: List of transcription segments with speaker info
            
        Returns:
            List of extracted highlights
        """
        if not transcription_results:
            return []
        
        # Combine all text
        all_text = " ".join([result.get('text', '') for result in transcription_results])
        
        if len(all_text.strip()) < 50:  # Too short for meaningful analysis
            return []
        
        highlights = []
        
        try:
            # 1. Extract semantic keywords using multiple methods
            semantic_keywords = self._extract_semantic_keywords(all_text)
            
            # 2. Extract named entities (people, organizations, locations, etc.)
            named_entities = self._extract_named_entities(all_text)
            
            # 3. Extract important phrases and concepts
            important_phrases = self._extract_important_phrases(all_text)
            
            # 4. Extract action items and decisions
            action_items = self._extract_action_items(all_text)
            
            # 5. Extract key topics based on sentence importance
            key_topics = self._extract_key_topics(all_text)
            
            # 6. Extract sentiment-based insights
            sentiment_insights = self._extract_sentiment_insights(transcription_results)
            
            # Combine and rank all highlights
            all_highlights = {
                'Semantic Keywords': semantic_keywords,
                'Named Entities': named_entities,
                'Important Phrases': important_phrases,
                'Action Items': action_items,
                'Key Topics': key_topics,
                'Sentiment Insights': sentiment_insights
            }
            
            # Filter and rank highlights
            final_highlights = self._rank_and_filter_highlights(all_highlights, all_text)
            
            return final_highlights[:8]  # Return top 8 highlights
            
        except Exception as e:
            self.logger.error(f"Error in highlight extraction: {e}")
            # Fallback to simple extraction
            return self._fallback_extraction(all_text)
    
    def _extract_semantic_keywords(self, text: str) -> List[str]:
        """Extract semantically important keywords"""
        keywords = []
        
        # Method 1: KeyBERT (semantic similarity)
        if self.keybert:
            try:
                keybert_keywords = self.keybert.extract_keywords(
                    text, 
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    top_k=10
                )
                keywords.extend([f"{kw[0]} (semantic)" for kw in keybert_keywords[:5]])
            except Exception as e:
                self.logger.debug(f"KeyBERT extraction failed: {e}")
        
        # Method 2: YAKE (unsupervised)
        if self.yake_extractor:
            try:
                yake_keywords = self.yake_extractor.extract_keywords(text)
                # YAKE returns (score, keyword) - lower score is better
                sorted_yake = sorted(yake_keywords, key=lambda x: x[0])[:5]
                keywords.extend([f"{kw[1]} (key concept)" for kw in sorted_yake])
            except Exception as e:
                self.logger.debug(f"YAKE extraction failed: {e}")
        
        return keywords
    
    def _extract_named_entities(self, text: str) -> List[str]:
        """Extract named entities (people, organizations, etc.)"""
        entities = []
        
        # Method 1: spaCy NER
        if self.nlp:
            try:
                doc = self.nlp(text)
                entity_counts = Counter()
                
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                        entity_counts[f"{ent.text} ({ent.label_})"] += 1
                
                # Return entities mentioned multiple times
                for entity, count in entity_counts.most_common(5):
                    if count > 1:
                        entities.append(f"{entity} - mentioned {count} times")
                    else:
                        entities.append(entity)
                        
            except Exception as e:
                self.logger.debug(f"spaCy NER failed: {e}")
        
        # Method 2: NLTK NER (fallback)
        if not entities:
            try:
                sentences = sent_tokenize(text)
                for sentence in sentences[:10]:  # Analyze first 10 sentences
                    tokens = word_tokenize(sentence)
                    pos_tags = pos_tag(tokens)
                    chunks = ne_chunk(pos_tags)
                    
                    for chunk in chunks:
                        if hasattr(chunk, 'label'):
                            entity_name = ' '.join([token for token, pos in chunk.leaves()])
                            entities.append(f"{entity_name} ({chunk.label()})")
                            
                entities = list(set(entities))[:5]  # Remove duplicates
                
            except Exception as e:
                self.logger.debug(f"NLTK NER failed: {e}")
        
        return entities
    
    def _extract_important_phrases(self, text: str) -> List[str]:
        """Extract important phrases using linguistic patterns"""
        phrases = []
        
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                # Extract noun phrases that are likely to be important
                noun_phrases = []
                for chunk in doc.noun_chunks:
                    phrase = chunk.text.strip()
                    # Filter out short or common phrases
                    if (len(phrase.split()) >= 2 and 
                        len(phrase) > 5 and
                        phrase.lower() not in self.stop_words):
                        noun_phrases.append(phrase)
                
                # Count occurrences
                phrase_counts = Counter(noun_phrases)
                
                # Get most common meaningful phrases
                for phrase, count in phrase_counts.most_common(5):
                    if count > 1:
                        phrases.append(f"'{phrase}' (mentioned {count} times)")
                    else:
                        phrases.append(f"'{phrase}'")
                        
            except Exception as e:
                self.logger.debug(f"Phrase extraction failed: {e}")
        
        return phrases
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items and decisions using pattern matching"""
        action_items = []
        
        # Action-oriented patterns
        action_patterns = [
            r"(?i)(need to|should|must|have to|going to|will)\s+([^.!?]*)",
            r"(?i)(let's|let us)\s+([^.!?]*)",
            r"(?i)(action item|action point|next step|follow up|todo|to do)[:]*\s*([^.!?]*)",
            r"(?i)(decided to|agreed to|committed to)\s+([^.!?]*)",
            r"(?i)(responsible for|assigned to|owner)\s+([^.!?]*)"
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # match is a tuple, get the actual action part
                action = match[1] if len(match) > 1 else match[0]
                action = action.strip()
                
                if len(action) > 10 and len(action) < 100:  # Reasonable length
                    action_items.append(f"Action: {action}")
        
        # Decision patterns
        decision_patterns = [
            r"(?i)(decided|concluded|agreed|determined)\s+([^.!?]*)",
            r"(?i)(the decision is|we decided|it was decided)\s+([^.!?]*)"
        ]
        
        for pattern in decision_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                decision = match[1] if len(match) > 1 else match[0]
                decision = decision.strip()
                
                if len(decision) > 10 and len(decision) < 100:
                    action_items.append(f"Decision: {decision}")
        
        return list(set(action_items))[:4]  # Remove duplicates, max 4
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics using sentence importance scoring"""
        topics = []
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 3:
                return topics
            
            # Score sentences based on various criteria
            sentence_scores = []
            
            for sentence in sentences:
                score = 0
                sentence_lower = sentence.lower()
                
                # Length score (medium sentences are often more informative)
                length = len(sentence.split())
                if 10 <= length <= 25:
                    score += 2
                elif 5 <= length <= 35:
                    score += 1
                
                # Question words often indicate important topics
                question_words = ['what', 'why', 'how', 'when', 'where', 'who']
                if any(word in sentence_lower for word in question_words):
                    score += 2
                
                # Topic indicator words
                topic_indicators = [
                    'important', 'key', 'main', 'primary', 'focus', 'issue',
                    'problem', 'solution', 'result', 'conclusion', 'finding',
                    'research', 'study', 'analysis', 'data', 'evidence'
                ]
                score += sum(1 for word in topic_indicators if word in sentence_lower)
                
                # Avoid very short or very long sentences
                if length < 5 or length > 40:
                    score -= 1
                
                sentence_scores.append((sentence, score))
            
            # Sort by score and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            for sentence, score in sentence_scores[:3]:
                if score > 1:  # Only include sentences with decent scores
                    # Clean up the sentence
                    clean_sentence = re.sub(r'\s+', ' ', sentence.strip())
                    if len(clean_sentence) > 20:
                        topics.append(f"Key point: {clean_sentence}")
            
        except Exception as e:
            self.logger.debug(f"Topic extraction failed: {e}")
        
        return topics
    
    def _extract_sentiment_insights(self, transcription_results: List[Dict]) -> List[str]:
        """Extract insights based on sentiment analysis"""
        insights = []
        
        if not self.sentiment_analyzer:
            return insights
        
        try:
            # Analyze sentiment per speaker
            speaker_sentiments = defaultdict(list)
            
            for result in transcription_results:
                text = result.get('text', '').strip()
                speaker = result.get('speaker', 'Unknown')
                
                if len(text) > 20:  # Only analyze substantial text
                    sentiment = self.sentiment_analyzer(text[:512])[0]  # Limit text length
                    speaker_sentiments[speaker].append(sentiment)
            
            # Analyze overall sentiment patterns
            all_sentiments = []
            for sentiments in speaker_sentiments.values():
                all_sentiments.extend(sentiments)
            
            if all_sentiments:
                # Count sentiment types
                sentiment_counts = Counter([s['label'] for s in all_sentiments])
                
                # Generate insights
                if sentiment_counts['NEGATIVE'] > len(all_sentiments) * 0.4:
                    insights.append("Discussion tone: Concerns or challenges raised")
                elif sentiment_counts['POSITIVE'] > len(all_sentiments) * 0.6:
                    insights.append("Discussion tone: Generally positive and constructive")
                
                # Per-speaker insights
                for speaker, sentiments in speaker_sentiments.items():
                    if len(sentiments) >= 2:
                        speaker_sentiment = Counter([s['label'] for s in sentiments])
                        if speaker_sentiment['NEGATIVE'] > len(sentiments) * 0.7:
                            insights.append(f"{speaker}: Expressed concerns or disagreement")
                        elif speaker_sentiment['POSITIVE'] > len(sentiments) * 0.7:
                            insights.append(f"{speaker}: Generally supportive and positive")
            
        except Exception as e:
            self.logger.debug(f"Sentiment analysis failed: {e}")
        
        return insights[:2]  # Max 2 sentiment insights
    
    def _rank_and_filter_highlights(self, all_highlights: Dict, original_text: str) -> List[str]:
        """Rank and filter highlights based on relevance and importance"""
        ranked_highlights = []
        
        for category, highlights in all_highlights.items():
            for highlight in highlights:
                if highlight and len(highlight.strip()) > 5:
                    # Add category prefix for context
                    if not any(prefix in highlight.lower() for prefix in ['action:', 'decision:', 'key point:']):
                        ranked_highlights.append(highlight)
                    else:
                        ranked_highlights.append(highlight)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_highlights = []
        for highlight in ranked_highlights:
            # Simple similarity check
            highlight_clean = re.sub(r'[^\w\s]', '', highlight.lower())
            if highlight_clean not in seen and len(highlight_clean) > 10:
                seen.add(highlight_clean)
                unique_highlights.append(highlight)
        
        return unique_highlights
    
    def _fallback_extraction(self, text: str) -> List[str]:
        """Fallback extraction method using basic NLP"""
        highlights = []
        
        try:
            # Simple frequency-based extraction
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            
            # Get most common meaningful words
            word_counts = Counter(words)
            common_words = word_counts.most_common(10)
            
            # Filter and format
            for word, count in common_words:
                if count > 2 and len(word) > 4:
                    highlights.append(f"{word.title()} (mentioned {count} times)")
            
            # Extract sentences with numbers (often important)
            sentences = sent_tokenize(text)
            for sentence in sentences:
                if re.search(r'\d+', sentence) and len(sentence.split()) < 20:
                    highlights.append(f"Data point: {sentence.strip()}")
                    if len(highlights) >= 5:
                        break
        
        except Exception as e:
            self.logger.debug(f"Fallback extraction failed: {e}")
        
        return highlights[:5]