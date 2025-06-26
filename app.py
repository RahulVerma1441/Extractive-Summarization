from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re
import nltk
import os
import sys

# Force NLTK to use a specific data path
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)
CORS(app)

def ensure_nltk_data():
    """Ensure NLTK data is downloaded"""
    required_data = ['punkt', 'stopwords']
    
    for data_name in required_data:
        try:
            # Try to use the data
            if data_name == 'punkt':
                sent_tokenize("Test sentence.")
            elif data_name == 'stopwords':
                stopwords.words('english')
        except LookupError:
            print(f"Downloading {data_name}...")
            try:
                nltk.download(data_name, quiet=True)
            except:
                # If download fails, try alternative approach
                print(f"Standard download failed for {data_name}, trying alternative...")
                nltk.download(data_name, quiet=False, force=True)

# Download required NLTK data with error handling
print("Checking NLTK data...")
try:
    ensure_nltk_data()
    print("NLTK data ready!")
except Exception as e:
    print(f"NLTK setup error: {e}")
    print("Please run: python setup_nltk.py first")
    sys.exit(1)

class ExtractiveSummarizer:
    def __init__(self):
        # Initialize the sentence transformer model
        print("Loading SentenceTransformer model...")
        try:
            self.model = SentenceTransformer("all-mpnet-base-v2")
            self.stemmer = PorterStemmer()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e
    
    def preprocess_text(self, paragraph):
        """Preprocess the input text"""
        try:
            # Tokenize into sentences
            document = sent_tokenize(paragraph)
            
            corpus = []
            original_sentences = []
            
            for i in range(len(document)):
                # Store original sentence for final summary
                original_sentences.append(document[i])
                
                # Clean and preprocess for similarity calculation
                new_sent = re.sub(r"[^a-zA-Z]", " ", document[i])
                new_sent = new_sent.lower()
                new_sent = new_sent.split()
                
                # Remove stopwords
                try:
                    stop_words = set(stopwords.words('english'))
                    new_sent = [word for word in new_sent if word not in stop_words]
                except LookupError:
                    # If stopwords not available, continue without removing them
                    print("Warning: Stopwords not available, skipping stopword removal")
                
                new_sent = " ".join(new_sent)
                corpus.append(new_sent)
            
            return corpus, original_sentences
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Fallback: split by periods if sent_tokenize fails
            sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
            return sentences, sentences
    
    def calculate_compression_ratio(self, compression_percentage, total_sentences):
        """Calculate how many sentences to extract based on compression ratio"""
        if compression_percentage >= 90:
            return max(1, total_sentences // 10)  # Very high compression
        elif compression_percentage >= 70:
            return max(2, total_sentences // 5)   # High compression
        elif compression_percentage >= 50:
            return max(3, total_sentences // 3)   # Medium compression
        elif compression_percentage >= 30:
            return max(4, total_sentences // 2)   # Low compression
        else:
            return max(5, int(total_sentences * 0.7))  # Very low compression
    
    def query_focused_summarization(self, corpus, original_sentences, query, top_k):
        """Generate query-focused extractive summary"""
        if not query.strip():
            # If no query, use standard extractive summarization
            return self.standard_extractive_summary(corpus, original_sentences, top_k)
        
        try:
            # Encode the query and all sentences
            query_embedding = self.model.encode([query.lower()])
            sentence_embeddings = self.model.encode(corpus)
            
            # Calculate similarity between query and each sentence
            query_similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]
            
            # Get top sentences most similar to query
            top_results = query_similarities.topk(k=min(top_k, len(corpus)))
            
            # Sort by original order to maintain coherence
            selected_indices = sorted(top_results[1].tolist())
            
            summary = [original_sentences[i] for i in selected_indices]
            return summary
            
        except Exception as e:
            print(f"Error in query-focused summarization: {e}")
            # Fallback to standard summarization
            return self.standard_extractive_summary(corpus, original_sentences, top_k)
    
    def standard_extractive_summary(self, corpus, original_sentences, top_k):
        """Generate standard extractive summary using sentence similarity"""
        if len(corpus) == 0:
            return ["No content to summarize."]
        
        try:
            # Encode sentences
            embeddings = self.model.encode(corpus)
            
            # Calculate cosine similarity matrix
            cos_scores = util.cos_sim(embeddings, embeddings)
            
            # Calculate sentence scores based on similarity to all other sentences
            sentence_scores = cos_scores.sum(dim=1)
            
            # Get top-k sentences
            top_results = sentence_scores.topk(k=min(top_k, len(corpus)))
            
            # Sort by original order to maintain coherence
            selected_indices = sorted(top_results[1].tolist())
            
            summary = [original_sentences[i] for i in selected_indices]
            return summary
            
        except Exception as e:
            print(f"Error in standard summarization: {e}")
            # Fallback: return first few sentences
            return original_sentences[:top_k]
    
    def summarize(self, text, query="", compression_ratio=50, content_type="simple-text"):
        """Main summarization function"""
        try:
            if not text or len(text.strip()) < 50:
                return "Text is too short to summarize effectively."
            
            # Preprocess text
            corpus, original_sentences = self.preprocess_text(text)
            
            if len(original_sentences) <= 2:
                return " ".join(original_sentences)
            
            # Calculate number of sentences for summary
            top_k = self.calculate_compression_ratio(compression_ratio, len(original_sentences))
            
            # Generate summary based on query
            if query and query.strip():
                summary_sentences = self.query_focused_summarization(
                    corpus, original_sentences, query, top_k
                )
            else:
                summary_sentences = self.standard_extractive_summary(
                    corpus, original_sentences, top_k
                )

            # Get indices of selected sentences
            selected_indices = self.get_sentence_indices(original_sentences, summary_sentences)

            return {
                'summary': " ".join(summary_sentences),
                'selected_indices': selected_indices,
                'total_sentences': len(original_sentences)
            }
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
        
    def get_sentence_indices(self, original_sentences, summary_sentences):
        """Get indices of sentences used in the summary"""
        selected_indices = []
        
        for summary_sent in summary_sentences:
            for i, original_sent in enumerate(original_sentences):
                # Clean both sentences for comparison
                clean_summary = summary_sent.strip().lower()
                clean_original = original_sent.strip().lower()
                
                if clean_summary == clean_original:
                    selected_indices.append(i)
                    break
        
        return selected_indices

# Initialize the summarizer
try:
    summarizer = ExtractiveSummarizer()
except Exception as e:
    print(f"Failed to initialize summarizer: {e}")
    print("Please check your internet connection and try again")
    sys.exit(1)

def fetch_web_content(url):
    """Fetch content from web URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text content from main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup
        text = main_content.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        raise Exception(f"Error fetching web content: {str(e)}")

def process_content_by_type(content_type, text_input):
    """Process different content types"""
    if content_type == "web-pages":
        if text_input.startswith(('http://', 'https://')):
            return fetch_web_content(text_input)
        else:
            # Treat as text content if not a URL
            return text_input
    
    elif content_type in ["comments", "social-media", "documents"]:
        # For now, treat these as text input
        # You can extend this to handle specific APIs later
        return text_input
    
    else:  # simple-text
        return text_input

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.json
        
        # Extract parameters
        content_type = data.get('contentType', 'simple-text')
        compression_ratio = int(data.get('compressionRatio', 50))
        user_query = data.get('userQuery', '')
        text_content = data.get('textContent', '')
        
        # Validate required fields
        if not text_content.strip():
            return jsonify({'error': 'Text content is required'}), 400
        
        # Process content based on type
        processed_text = process_content_by_type(content_type, text_content)
        
        if len(processed_text.strip()) < 50:
            return jsonify({'error': 'Text is too short to summarize (minimum 50 characters)'}), 400
        
        # Generate summary
        summary_result = summarizer.summarize(
            text=processed_text,
            query=user_query,
            compression_ratio=compression_ratio,
            content_type=content_type
        )

        # Handle both old string format and new dict format
        if isinstance(summary_result, dict):
            summary = summary_result['summary']
            selected_indices = summary_result['selected_indices']
            total_sentences = summary_result['total_sentences']
        else:
            # Fallback for error cases
            summary = summary_result
            selected_indices = []
            total_sentences = 0

        # Calculate statistics
        original_length = len(processed_text)
        summary_length = len(summary)
        actual_compression = round((1 - summary_length/original_length) * 100, 2) if original_length > 0 else 0

        # Count sentences
        original_sentences = len(sent_tokenize(processed_text))
        summary_sentences = len(sent_tokenize(summary))

        # Add selected_indices to the response
        return jsonify({
            'success': True,
            'summary': summary,
            'selected_indices': selected_indices,
            'statistics': {
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_achieved': actual_compression,
                'original_sentences': original_sentences,
                'summary_sentences': summary_sentences,
                'content_type': content_type,
                'query_used': bool(user_query.strip())
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'message': 'Extractive Summarization API is running',
        'model': 'all-mpnet-base-v2'
    })

if __name__ == '__main__':
    print("Starting Extractive Summarization API...")
    print("Make sure to install dependencies: pip install flask flask-cors nltk sentence-transformers beautifulsoup4 requests")
    app.run(debug=True, host='0.0.0.0', port=5000)