import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# BERT imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Marathi Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class MarathiTextProcessor:
    """Class to handle Marathi text preprocessing and dataset management"""
    
    def __init__(self, dataset_path="Dataset"):
        self.dataset_path = dataset_path
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        
    def load_combined_data(self):
        """Load and combine emotion and hate speech datasets"""
        try:
            # Load emotion datasets
            logger.info("Loading emotion datasets...")
            emotion_files = {
                'train': f"{self.dataset_path}/emotion/train.csv",
                'validation': f"{self.dataset_path}/emotion/validation.csv", 
                'test': f"{self.dataset_path}/emotion/test.csv"
            }
            
            emotion_dfs = {}
            for split, filepath in emotion_files.items():
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    logger.info(f"Raw emotion {split} columns: {list(df.columns)}")
                    
                    # Standardize column names
                    if 'text' not in df.columns:
                        text_cols = [col for col in df.columns if 'text' in col.lower() or 'sentence' in col.lower() or 'comment' in col.lower()]
                        if text_cols:
                            df = df.rename(columns={text_cols[0]: 'text'})
                    
                    if 'label' not in df.columns:
                        label_cols = [col for col in df.columns if 'label' in col.lower() or 'category' in col.lower() or 'sentiment' in col.lower()]
                        if label_cols:
                            df = df.rename(columns={label_cols[0]: 'label'})
                    
                    if 'text' not in df.columns or 'label' not in df.columns:
                        emotion_dfs[split] = pd.DataFrame(columns=['text', 'label'])
                        continue
                    
                    df = df[['text', 'label']].dropna().drop_duplicates().reset_index(drop=True)
                    emotion_dfs[split] = df
                else:
                    emotion_dfs[split] = pd.DataFrame(columns=['text', 'label'])
            
            # Load hate speech datasets
            logger.info("Loading hate speech datasets...")
            hate_files = {
                'train': f"{self.dataset_path}/hate/hate_bin_train.csv",
                'civil': f"{self.dataset_path}/hate/civil_hate_augmented.csv",
                'validation': f"{self.dataset_path}/hate/hate_bin_valid.csv",
                'test': f"{self.dataset_path}/hate/hate_bin_test.csv"
            }
            
            hate_dfs = {}
            for split, filepath in hate_files.items():
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    
                    if 'sentence' in df.columns:
                        df = df.rename(columns={'sentence': 'text'})
                    if 'category' in df.columns:
                        df = df.rename(columns={'category': 'label'})
                    
                    if 'text' not in df.columns:
                        text_cols = [col for col in df.columns if 'text' in col.lower() or 'sentence' in col.lower() or 'comment' in col.lower()]
                        if text_cols:
                            df = df.rename(columns={text_cols[0]: 'text'})
                    
                    if 'label' not in df.columns:
                        label_cols = [col for col in df.columns if 'label' in col.lower() or 'category' in col.lower() or 'sentiment' in col.lower()]
                        if label_cols:
                            df = df.rename(columns={label_cols[0]: 'label'})
                    
                    if 'text' not in df.columns or 'label' not in df.columns:
                        hate_dfs[split] = pd.DataFrame(columns=['text', 'label'])
                        continue
                    
                    df = df[['text', 'label']].dropna().drop_duplicates().reset_index(drop=True)
                    hate_dfs[split] = df
                else:
                    hate_dfs[split] = pd.DataFrame(columns=['text', 'label'])
            
            # Combine datasets
            if 'train' in hate_dfs and 'civil' in hate_dfs and len(hate_dfs['civil']) > 0:
                hate_train_combined = pd.concat([hate_dfs['train'], hate_dfs['civil']], ignore_index=True)
                hate_train_combined = hate_train_combined.dropna().drop_duplicates().reset_index(drop=True)
            else:
                hate_train_combined = hate_dfs.get('train', pd.DataFrame(columns=['text', 'label']))
            
            # Combine emotion and hate datasets
            train_parts = []
            if len(emotion_dfs['train']) > 0:
                train_parts.append(emotion_dfs['train'])
            if len(hate_train_combined) > 0:
                train_parts.append(hate_train_combined)
            
            if train_parts:
                self.train_df = pd.concat(train_parts, ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
            else:
                self.train_df = pd.DataFrame(columns=['text', 'label'])
            
            # Validation dataset
            valid_parts = []
            if len(emotion_dfs['validation']) > 0:
                valid_parts.append(emotion_dfs['validation'])
            if len(hate_dfs['validation']) > 0:
                valid_parts.append(hate_dfs['validation'])
            
            if valid_parts:
                self.valid_df = pd.concat(valid_parts, ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
            else:
                self.valid_df = pd.DataFrame(columns=['text', 'label'])
            
            # Test dataset
            test_parts = []
            if len(emotion_dfs['test']) > 0:
                test_parts.append(emotion_dfs['test'])
            if len(hate_dfs['test']) > 0:
                test_parts.append(hate_dfs['test'])
            
            if test_parts:
                self.test_df = pd.concat(test_parts, ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
            else:
                self.test_df = pd.DataFrame(columns=['text', 'label'])
            
            return self.train_df, self.valid_df, self.test_df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame(columns=['text', 'label']), pd.DataFrame(columns=['text', 'label']), pd.DataFrame(columns=['text', 'label'])
    
    @staticmethod
    def preprocess_marathi_text(text):
        """Enhanced preprocessing for Marathi text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags but keep the text
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove numbers (often not useful for emotion)
        text = re.sub(r'\d+', '', text)
        
        # Keep Devanagari script, basic punctuation, and common symbols
        text = re.sub(r'[^\u0900-\u097F\s\.,!?।\-\'\"]', ' ', text)
        
        # Remove repeated characters (like हाहाहा -> हाहा)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def prepare_combined_data(self):
        """Load and preprocess combined data"""
        if self.train_df is None:
            self.load_combined_data()
        
        logger.info("Preprocessing text data...")
        
        # Apply preprocessing
        self.train_df['processed_text'] = self.train_df['text'].apply(self.preprocess_marathi_text)
        self.valid_df['processed_text'] = self.valid_df['text'].apply(self.preprocess_marathi_text)
        self.test_df['processed_text'] = self.test_df['text'].apply(self.preprocess_marathi_text)
        
        # Remove empty processed texts
        self.train_df = self.train_df[self.train_df['processed_text'].str.len() > 0].reset_index(drop=True)
        self.valid_df = self.valid_df[self.valid_df['processed_text'].str.len() > 0].reset_index(drop=True)
        self.test_df = self.test_df[self.test_df['processed_text'].str.len() > 0].reset_index(drop=True)
        
        return self.train_df, self.valid_df, self.test_df

class MarathiTextClassifier:
    """Class to handle model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.model_scores = {}
        self.model_f1_scores = {}
        
    def train_models(self, train_df, valid_df=None, test_df=None):
        """Train multiple models on the dataset"""
        
        if len(train_df) == 0:
            raise ValueError("Training dataset is empty!")
        
        X_train = train_df['processed_text']
        y_train = train_df['label']
        
        if X_train.str.len().sum() == 0:
            raise ValueError("All processed text is empty!")
        
        if valid_df is not None and test_df is not None and not valid_df.empty and not test_df.empty:
            X_valid = valid_df['processed_text']
            y_valid = valid_df['label']
            X_test = test_df['processed_text']
            y_test = test_df['label']
        else:
            if len(y_train.unique()) < 2:
                raise ValueError("Need at least 2 different labels for training!")
                
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
            )
        
        # Vectorize text with optimized parameters for multilingual text
        self.vectorizer = TfidfVectorizer(
            max_features=15000,      # Increased for better feature coverage
            ngram_range=(1, 2),      # Reduced to (1,2) - trigrams might add noise
            stop_words=None,         # Keep all words (important for Marathi)
            min_df=3,               # Slightly higher to remove very rare words
            max_df=0.9,             # Lower threshold for common words
            sublinear_tf=True,      # Use log scaling
            use_idf=True,           # Use inverse document frequency
            smooth_idf=True,        # Smooth IDF weights
            norm='l2',              # L2 normalization
            analyzer='word',        # Word-level analysis
            lowercase=True,         # Already done in preprocessing but ensure consistency
            token_pattern=r'[\u0900-\u097F]+|[a-zA-Z]+',  # Marathi + English tokens
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_valid_tfidf = self.vectorizer.transform(X_valid)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Define models with improved hyperparameters for multilingual text
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=3000,        # Increased iterations
                C=10.0,              # Higher C for complex text patterns
                solver='liblinear',
                class_weight='balanced'  # Handle class imbalance
            ),
            'Naive Bayes': MultinomialNB(
                alpha=0.5,           # Slightly higher smoothing for multilingual
                fit_prior=True       # Use class priors
            ),
            'Linear SVM': LinearSVC(
                random_state=42,
                C=10.0,              # Higher C for better performance
                max_iter=5000,       # More iterations
                dual=False,
                class_weight='balanced',
                loss='squared_hinge'  # Better loss function
            ),
            'SGD Classifier': SGDClassifier(
                loss='log_loss',
                random_state=42,
                max_iter=3000,
                alpha=0.00001,       # Lower regularization
                class_weight='balanced',
                learning_rate='adaptive',  # Adaptive learning rate
                eta0=0.01
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42, 
                n_estimators=300,     # More trees
                max_depth=15,        # Control overfitting
                min_samples_split=8,  # Higher split threshold
                min_samples_leaf=3,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt',  # Feature sampling
                n_jobs=-1
            )
        }
        
        # Train individual models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_tfidf, y_train)
            
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            self.models[name] = model
            self.model_scores[name] = accuracy
            self.model_f1_scores[name] = f1
        
        # Train ensemble model
        ensemble = VotingClassifier([
            ('lr', models['Logistic Regression']),
            ('svm', models['Linear SVM']),
            ('sgd', models['SGD Classifier']),
            ('rf', models['Random Forest'])
        ], voting='hard')
        
        ensemble.fit(X_train_tfidf, y_train)
        ensemble_pred = ensemble.predict(X_test_tfidf)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
        
        self.models['Ensemble'] = ensemble
        self.model_scores['Ensemble'] = ensemble_accuracy
        self.model_f1_scores['Ensemble'] = ensemble_f1
        
        return self.models, self.vectorizer, self.model_scores, self.model_f1_scores, X_test, y_test
    
    def predict(self, text, model_name='Ensemble'):
        """Predict sentiment for given text"""
        if model_name not in self.models or self.vectorizer is None:
            raise ValueError("Model not trained yet!")
        
        processed_text = MarathiTextProcessor.preprocess_marathi_text(text)
        text_tfidf = self.vectorizer.transform([processed_text])
        
        prediction = self.models[model_name].predict(text_tfidf)[0]
        
        # Get probabilities (handle models that don't support predict_proba)
        try:
            probabilities = self.models[model_name].predict_proba(text_tfidf)[0]
            prob_dict = dict(zip(self.models[model_name].classes_, probabilities))
        except:
            prob_dict = {prediction: 1.0}
        
        return prediction, prob_dict

# Cached BERT model loading function (similar to app2.py)
@st.cache_resource
def load_bert_model_cached(model_name="l3cube-pune/marathi-sentiment-tweets"):
    """Load pre-trained Marathi BERT model for sentiment analysis with caching"""
    if not BERT_AVAILABLE:
        return None
    
    try:
        # Try to load L3Cube Marathi BERT model (same as app2.py)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", 
                                    model=model, 
                                    tokenizer=tokenizer)
        return sentiment_pipeline
    except Exception as e:
        logger.warning(f"Could not load BERT model {model_name}: {str(e)}")
        return None

def apply_bert_mapping(prediction, prob_dict):
    """Apply custom mapping for BERT predictions: Positive -> NOT, Negative -> HOF"""
    if prediction == "Positive":
        prediction = "NOT"
        prob_dict = {"NOT": prob_dict.get("Positive", 0)}
    elif prediction == "Negative":
        prediction = "HOF"
        prob_dict = {"HOF": prob_dict.get("Negative", 0)}
    
    return prediction, prob_dict

#adding l3cude 
class BERT:
    pass
class BERTMarathiClassifier:
    """BERT-based classifier for Marathi sentiment analysis"""
    
    def __init__(self):
        self.pipeline = None
        self.is_trained = False
        
    def load_pretrained_model(self, model_name="l3cube-pune/marathi-sentiment-tweets"):
        """Load a pre-trained Marathi BERT model using the same approach as app2.py"""
        try:
            if not BERT_AVAILABLE:
                raise ImportError("Transformers library not available")
            
            logger.info(f"Loading Marathi BERT model: {model_name}")
            
            # Use the same approach as app2.py - load L3Cube Marathi sentiment model
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.pipeline = pipeline("sentiment-analysis", 
                                       model=model, 
                                       tokenizer=tokenizer)
                self.is_trained = True
                logger.info("Successfully loaded BERT model")
                return True
                
            except Exception as e:
                logger.error(f"Error loading specific model {model_name}: {str(e)}")
                
                # Fallback: Try alternative models
                fallback_models = [
                    "l3cube-pune/marathi-bert",
                    "ai4bharat/indic-bert",
                    "bert-base-multilingual-cased"
                ]
                
                for fallback_model in fallback_models:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                        model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
                        self.pipeline = pipeline("sentiment-analysis", 
                                               model=model, 
                                               tokenizer=tokenizer)
                        self.is_trained = True
                        logger.info(f"Successfully loaded fallback model: {fallback_model}")
                        return True
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} failed: {str(fallback_error)}")
                        continue
                
                # If all models fail, use zero-shot classification as last resort
                try:
                    logger.info("Trying zero-shot classification as last resort")
                    self.pipeline = pipeline(
                        "zero-shot-classification",
                        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                        tokenizer="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
                    )
                    self.is_trained = True
                    logger.info("Successfully loaded zero-shot classification model")
                    return True
                except Exception as zero_shot_error:
                    logger.error(f"Zero-shot classification also failed: {str(zero_shot_error)}")
                    raise Exception("All BERT model loading attempts failed")
            
        except Exception as e:
            logger.error(f"Error loading Marathi BERT model: {str(e)}")
            st.error(f"BERT Loading Error: {str(e)}")
            return False
    
    def predict(self, text):
        """Predict sentiment using BERT model - same approach as app2.py with custom mapping"""
        if not self.is_trained or not self.pipeline:
            return "Model not loaded", {}
        
        try:
            # Preprocess text
            processed_text = MarathiTextProcessor.preprocess_marathi_text(text)
            
            if not processed_text.strip():
                return "Neutral", {"Neutral": 1.0}
            
            # Use the pipeline for prediction (same as app2.py)
            if hasattr(self.pipeline, 'task') and self.pipeline.task == "zero-shot-classification":
                # Zero-shot classification
                candidate_labels = [
                    "Anger", "Disgust", "Surprise", "Sadness", 
                    "Neutral", "Sarcasm", "Fear", "Pride", "HOF", "Joy"
                ]
                result = self.pipeline(processed_text, candidate_labels)
                prediction = result['labels'][0]
                prob_dict = dict(zip(result['labels'], result['scores']))
            else:
                # Regular sentiment analysis pipeline
                result = self.pipeline(processed_text)[0]
                prediction = result['label']
                prob_dict = {prediction: result['score']}
            
            # Apply custom mapping: Positive -> NOT, Negative -> HOF
            prediction, prob_dict = apply_bert_mapping(prediction, prob_dict)
            
            return prediction, prob_dict
            
        except Exception as e:
            logger.error(f"BERT prediction error: {str(e)}")
            return "Error", {"Error": 1.0} 



def get_emotion_color(emotion):
    """Get color for emotion visualization"""
    colors = {
        'Anger': '#ff4444',
        'Disgust': '#8b4513',
        'Surprise': '#ffa500',
        'Sadness': '#4169e1',
        'Neutral': '#808080',
        'Sarcasm': '#9932cc',
        'Fear': '#2f4f4f',
        'Pride': '#ffd700',
        'HOF': '#ff69b4',  # Hall of Fame
        'Joy': '#32cd32',
        'Happy': '#32cd32',
        'NOT': '#32cd32',
        'HOF': '#ff4444',
        'Hate': '#8b0000',
        'Civil': '#228b22'
    }
    return colors.get(emotion, '#666666')

def create_emotion_badge(emotion, confidence=None):
    """Create a styled emotion badge"""
    color = get_emotion_color(emotion)
    confidence_text = f" ({confidence:.2%})" if confidence else ""
    return f'<span class="emotion-badge" style="background-color: {color};">{emotion}{confidence_text}</span>'

def main():
    # Main title
    st.markdown('<h1 class="main-header">🎭 Marathi Hate Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Home", "Model Training", "Single Prediction", "Batch Analysis", "Model Comparison"])
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = MarathiTextProcessor()
    if 'classifier' not in st.session_state:
        st.session_state.classifier = MarathiTextClassifier()
    if 'bert_classifier' not in st.session_state:
        st.session_state.bert_classifier = BERTMarathiClassifier()
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'bert_loaded' not in st.session_state:
        st.session_state.bert_loaded = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if page == "Home":
        show_home_page()
    elif page == "Model Training":
        show_training_page()
    elif page == "Single Prediction":
        show_prediction_page()
    elif page == "Batch Analysis":
        show_batch_analysis_page()
    elif page == "Model Comparison":
        show_comparison_page()

def show_home_page():
    """Display the home page"""
    st.markdown("""
    ## Welcome to Marathi Hate Sentiment Analysis Dashboard! 🚀
    
    This application provides comprehensive sentiment analysis for Marathi text using multiple machine learning approaches:
    
    ### 🔧 Available Models:
    - **Traditional ML Models**: Logistic Regression, Naive Bayes, SVM, Random Forest, SGD Classifier
    - **Ensemble Model**: Voting classifier combining best performers
    - **BERT Model**: State-of-the-art transformer model for Indic languages
    
    
    """)
    
    # emotions = ["Anger/Disgust", "Surprise", "Sadness", "Neutral", "Sarcasm", "Fear", "Pride", "HOF"]
    
    # cols = st.columns(4)
    # for i, emotion in enumerate(emotions):
    #     with cols[i % 4]:
    #         st.markdown(create_emotion_badge(emotion), unsafe_allow_html=True)
    
    st.markdown("""
    
    ### 🎯 Features:
    - **Real-time Prediction**: Analyze individual texts instantly
    - **Batch Processing**: Upload CSV files for bulk analysis
    - **Model Comparison**: Compare performance across different algorithms
    - **Visualization**: Interactive charts and confidence scores
    - **Multi-lingual Support**: Optimized for Marathi text processing
    
    ### 🚀 Getting Started:
    1. **Model Training**: Load your dataset and train models
    2. **Single Prediction**: Test individual text samples
    3. **Batch Analysis**: Process multiple texts at once
    4. **Model Comparison**: Evaluate different approaches
    
    Navigate using the sidebar to get started!
    """)

def show_training_page():
    """Display the model training page"""
    st.title("🔧 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Dataset Loading and Model Training")
        
        # Dataset path input
        dataset_path = st.text_input("Dataset Path", value="Dataset", help="Path to your dataset folder")
        
        if st.button("🔄 Load Dataset", type="primary"):
            with st.spinner("Loading dataset..."):
                st.session_state.processor.dataset_path = dataset_path
                try:
                    train_df, valid_df, test_df = st.session_state.processor.prepare_combined_data()
                    
                    if train_df is not None and not train_df.empty:
                        st.session_state.data_loaded = True
                        st.success(f"✅ Dataset loaded successfully!")
                        st.markdown(f"**Train samples**: {len(train_df)}")
                        st.markdown(f"**Validation samples**: {len(valid_df) if valid_df is not None else 0}")
                        st.markdown(f"**Test samples**: {len(test_df) if test_df is not None else 0}")
                        
                        # Show label distribution with detailed analysis
                        st.markdown("### Label Distribution Analysis")
                        label_counts = train_df['label'].value_counts()
                        
                        # Calculate class balance metrics
                        min_samples = label_counts.min()
                        max_samples = label_counts.max()
                        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Classes", len(label_counts))
                        with col_b:
                            st.metric("Min Samples", min_samples)
                        with col_c:
                            st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}")
                        
                        if imbalance_ratio > 5:
                            st.warning("⚠️ Significant class imbalance detected! This may affect model performance.")
                        
                        # Visualization
                        fig = px.bar(x=label_counts.index, y=label_counts.values, 
                                   title="Sample Count per Emotion",
                                   color=label_counts.values,
                                   color_continuous_scale="viridis")
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Text length analysis
                        train_df['text_length'] = train_df['text'].str.len()
                        train_df['word_count'] = train_df['text'].str.split().str.len()
                        
                        st.markdown("### Text Quality Analysis")
                        col_d, col_e, col_f = st.columns(3)
                        with col_d:
                            st.metric("Avg Text Length", f"{train_df['text_length'].mean():.0f}")
                        with col_e:
                            st.metric("Avg Word Count", f"{train_df['word_count'].mean():.1f}")
                        with col_f:
                            short_texts = (train_df['word_count'] < 3).sum()
                            st.metric("Very Short Texts", f"{short_texts} ({short_texts/len(train_df)*100:.1f}%)")
                        
                        if short_texts > len(train_df) * 0.2:
                            st.warning("⚠️ Many texts are very short (<3 words). This may impact model performance.")
                        
                        fig = px.pie(values=label_counts.values, names=label_counts.index, 
                                   title="Training Data Label Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show sample data
                        st.markdown("### Sample Data")
                        st.dataframe(train_df[['text', 'label']].head(10))
                        
                    else:
                        st.error("❌ No data found! Please check your dataset path and file structure.")
                        
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
        
        st.markdown("---")
        
        # Traditional ML Models Training
        st.markdown("### 🤖 Traditional ML Models")
        if st.button("🚀 Train Traditional Models", type="primary"):
            if st.session_state.processor.train_df is None or len(st.session_state.processor.train_df) == 0:
                st.error("Please load dataset first!")
            else:
                with st.spinner("Training models... This may take a few minutes."):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Preparing data...")
                        progress_bar.progress(20)
                        
                        models, vectorizer, scores, f1_scores, X_test, y_test = st.session_state.classifier.train_models(
                            st.session_state.processor.train_df,
                            st.session_state.processor.valid_df,
                            st.session_state.processor.test_df
                        )
                        
                        progress_bar.progress(100)
                        st.session_state.models_trained = True
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Display results
                        st.markdown("### 📊 Model Performance")
                        
                        results_df = pd.DataFrame({
                            'Model': list(scores.keys()),
                            'Accuracy': list(scores.values()),
                            'F1 Score': list(f1_scores.values())
                        })
                        
                        st.dataframe(results_df.round(4))
                        
                        # Visualization
                        fig = make_subplots(rows=1, cols=2, 
                                          subplot_titles=['Accuracy Scores', 'F1 Scores'])
                        
                        fig.add_trace(
                            go.Bar(x=list(scores.keys()), y=list(scores.values()), 
                                   name='Accuracy', marker_color='lightblue'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(x=list(f1_scores.keys()), y=list(f1_scores.values()), 
                                   name='F1 Score', marker_color='lightcoral'),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error training models: {str(e)}")
        
        st.markdown("---")
        
        # BERT Model Section
        st.markdown("### 🤗 BERT Model")
        
        # bert_model_name = st.selectbox("Select BERT Model", [
        #     "ai4bharat/indic-bert",
        #     "google/muril-base-cased",
        #     "bert-base-multilingual-cased"
        # ])

        bert_model_name = st.selectbox("Select BERT Model", [
    "l3cube-pune/marathi-sentiment-tweets",  # L3Cube Marathi sentiment model (same as app2.py)
    "l3cube-pune/marathi-bert",              # Marathi-specific BERT
    "ai4bharat/indic-bert",                 # General Indic languages
    "bert-base-multilingual-cased",         # Multilingual BERT
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # Zero-shot model
        ])
        
        if st.button("🔥 Load BERT Model", type="primary"):
            if not BERT_AVAILABLE:
                st.error("❌ Transformers library not available. Please install: pip install transformers torch")
            else:
                with st.spinner("Loading BERT model... This may take a few minutes."):
                    # Try cached loading first (faster)
                    cached_pipeline = load_bert_model_cached(bert_model_name)
                    if cached_pipeline:
                        st.session_state.bert_classifier.pipeline = cached_pipeline
                        st.session_state.bert_classifier.is_trained = True
                        st.session_state.bert_loaded = True
                        st.success("✅ BERT model loaded successfully!")
                    else:
                        # Fallback to the class method
                        success = st.session_state.bert_classifier.load_pretrained_model(bert_model_name)
                        if success:
                            st.session_state.bert_loaded = True
                            st.success("✅ BERT model loaded successfully!")
                        else:
                            st.error("❌ Failed to load BERT model. Check your internet connection.")
    
    with col2:
        st.markdown("### 📈 Training Status")
        
        # Status indicators
        st.markdown("#### Traditional Models")
        if st.session_state.models_trained:
            st.success("✅ Trained")
        else:
            st.warning("⏳ Not trained")
        
        st.markdown("#### BERT Model")
        if st.session_state.bert_loaded:
            st.success("✅ Loaded")
        else:
            st.warning("⏳ Not loaded")
        
        # Quick stats
        if hasattr(st.session_state.processor, 'train_df') and st.session_state.processor.train_df is not None and not st.session_state.processor.train_df.empty:
            st.markdown("### 📊 Dataset Stats")
            st.metric("Training Samples", len(st.session_state.processor.train_df))
            if st.session_state.processor.valid_df is not None and not st.session_state.processor.valid_df.empty:
                st.metric("Validation Samples", len(st.session_state.processor.valid_df))
            if st.session_state.processor.test_df is not None and not st.session_state.processor.test_df.empty:
                st.metric("Test Samples", len(st.session_state.processor.test_df))

def show_prediction_page():
    """Display the single prediction page"""
    st.title("🔍 Single Text Prediction")
    
    # Input text
    st.markdown("### Enter Marathi Text for Analysis")
    user_text = st.text_area(
        "Text Input", 
        placeholder="मराठी मजकूर येथे लिहा... (Enter Marathi text here...)",
        height=150
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        traditional_model = st.selectbox("Traditional ML Model", 
                                       ["Logistic Regression", "Ensemble", "Naive Bayes", 
                                        "Linear SVM", "SGD Classifier", "Random Forest"])
    
    with col2:
        use_bert = st.checkbox("Use BERT Model", value=True)
    
    if st.button("🎯 Analyze Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text to analyze!")
        else:
            analysis_cols = st.columns(2)
            
            # Traditional ML Analysis
            with analysis_cols[0]:
                st.markdown("### 🤖 Traditional ML Results")
                
                if st.session_state.models_trained:
                    try:
                        prediction, probabilities = st.session_state.classifier.predict(
                            user_text, traditional_model
                        )
                        
                        st.markdown(f"**Predicted Emotion**: {create_emotion_badge(prediction)}", 
                                  unsafe_allow_html=True)
                        
                        # Show confidence scores
                        st.markdown("**Confidence Scores:**")
                        for emotion, confidence in sorted(probabilities.items(), 
                                                        key=lambda x: x[1], reverse=True):
                            st.progress(confidence, text=f"{emotion}: {confidence:.2%}")
                        
                    except Exception as e:
                        st.error(f"Error in traditional ML prediction: {str(e)}")
                else:
                    st.warning("⚠️ Traditional models not trained yet!")
            
            # BERT Analysis
            with analysis_cols[1]:
                st.markdown("### 🤗 BERT Results")
                
                if use_bert and st.session_state.bert_loaded:
                    try:
                        with st.spinner("Analyzing with BERT..."):
                            bert_prediction, bert_probabilities = st.session_state.bert_classifier.predict(user_text)
                        
                        st.markdown(f"**Predicted Emotion**: {create_emotion_badge(bert_prediction)}", 
                                  unsafe_allow_html=True)
                        
                        # Show confidence scores
                        st.markdown("**Confidence Scores:**")
                        for emotion, confidence in sorted(bert_probabilities.items(), 
                                                        key=lambda x: x[1], reverse=True):
                            st.progress(confidence, text=f"{emotion}: {confidence:.2%}")
                        
                    except Exception as e:
                        st.error(f"Error in BERT prediction: {str(e)}")
                elif use_bert and not st.session_state.bert_loaded:
                    st.warning("⚠️ BERT model not loaded yet!")
                else:
                    st.info("BERT analysis disabled")
    
    # Sample texts for testing
    # st.markdown("---")
    # st.markdown("### 📝 Sample Texts for Testing")
    
    # sample_texts = {
    #     "Anger": "हे खूप वाईट आहे! मला राग येत आहे.",
    #     "Joy": "आज खूप आनंद झाला! छान दिवस होता.",
    #     "Sadness": "मला खूप दुःख होत आहे.",
    #     "Surprise": "वाह! हे तर आश्चर्यकारक आहे!",
    #     "Fear": "मला भीती वाटत आहे.",
    #     "Neutral": "आज हवामान सामान्य आहे."
    # }
    
    # sample_cols = st.columns(3)
    # for i, (emotion, text) in enumerate(sample_texts.items()):
    #     with sample_cols[i % 3]:
    #         if st.button(f"{emotion}: {text[:20]}..."):
    #             st.session_state.sample_text = text
    #             st.rerun()
    
    # if hasattr(st.session_state, 'sample_text'):
    #     st.text_area("Selected Sample", value=st.session_state.sample_text, key="sample_display")

def show_batch_analysis_page():
    """Display the batch analysis page"""
    st.title("📊 Batch Analysis")
    
    st.markdown("### Upload CSV File for Batch Processing")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown("### 📋 File Preview")
            st.dataframe(df.head())
            
            # Column selection
            text_column = st.selectbox("Select text column", df.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                use_traditional = st.checkbox("Use Traditional ML", value=True)
                if use_traditional:
                    selected_model = st.selectbox("Select Model", 
                                                 ["Ensemble", "Logistic Regression", "Naive Bayes", 
                                                  "Linear SVM", "SGD Classifier", "Random Forest"])
            
            with col2:
                use_bert_batch = st.checkbox("Use BERT", value=False)
            
            if st.button("🚀 Process Batch", type="primary"):
                if not use_traditional and not use_bert_batch:
                    st.warning("Please select at least one model!")
                else:
                    results = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, text in enumerate(df[text_column]):
                        status_text.text(f"Processing {i+1}/{len(df)}...")
                        
                        result = {'original_text': text, 'processed_text': MarathiTextProcessor.preprocess_marathi_text(text)}
                        
                        # Traditional ML prediction
                        if use_traditional and st.session_state.models_trained:
                            try:
                                pred, probs = st.session_state.classifier.predict(text, selected_model)
                                result['traditional_prediction'] = pred
                                result['traditional_confidence'] = max(probs.values()) if probs else 0
                            except:
                                result['traditional_prediction'] = 'Error'
                                result['traditional_confidence'] = 0
                        
                        # BERT prediction
                        if use_bert_batch and st.session_state.bert_loaded:
                            try:
                                bert_pred, bert_probs = st.session_state.bert_classifier.predict(text)
                                result['bert_prediction'] = bert_pred
                                result['bert_confidence'] = max(bert_probs.values()) if bert_probs else 0
                            except:
                                result['bert_prediction'] = 'Error'
                                result['bert_confidence'] = 0
                        
                        results.append(result)
                        progress_bar.progress((i + 1) / len(df))
                    
                    status_text.text("Processing complete!")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.markdown("### 📊 Results")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    if use_traditional and 'traditional_prediction' in results_df.columns:
                        st.markdown("#### Traditional ML Results Distribution")
                        traditional_counts = results_df['traditional_prediction'].value_counts()
                        fig1 = px.pie(values=traditional_counts.values, names=traditional_counts.index,
                                     title="Traditional ML Predictions")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    if use_bert_batch and 'bert_prediction' in results_df.columns:
                        st.markdown("#### BERT Results Distribution")
                        bert_counts = results_df['bert_prediction'].value_counts()
                        fig2 = px.pie(values=bert_counts.values, names=bert_counts.index,
                                     title="BERT Predictions")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Comparison if both models used
                    if use_traditional and use_bert_batch and 'traditional_prediction' in results_df.columns and 'bert_prediction' in results_df.columns:
                        st.markdown("#### Model Agreement Analysis")
                        agreement = (results_df['traditional_prediction'] == results_df['bert_prediction']).mean()
                        st.metric("Agreement Rate", f"{agreement:.2%}")
                        
                        # Confusion matrix between models
                        fig3 = px.density_heatmap(
                            results_df, 
                            x='traditional_prediction', 
                            y='bert_prediction',
                            title="Model Predictions Comparison"
                        )
                        st.plotly_chart(fig3, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_comparison_page():
    """Display the model comparison page"""
    st.title("⚖️ Model Comparison")
    
    if not st.session_state.models_trained and not st.session_state.bert_loaded:
        st.warning("Please train models first to see comparisons!")
        return
    
    st.markdown("### 📊 Model Performance Comparison")
    
    # Performance metrics visualization
    if st.session_state.models_trained:
        # Traditional model scores
        scores_df = pd.DataFrame({
            'Model': list(st.session_state.classifier.model_scores.keys()),
            'Accuracy': list(st.session_state.classifier.model_scores.values()),
            'F1_Score': list(st.session_state.classifier.model_f1_scores.values())
        })
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy Comparison', 'F1 Score Comparison', 
                          'Combined Metrics', 'Model Rankings'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Accuracy bars
        fig.add_trace(
            go.Bar(x=scores_df['Model'], y=scores_df['Accuracy'], 
                   name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        
        # F1 Score bars
        fig.add_trace(
            go.Bar(x=scores_df['Model'], y=scores_df['F1_Score'], 
                   name='F1 Score', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Scatter plot: Accuracy vs F1
        fig.add_trace(
            go.Scatter(x=scores_df['Accuracy'], y=scores_df['F1_Score'],
                      mode='markers+text', text=scores_df['Model'],
                      textposition="top center", name='Models',
                      marker=dict(size=12, color='green')),
            row=2, col=1
        )
        
        # Overall ranking (combined score)
        scores_df['Combined_Score'] = (scores_df['Accuracy'] + scores_df['F1_Score']) / 2
        scores_df_sorted = scores_df.sort_values('Combined_Score', ascending=True)
        
        fig.add_trace(
            go.Bar(x=scores_df_sorted['Combined_Score'], y=scores_df_sorted['Model'],
                   orientation='h', name='Combined Score', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_xaxes(title_text="Models", row=1, col=2)
        fig.update_xaxes(title_text="Accuracy", row=2, col=1)
        fig.update_xaxes(title_text="Combined Score", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="F1 Score", row=1, col=2)
        fig.update_yaxes(title_text="F1 Score", row=2, col=1)
        fig.update_yaxes(title_text="Models", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model recommendation
        best_model = scores_df.loc[scores_df['Combined_Score'].idxmax()]
        st.markdown("### 🏆 Best Model Recommendation")
        st.success(f"""
        **Recommended Model**: {best_model['Model']}
        - **Accuracy**: {best_model['Accuracy']:.4f}
        - **F1 Score**: {best_model['F1_Score']:.4f}
        - **Combined Score**: {best_model['Combined_Score']:.4f}
        """)
    
    # Model characteristics table
    st.markdown("### 📋 Model Characteristics")
    
    characteristics = {
        'Model': ['Logistic Regression', 'Naive Bayes', 'Linear SVM', 'SGD Classifier', 'Random Forest', 'Ensemble', 'BERT'],
        'Speed': ['Fast', 'Very Fast', 'Fast', 'Very Fast', 'Moderate', 'Moderate', 'Slow'],
        'Memory Usage': ['Low', 'Low', 'Low', 'Low', 'High', 'High', 'Very High'],
        'Interpretability': ['High', 'High', 'Medium', 'Medium', 'Low', 'Low', 'Low'],
        'Best For': ['Balanced datasets', 'Small datasets', 'Large datasets', 'Online learning', 'Complex patterns', 'Stability', 'State-of-the-art accuracy']
    }
    
    char_df = pd.DataFrame(characteristics)
    st.dataframe(char_df, use_container_width=True)
    
    # Interactive model selector for detailed analysis
    st.markdown("### 🔍 Detailed Model Analysis")
    
    if st.session_state.models_trained:
        selected_model = st.selectbox("Select model for detailed analysis", 
                                    list(st.session_state.classifier.models.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{st.session_state.classifier.model_scores[selected_model]:.4f}")
        
        with col2:
            st.metric("F1 Score", f"{st.session_state.classifier.model_f1_scores[selected_model]:.4f}")
        
        # Model-specific insights
        model_insights = {
            'Logistic Regression': "Good balance of speed and accuracy. Provides probability estimates.",
            'Naive Bayes': "Very fast and works well with small datasets. Assumes feature independence.",
            'Linear SVM': "Excellent for text classification. Good generalization.",
            'SGD Classifier': "Very fast, suitable for online learning and large datasets.",
            'Random Forest': "Handles complex patterns well. Less prone to overfitting.",
            'Ensemble': "Combines multiple models for better stability and accuracy.",
        }
        
        if selected_model in model_insights:
            st.info(f"**Insight**: {model_insights[selected_model]}")

if __name__ == "__main__":
    main()