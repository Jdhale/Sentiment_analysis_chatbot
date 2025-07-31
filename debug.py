import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                    logger.info(f"Raw emotion {split} shape: {df.shape}")
                    logger.info(f"Sample emotion {split} data:\n{df.head(2)}")
                    
                    # Standardize column names - check what columns actually exist
                    if 'text' not in df.columns:
                        # Try common alternatives
                        text_cols = [col for col in df.columns if 'text' in col.lower() or 'sentence' in col.lower() or 'comment' in col.lower()]
                        if text_cols:
                            df = df.rename(columns={text_cols[0]: 'text'})
                            logger.info(f"Renamed {text_cols[0]} to 'text'")
                    
                    if 'label' not in df.columns:
                        # Try common alternatives
                        label_cols = [col for col in df.columns if 'label' in col.lower() or 'category' in col.lower() or 'sentiment' in col.lower()]
                        if label_cols:
                            df = df.rename(columns={label_cols[0]: 'label'})
                            logger.info(f"Renamed {label_cols[0]} to 'label'")
                    
                    # Ensure we have required columns
                    if 'text' not in df.columns or 'label' not in df.columns:
                        logger.error(f"Required columns missing in {filepath}. Available: {list(df.columns)}")
                        emotion_dfs[split] = pd.DataFrame(columns=['text', 'label'])
                        continue
                    
                    # Clean data
                    df = df[['text', 'label']].dropna().drop_duplicates().reset_index(drop=True)
                    emotion_dfs[split] = df
                    logger.info(f"Loaded {len(df)} emotion {split} samples after cleaning")
                else:
                    logger.warning(f"File not found: {filepath}")
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
                    logger.info(f"Raw hate {split} columns: {list(df.columns)}")
                    logger.info(f"Raw hate {split} shape: {df.shape}")
                    logger.info(f"Sample hate {split} data:\n{df.head(2)}")
                    
                    # Standardize column names
                    if 'sentence' in df.columns:
                        df = df.rename(columns={'sentence': 'text'})
                        logger.info("Renamed 'sentence' to 'text'")
                    if 'category' in df.columns:
                        df = df.rename(columns={'category': 'label'})
                        logger.info("Renamed 'category' to 'label'")
                    
                    # Try other common column names
                    if 'text' not in df.columns:
                        text_cols = [col for col in df.columns if 'text' in col.lower() or 'sentence' in col.lower() or 'comment' in col.lower()]
                        if text_cols:
                            df = df.rename(columns={text_cols[0]: 'text'})
                            logger.info(f"Renamed {text_cols[0]} to 'text'")
                    
                    if 'label' not in df.columns:
                        label_cols = [col for col in df.columns if 'label' in col.lower() or 'category' in col.lower() or 'sentiment' in col.lower()]
                        if label_cols:
                            df = df.rename(columns={label_cols[0]: 'label'})
                            logger.info(f"Renamed {label_cols[0]} to 'label'")
                    
                    # Ensure we have required columns
                    if 'text' not in df.columns or 'label' not in df.columns:
                        logger.error(f"Required columns missing in {filepath}. Available: {list(df.columns)}")
                        hate_dfs[split] = pd.DataFrame(columns=['text', 'label'])
                        continue
                    
                    df = df[['text', 'label']].dropna().drop_duplicates().reset_index(drop=True)
                    hate_dfs[split] = df
                    logger.info(f"Loaded {len(df)} hate {split} samples after cleaning")
                else:
                    logger.warning(f"File not found: {filepath}")
                    hate_dfs[split] = pd.DataFrame(columns=['text', 'label'])
            
            # Debug: Check individual dataframes before combining
            logger.info("=== PRE-COMBINATION DEBUG ===")
            logger.info(f"Emotion train: {len(emotion_dfs.get('train', pd.DataFrame()))}")
            logger.info(f"Hate train: {len(hate_dfs.get('train', pd.DataFrame()))}")
            logger.info(f"Hate civil: {len(hate_dfs.get('civil', pd.DataFrame()))}")
            
            # Combine hate train with civil data
            if 'train' in hate_dfs and 'civil' in hate_dfs and len(hate_dfs['civil']) > 0:
                hate_train_combined = pd.concat([hate_dfs['train'], hate_dfs['civil']], 
                                              ignore_index=True)
                hate_train_combined = hate_train_combined.dropna().drop_duplicates().reset_index(drop=True)
                logger.info(f"Combined hate train: {len(hate_train_combined)}")
            else:
                hate_train_combined = hate_dfs.get('train', pd.DataFrame(columns=['text', 'label']))
                logger.info(f"Using only hate train: {len(hate_train_combined)}")
            
            # Combine emotion and hate datasets
            logger.info("=== COMBINING DATASETS ===")
            
            # Train dataset
            train_parts = []
            if len(emotion_dfs['train']) > 0:
                train_parts.append(emotion_dfs['train'])
                logger.info(f"Adding emotion train: {len(emotion_dfs['train'])}")
            if len(hate_train_combined) > 0:
                train_parts.append(hate_train_combined)
                logger.info(f"Adding hate train: {len(hate_train_combined)}")
            
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
            
            logger.info(f"Final combined datasets - Train: {len(self.train_df)}, Valid: {len(self.valid_df)}, Test: {len(self.test_df)}")
            
            # Debug: Show sample data if available
            if len(self.train_df) > 0:
                logger.info(f"Sample train data:\n{self.train_df.head(2)}")
            if len(self.valid_df) > 0:
                logger.info(f"Sample valid data:\n{self.valid_df.head(2)}")
            
            return self.train_df, self.valid_df, self.test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    @staticmethod
    def preprocess_marathi_text(text):
        """Preprocess Marathi text for sentiment analysis"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep Devanagari script and basic punctuation
        text = re.sub(r'[^\u0900-\u097F\s\.,!?।]', ' ', text)
        
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
        
        logger.info("Text preprocessing completed")
        
        return self.train_df, self.valid_df, self.test_df
    
    def get_dataset_info(self):
        """Get information about the datasets"""
        if self.train_df is None:
            logger.warning("Data not loaded yet. Call prepare_combined_data() first.")
            return
        
        info = {
            'train_size': len(self.train_df),
            'valid_size': len(self.valid_df),
            'test_size': len(self.test_df),
            'train_labels': self.train_df['label'].value_counts().to_dict(),
            'valid_labels': self.valid_df['label'].value_counts().to_dict(),
            'test_labels': self.test_df['label'].value_counts().to_dict()
        }
        
        return info
    
    def display_sample_data(self, n_samples=5):
        """Display sample data from each dataset"""
        if self.train_df is None:
            logger.warning("Data not loaded yet. Call prepare_combined_data() first.")
            return
        
        print("=== TRAIN SAMPLES ===")
        print(self.train_df[['text', 'label', 'processed_text']].head(n_samples))
        
        print("\n=== VALIDATION SAMPLES ===")
        print(self.valid_df[['text', 'label', 'processed_text']].head(n_samples))
        
        print("\n=== TEST SAMPLES ===")
        print(self.test_df[['text', 'label', 'processed_text']].head(n_samples))

class MarathiTextClassifier:
    """Class to handle model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.model_scores = {}
        self.model_f1_scores = {}
        
    def train_models(self, train_df, valid_df=None, test_df=None):
        """Train multiple models on the dataset"""
        
        # Check if we have data
        if len(train_df) == 0:
            raise ValueError("Training dataset is empty! Please check your data loading.")
        
        # Prepare data
        X_train = train_df['processed_text']
        y_train = train_df['label']
        
        # Check if processed text is empty
        if X_train.str.len().sum() == 0:
            raise ValueError("All processed text is empty! Please check your text preprocessing.")
        
        logger.info(f"Training with {len(X_train)} samples")
        logger.info(f"Label distribution: {y_train.value_counts().to_dict()}")
        
        if valid_df is not None and test_df is not None and len(valid_df) > 0 and len(test_df) > 0:
            X_valid = valid_df['processed_text']
            y_valid = valid_df['label']
            X_test = test_df['processed_text']
            y_test = test_df['label']
        else:
            # Split if validation/test not provided or empty
            if len(y_train.unique()) < 2:
                raise ValueError("Need at least 2 different labels for training!")
                
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
            )
        
        # Vectorize text
        logger.info("Vectorizing text data...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,
            min_df=2,
            max_df=0.95
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_valid_tfidf = self.vectorizer.transform(X_valid)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train individual models
        logger.info("Training individual models...")
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_tfidf, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            self.models[name] = model
            self.model_scores[name] = accuracy
            self.model_f1_scores[name] = f1
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        ensemble = VotingClassifier([
            ('lr', models['Logistic Regression']),
            ('nb', models['Naive Bayes']),
            ('svm', models['SVM']),
            ('rf', models['Random Forest'])
        ], voting='soft')
        
        ensemble.fit(X_train_tfidf, y_train)
        ensemble_pred = ensemble.predict(X_test_tfidf)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
        
        self.models['Ensemble'] = ensemble
        self.model_scores['Ensemble'] = ensemble_accuracy
        self.model_f1_scores['Ensemble'] = ensemble_f1
        
        logger.info(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
        
        return self.models, self.vectorizer, self.model_scores, self.model_f1_scores, X_test, y_test
    
    def get_best_model(self):
        """Get the best performing model based on F1 score"""
        if not self.model_f1_scores:
            return None, 0
        
        best_model_name = max(self.model_f1_scores, key=self.model_f1_scores.get)
        best_score = self.model_f1_scores[best_model_name]
        
        return best_model_name, best_score
    
    def predict(self, text, model_name='Ensemble'):
        """Predict sentiment for given text"""
        if model_name not in self.models or self.vectorizer is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess text
        processed_text = MarathiTextProcessor.preprocess_marathi_text(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.models[model_name].predict(text_tfidf)[0]
        probabilities = self.models[model_name].predict_proba(text_tfidf)[0]
        
        return prediction, probabilities

# Usage example
if __name__ == "__main__":
    # Initialize processor
    processor = MarathiTextProcessor()
    
    # Load and prepare data
    train_df, valid_df, test_df = processor.prepare_combined_data()
    
    # Display dataset information
    info = processor.get_dataset_info()
    print("Dataset Information:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Display sample data
    processor.display_sample_data()
    
    # Initialize and train classifier
    classifier = MarathiTextClassifier()
    models, vectorizer, scores, f1_scores, X_test, y_test = classifier.train_models(
        train_df, valid_df, test_df
    )
    
    # Display results
    print("\n=== MODEL PERFORMANCE ===")
    for model_name in scores:
        print(f"{model_name}: Accuracy={scores[model_name]:.4f}, F1={f1_scores[model_name]:.4f}")
    
    # Get best model
    best_model, best_score = classifier.get_best_model()
    print(f"\nBest Model: {best_model} (F1 Score: {best_score:.4f})")