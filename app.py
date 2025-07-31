# Marathi Sentiment Analysis using NLP and Streamlit
# This project uses L3CubeMahaSent dataset and MahaBERT model

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# emotion_train_df = pd.read_csv("Dataset/emotion/emotion_train.csv")
# emotion_valid_df = pd.read_csv("Dataset/emotion/emotion_valid.csv")
# emotion_test_df = pd.read_csv("Dataset/emotion/emotion_test.csv")
# # emotion_train_df = pd.concat([emotion_train_df, emotion_valid_df], ignore_index=True)
# emotion_train_df = emotion_train_df.dropna()
# emotion_train_df = emotion_train_df.drop_duplicates()
# emotion_train_df = emotion_train_df.reset_index(drop=True)

# hate_train_df = pd.read_csv("Dataset/hate/hate_bin_train.csv")
# civil_df = pd.read_csv("Dataset/hate/civil_hate_augmented.csv")
# hate_valid_df = pd.read_csv("Dataset/hate/hate_bin_valid.csv")
# hate_test_df = pd.read_csv("Dataset/hate/hate_bin_test.csv")
# hate_train_df = pd.concat([hate_train_df, civil_df], ignore_index=True)
# hate_train_df = hate_train_df.dropna()
# hate_train_df = hate_train_df.drop_duplicates()
# hate_train_df = hate_train_df.reset_index(drop=True)

# train_df = pd.concat([emotion_train_df, hate_train_df], ignore_index=True)
# valid_df = pd.concat([emotion_valid_df, hate_valid_df], ignore_index=True)
# test_df = pd.concat([emotion_test_df, hate_test_df], ignore_index=True)

# train_df = train_df.dropna()
# train_df = train_df.drop_duplicates()
# train_df = train_df.reset_index(drop=True)
# valid_df = valid_df.dropna()
# valid_df = valid_df.drop_duplicates()
# valid_df = valid_df.reset_index(drop=True)
# test_df = test_df.dropna()
# test_df = test_df.drop_duplicates()
# test_df = 

# For BERT models (optional - requires transformers library)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers library not available. Using traditional ML models only.")

# Configure Streamlit page
st.set_page_config(
    page_title="Marathi Sentiment Analysis",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        padding-top: 0px;
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        color: #2c3e50;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .prediction-negative {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .prediction-neutral {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔍 मराठी भावना विश्लेषण</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #666;">Marathi Sentiment Analysis using NLP</h2>', unsafe_allow_html=True)

# -----------------------------
# Use merged dataset
# -----------------------------
@st.cache_data

def load_combined_data():
    # Load and clean emotion data
    emotion_train_df = pd.read_csv("Dataset/emotion/train.csv")
    emotion_valid_df = pd.read_csv("Dataset/emotion/validation.csv")
    emotion_test_df = pd.read_csv("Dataset/emotion/test.csv")
    emotion_train_df = emotion_train_df.dropna().drop_duplicates().reset_index(drop=True)

    # Load and clean hate data
    hate_train_df = pd.read_csv("Dataset/hate/hate_bin_train.csv")
    civil_df = pd.read_csv("Dataset/hate/civil_hate_augmented.csv")
    hate_valid_df = pd.read_csv("Dataset/hate/hate_bin_valid.csv")
    hate_test_df = pd.read_csv("Dataset/hate/hate_bin_test.csv")
    hate_train_df = pd.concat([hate_train_df, civil_df], ignore_index=True)
    hate_train_df = hate_train_df.dropna().drop_duplicates().reset_index(drop=True)

    # Merge train, valid, test datasets
    train_df = pd.concat([emotion_train_df, hate_train_df], ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
    valid_df = pd.concat([emotion_valid_df, hate_valid_df], ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
    test_df = pd.concat([emotion_test_df, hate_test_df], ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)

    return train_df, valid_df, test_df


# Text preprocessing for Marathi (Barabar aahe as previous code)
def preprocess_marathi_text(text):
    """Preprocess Marathi text for sentiment analysis"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters but keep Devanagari script
    text = re.sub(r'[^\u0900-\u097F\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Convert to lowercase (for Devanagari, this doesn't change much but good practice)
    text = text.strip()
    
    return text


# -----------------------------
# Prepare data(Navin try)
# -----------------------------
@st.cache_data

def prepare_combined_data():
    train_df, valid_df, test_df = load_combined_data()
    train_df['processed_text'] = train_df['text'].apply(preprocess_marathi_text)
    valid_df['processed_text'] = valid_df['text'].apply(preprocess_marathi_text)
    test_df['processed_text'] = test_df['text'].apply(preprocess_marathi_text)
    return train_df, valid_df, test_df



# -----------------------------
# Train models
# -----------------------------
@st.cache_resource

def train_models(train_df, valid_df=None, test_df=None):
    """
    Train models using the provided datasets.
    If valid_df and test_df are provided, use them for validation and testing.
    Otherwise, split the train_df for validation and testing.
    """
    # Use train_df for training
    X_train = train_df['processed_text']
    y_train = train_df['label']
    
    # If validation and test sets are provided, use them
    if valid_df is not None and test_df is not None:
        X_valid = valid_df['processed_text']
        y_valid = valid_df['label']
        X_test = test_df['processed_text']
        y_test = test_df['label']
    else:
        # Split train_df for validation and testing
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=None
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    trained_models = {}
    model_scores = {}
    model_f1_scores = {}

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        trained_models[name] = model
        model_scores[name] = accuracy
        model_f1_scores[name] = f1

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

    trained_models['Ensemble'] = ensemble
    model_scores['Ensemble'] = ensemble_accuracy
    model_f1_scores['Ensemble'] = ensemble_f1

    return trained_models, vectorizer, model_scores, model_f1_scores, X_test, y_test


# Load BERT model (if available)
@st.cache_resource
def load_bert_model():
    """Load pre-trained Marathi BERT model for sentiment analysis"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # Try to load L3Cube Marathi BERT model
        model_name = "l3cube-pune/marathi-sentiment-tweets"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", 
                                    model=model, 
                                    tokenizer=tokenizer)
        return sentiment_pipeline
    except:
        st.warning("Could not load BERT model. Using traditional ML models.")
        return None





# Main app
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🏠 Home", "📊 Dataset Overview", "🤖 Model Training", "🔮 Prediction", "📈 Analytics"])
    
    # Load combined data
    train_df, valid_df, test_df = prepare_combined_data()
    
    if train_df is None:
        st.error("❌ Could not load datasets. Please check your file paths and ensure datasets are available.")
        st.stop()
    
    # Store data in session state
    if 'train_df' not in st.session_state:
        st.session_state.train_df = train_df
        st.session_state.valid_df = valid_df
        st.session_state.test_df = test_df
    
    if page == "🏠 Home":
        st.markdown("### Welcome to Marathi Emotion & Sentiment Analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Project Features")
            st.markdown("""
            - **Multi-task Analysis**: Combined emotion and hate speech detection
            - **L3Cube-MahaEmotions Dataset**: Using official Marathi emotion datasets
            - **Multiple Models**: Logistic Regression, SVM, Random Forest, Naive Bayes
            - **Ensemble Learning**: Combines multiple models for better accuracy
            - **BERT Integration**: Optional transformer-based analysis
            - **Interactive Dashboard**: Real-time predictions and analytics
            - **Marathi Text Support**: Proper Devanagari script handling
            """)
        
        with col2:
            st.markdown("#### 📚 Dataset Information")
            st.markdown(f"""
            - **Training Samples**: {len(train_df)} texts
            - **Validation Samples**: {len(valid_df)} texts
            - **Test Samples**: {len(test_df)} texts
            - **Total Classes**: {len(train_df['label'].unique())} different labels
            - **Language**: Marathi (मराठी)
            - **Source**: L3Cube-MahaEmotions (Emotion + Hate datasets)
            """)
        
        # Show dataset composition
        st.markdown("### 📊 Dataset Composition")
        
        # Combined class distribution
        all_labels = pd.concat([train_df['label'], valid_df['label'], test_df['label']])
        label_counts = all_labels.value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of label distribution
            fig_pie = px.pie(
                values=label_counts.values, 
                names=label_counts.index,
                title="Overall Label Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart of split distribution
            split_data = {
                'Train': len(train_df),
                'Validation': len(valid_df),
                'Test': len(test_df)
            }
            
            fig_bar = px.bar(
                x=list(split_data.keys()),
                y=list(split_data.values()),
                title="Data Split Distribution",
                color=list(split_data.keys()),
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Quick prediction demo
        st.markdown("### 🚀 Quick Demo")
        demo_text = st.text_area("Enter Marathi text for quick analysis:", 
                                value="आज खूप छान दिवस आहे!")
        
        if st.button("Analyze Text"):
            if demo_text:
                # Quick prediction using cached model
                if 'quick_model' not in st.session_state:
                    with st.spinner("Training quick model..."):
                        # Use a small subset for quick demo
                        sample_size = min(1000, len(train_df))
                        quick_train = train_df.sample(n=sample_size, random_state=42)
                        
                        X_quick = quick_train['processed_text']
                        y_quick = quick_train['label']
                        
                        vectorizer_quick = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
                        X_quick_tfidf = vectorizer_quick.fit_transform(X_quick)
                        
                        model_quick = LogisticRegression(random_state=42, max_iter=1000)
                        model_quick.fit(X_quick_tfidf, y_quick)
                        
                        st.session_state.quick_model = model_quick
                        st.session_state.quick_vectorizer = vectorizer_quick
                
                # Make prediction
                processed_text = preprocess_marathi_text(demo_text)
                text_tfidf = st.session_state.quick_vectorizer.transform([processed_text])
                prediction = st.session_state.quick_model.predict(text_tfidf)[0]
                probabilities = st.session_state.quick_model.predict_proba(text_tfidf)[0]
                confidence = max(probabilities)
                
                # Display result
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                           color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                    <h3>🎯 Prediction: {prediction.upper()}</h3>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif page == "📊 Dataset Overview":
        st.markdown("### 📊 Dataset Overview")
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📚 Training Data</h3>
                <h2>{len(train_df):,}</h2>
                <p>samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>✅ Validation Data</h3>
                <h2>{len(valid_df):,}</h2>
                <p>samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🧪 Test Data</h3>
                <h2>{len(test_df):,}</h2>
                <p>samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed analysis
        st.markdown("#### 📈 Detailed Dataset Analysis")
        
        # Label distribution across splits
        train_labels = train_df['label'].value_counts()
        valid_labels = valid_df['label'].value_counts()
        test_labels = test_df['label'].value_counts()
        
        # Create comparison DataFrame
        all_labels = sorted(set(train_labels.index) | set(valid_labels.index) | set(test_labels.index))
        comparison_data = []
        
        for label in all_labels:
            comparison_data.append({
                'Label': label,
                'Train': train_labels.get(label, 0),
                'Validation': valid_labels.get(label, 0),
                'Test': test_labels.get(label, 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.markdown("##### Label Distribution Across Splits")
        st.dataframe(comparison_df.set_index('Label'), use_container_width=True)
        
        # Visualization
        fig_comparison = px.bar(
            comparison_df.melt(id_vars=['Label'], var_name='Split', value_name='Count'),
            x='Label', y='Count', color='Split',
            title="Label Distribution Across Train/Validation/Test Splits",
            barmode='group'
        )
        fig_comparison.update_xaxis(tickangle=45)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Text length analysis
        st.markdown("#### 📏 Text Length Analysis")
        
        train_df['text_length'] = train_df['text'].str.len()
        valid_df['text_length'] = valid_df['text'].str.len()
        test_df['text_length'] = test_df['text'].str.len()
        
        length_data = []
        for split, df in [('Train', train_df), ('Validation', valid_df), ('Test', test_df)]:
            for _, row in df.iterrows():
                length_data.append({
                    'Split': split,
                    'Length': row['text_length'],
                    'Label': row['label']
                })
        
        length_df = pd.DataFrame(length_data)
        
        fig_length = px.box(
            length_df, x='Split', y='Length', color='Split',
            title="Text Length Distribution by Split"
        )
        st.plotly_chart(fig_length, use_container_width=True)
        
        # Sample data display
        st.markdown("#### 📝 Sample Data")
        
        sample_option = st.selectbox("Choose dataset split to view:", ["Training", "Validation", "Test"])
        
        if sample_option == "Training":
            sample_df = train_df
        elif sample_option == "Validation":
            sample_df = valid_df
        else:
            sample_df = test_df
        
        # Filter by label
        selected_labels = st.multiselect(
            "Filter by labels:", 
            options=sorted(sample_df['label'].unique()),
            default=sorted(sample_df['label'].unique())
        )
        
        if selected_labels:
            filtered_df = sample_df[sample_df['label'].isin(selected_labels)]
            st.dataframe(
                filtered_df[['text', 'label']].head(20),
                use_container_width=True
            )
        
    elif page == "🤖 Model Training":
        st.markdown("### 🤖 Model Training & Evaluation")
        
        # Training options
        col1, col2 = st.columns(2)
        
        with col1:
            use_full_data = st.checkbox("Use full dataset", value=True, 
                                      help="Uncheck to use a smaller sample for faster training")
        
        with col2:
            sample_size = st.slider("Sample size (if not using full data)", 
                                  min_value=100, max_value=5000, value=1000, step=100)
        
        if st.button("🚀 Train Models", type="primary"):
            # Prepare data based on user selection
            if use_full_data:
                final_train_df = train_df
                final_valid_df = valid_df
                final_test_df = test_df
                st.info(f"Training on full dataset: {len(train_df)} training samples")
            else:
                final_train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
                final_valid_df = valid_df.sample(n=min(sample_size//4, len(valid_df)), random_state=42)
                final_test_df = test_df.sample(n=min(sample_size//4, len(test_df)), random_state=42)
                st.info(f"Training on sample: {len(final_train_df)} training samples")
            
            # Train models
            with st.spinner("Training models... This may take a few minutes."):
                models, vectorizer, scores, f1_scores, X_test, y_test = train_models(
                    final_train_df, final_valid_df, final_test_df
                )
                
                # Store in session state
                st.session_state.models = models
                st.session_state.vectorizer = vectorizer
                st.session_state.model_scores = scores
                st.session_state.model_f1_scores = f1_scores
            
            st.success("✅ Models trained successfully!")
            
            # Display results
            st.markdown("#### 📈 Model Performance")
            
            # Create performance DataFrame
            performance_data = []
            for model_name in scores.keys():
                performance_data.append({
                    'Model': model_name,
                    'Accuracy': scores[model_name],
                    'F1-Score': f1_scores[model_name]
                })
            
            perf_df = pd.DataFrame(performance_data)
            
            # Display table
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(perf_df.set_index('Model'), use_container_width=True)
            
            with col2:
                # Best model highlight
                best_model = perf_df.loc[perf_df['Accuracy'].idxmax()]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #28a745, #20c997); 
                           color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h4>🏆 Best Model</h4>
                    <h3>{best_model['Model']}</h3>
                    <p>Accuracy: {best_model['Accuracy']:.3f}</p>
                    <p>F1-Score: {best_model['F1-Score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance visualization
            fig_perf = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Accuracy Comparison', 'F1-Score Comparison')
            )
            
            fig_perf.add_trace(
                go.Bar(x=perf_df['Model'], y=perf_df['Accuracy'], 
                      name='Accuracy', marker_color='#1f77b4'),
                row=1, col=1
            )
            
            fig_perf.add_trace(
                go.Bar(x=perf_df['Model'], y=perf_df['F1-Score'], 
                      name='F1-Score', marker_color='#ff7f0e'),
                row=1, col=2
            )
            
            fig_perf.update_layout(height=400, showlegend=False)
            fig_perf.update_xaxes(tickangle=45)
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Confusion Matrix for best model
            if len(X_test) > 0:
                st.markdown("#### 🔍 Confusion Matrix (Best Model)")
                
                best_model_name = best_model['Model']
                best_model_obj = models[best_model_name]
                
                # Get predictions
                X_test_tfidf = vectorizer.transform(X_test)
                y_pred = best_model_obj.predict(X_test_tfidf)
                
                # Create confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                labels = sorted(y_test.unique())
                
                fig_cm = px.imshow(
                    cm, 
                    text_auto=True,
                    title=f"Confusion Matrix - {best_model_name}",
                    labels=dict(x="Predicted", y="Actual"),
                    x=labels,
                    y=labels,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Classification report
                st.markdown("#### 📊 Detailed Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Convert to DataFrame for better display
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)
        
        # Show existing models if available
        elif 'models' in st.session_state:
            st.info("Using previously trained models. Click 'Train Models' to retrain.")
            
            # Display stored results
            scores = st.session_state.get('model_scores', {})
            f1_scores = st.session_state.get('model_f1_scores', {})
            
            if scores:
                st.markdown("#### 📈 Current Model Performance")
                
                performance_data = []
                for model_name in scores.keys():
                    performance_data.append({
                        'Model': model_name,
                        'Accuracy': scores[model_name],
                        'F1-Score': f1_scores.get(model_name, 0)
                    })
                
                perf_df = pd.DataFrame(performance_data)
                st.dataframe(perf_df.set_index('Model'), use_container_width=True)
    
    elif page == "🔮 Prediction":
        st.markdown("### 🔮 Text Classification & Prediction")
        
        # Check if models are trained
        if 'models' not in st.session_state:
            st.warning("⚠️ Please train models first in the 'Model Training' page.")
            return
        
        models = st.session_state.models
        vectorizer = st.session_state.vectorizer
        
        # Input methods
        input_method = st.radio("Choose input method:", 
                              ["Single Text", "Batch Processing"])
        
        if input_method == "Single Text":
            # Single text prediction
            user_text = st.text_area("Enter Marathi text:", 
                                   height=100,
                                   placeholder="मराठी मजकूर इथे टाका...")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_choice = st.selectbox("Choose Model:", 
                                          list(models.keys()),
                                          index=list(models.keys()).index('Ensemble') if 'Ensemble' in models else 0)
            
            with col2:
                show_probabilities = st.checkbox("Show all class probabilities", value=True)
            
            if st.button("🔍 Analyze Text", type="primary"):
                if user_text:
                    # Preprocess text
                    processed_text = preprocess_marathi_text(user_text)
                    
                    # Make prediction
                    text_tfidf = vectorizer.transform([processed_text])
                    selected_model = models[model_choice]
                    
                    prediction = selected_model.predict(text_tfidf)[0]
                    probabilities = selected_model.predict_proba(text_tfidf)[0]
                    confidence = max(probabilities)
                    
                    # Get class labels
                    classes = selected_model.classes_
                    
                    # Create probability distribution
                    prob_data = pd.DataFrame({
                        'Class': classes,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    # Display main prediction
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Determine color based on prediction type
                        if prediction in ['positive', 'joy', 'love', 'optimism']:
                            color = '#28a745'
                            emoji = '😊'
                        elif prediction in ['negative', 'sadness', 'anger', 'fear', 'pessimism', 'hate']:
                            color = '#dc3545'
                            emoji = '😞' if prediction != 'hate' else '😡'
                        else:
                            color = '#ffc107'
                            emoji = '😐'
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color}, #ffffff);
                                   padding: 2rem; border-radius: 15px; text-align: center;
                                   border: 2px solid {color};">
                            <h2>{emoji} {prediction.upper()}</h2>
                            <h3>Confidence: {confidence:.2%}</h3>
                            <p>Model: {model_choice}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if show_probabilities:
                            # Probability chart
                            fig_prob = px.bar(
                                prob_data.head(10), 
                                x='Probability', 
                                y='Class',
                                orientation='h',
                                title="Class Probabilities",
                                color='Probability',
                                color_continuous_scale='viridis'
                            )
                            fig_prob.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig_prob, use_container_width=True)
                        else:
                            # Show top 3 predictions
                            st.markdown("#### 🎯 Top Predictions:")
                            for i, row in prob_data.head(3).iterrows():
                                st.metric(
                                    f"#{i+1} {row['Class']}", 
                                    f"{row['Probability']:.1%}"
                                )
                    
                    # Text analysis
                    st.markdown("#### 📊 Text Analysis")
                    analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)
                    
                    with analysis_col1:
                        st.metric("Word Count", len(user_text.split()))
                    
                    with analysis_col2:
                        st.metric("Character Count", len(user_text))
                    
                    with analysis_col3:
                        st.metric("Processed Length", len(processed_text))
                    
                    with analysis_col4:
                        # Language detection confidence (simple heuristic)
                        marathi_chars = len(re.findall(r'[\u0900-\u097F]', user_text))
                        total_chars = len(re.sub(r'\s', '', user_text))
                        marathi_ratio = marathi_chars / max(total_chars, 1)
                        st.metric("Marathi Content", f"{marathi_ratio:.1%}")
                    
                    # BERT prediction (if available)
                    bert_model = load_bert_model()
                    if bert_model:
                        st.markdown("#### 🤖 BERT Model Prediction")
                        try:
                            bert_result = bert_model(user_text)[0]
                            bert_label = bert_result['label']
                            bert_score = bert_result['score']
                            
                            st.info(f"BERT Prediction: **{bert_label}** (Score: {bert_score:.3f})")
                        except Exception as e:
                            st.error(f"BERT prediction failed: {str(e)}")
                
                else:
                    st.warning("Please enter some text to analyze.")
        
        else:
            # Batch processing
            st.markdown("#### 📁 Batch Processing")
            
            uploaded_file = st.file_uploader("Upload CSV file with 'text' column", 
                                           type=['csv'])
            
            if uploaded_file:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    
                    if 'text' in batch_df.columns:
                        st.write(f"Loaded {len(batch_df)} texts for analysis")
                        st.dataframe(batch_df.head(), use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            batch_model = st.selectbox("Choose Model for Batch:", 
                                                     list(models.keys()),
                                                     index=list(models.keys()).index('Ensemble') if 'Ensemble' in models else 0)
                        
                        with col2:
                            max_samples = st.number_input("Max samples to process:", 
                                                        min_value=1, 
                                                        max_value=len(batch_df), 
                                                        value=min(100, len(batch_df)))
                        
                        if st.button("🚀 Process Batch"):
                            progress_bar = st.progress(0)
                            results = []
                            
                            selected_model = models[batch_model]
                            sample_df = batch_df.head(max_samples)
                            
                            for idx, row in sample_df.iterrows():
                                text = str(row['text'])
                                processed_text = preprocess_marathi_text(text)
                                text_tfidf = vectorizer.transform([processed_text])
                                
                                prediction = selected_model.predict(text_tfidf)[0]
                                probabilities = selected_model.predict_proba(text_tfidf)[0]
                                confidence = max(probabilities)
                                
                                results.append({
                                    'original_text': text,
                                    'processed_text': processed_text,
                                    'predicted_class': prediction,
                                    'confidence': confidence
                                })
                                
                                progress_bar.progress((idx + 1) / len(sample_df))
                            
                            # Create results DataFrame
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.success("✅ Batch processing completed!")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="classification_results.csv",
                                mime="text/csv"
                            )
                            
                            # Batch analysis visualization
                            batch_class_counts = results_df['predicted_class'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_batch_pie = px.pie(
                                    values=batch_class_counts.values,
                                    names=batch_class_counts.index,
                                    title="Batch Analysis - Class Distribution"
                                )
                                st.plotly_chart(fig_batch_pie, use_container_width=True)
                            
                            with col2:
                                fig_batch_conf = px.histogram(
                                    results_df, 
                                    x='confidence',
                                    title="Confidence Distribution",
                                    nbins=20
                                )
                                st.plotly_chart(fig_batch_conf, use_container_width=True)
                    
                    else:
                        st.error("CSV file must contain a 'text' column")
                
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
    
    elif page == "📈 Analytics":
        st.markdown("### 📈 Analytics Dashboard")
        
        # Check if models are available
        if 'models' not in st.session_state:
            st.warning("⚠️ Please train models first to view analytics.")
            return
        
        models = st.session_state.models
        vectorizer = st.session_state.vectorizer
        scores = st.session_state.get('model_scores', {})
        f1_scores = st.session_state.get('model_f1_scores', {})
        
        # Dataset analytics
        st.markdown("#### 📊 Dataset Analytics")
        
        # Class distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Training set class distribution
            train_class_dist = train_df['label'].value_counts()
            fig_train_dist = px.bar(
                x=train_class_dist.index,
                y=train_class_dist.values,
                title="Training Set - Class Distribution",
                color=train_class_dist.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_train_dist.update_xaxes(tickangle=45)
            st.plotly_chart(fig_train_dist, use_container_width=True)
        
        with col2:
            # Validation set class distribution  
            valid_class_dist = valid_df['label'].value_counts()
            fig_valid_dist = px.bar(
                x=valid_class_dist.index,
                y=valid_class_dist.values,
                title="Validation Set - Class Distribution",
                color=valid_class_dist.index,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_valid_dist.update_xaxes(tickangle=45)
            st.plotly_chart(fig_valid_dist, use_container_width=True)
        
        # Model performance comparison
        st.markdown("#### 🏆 Model Performance Analysis")
        
        if scores and f1_scores:
            # Create comprehensive performance comparison
            performance_metrics = []
            for model_name in scores.keys():
                performance_metrics.append({
                    'Model': model_name,
                    'Accuracy': scores[model_name],
                    'F1-Score': f1_scores[model_name],
                    'Combined Score': (scores[model_name] + f1_scores[model_name]) / 2
                })
            
            perf_df = pd.DataFrame(performance_metrics)
            
            # Performance radar chart
            fig_radar = go.Figure()
            
            for _, row in perf_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['F1-Score'], row['Combined Score']],
                    theta=['Accuracy', 'F1-Score', 'Combined Score'],
                    fill='toself',
                    name=row['Model']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=True,
                title="Model Performance Comparison (Radar Chart)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Feature importance analysis
        st.markdown("#### 🔍 Feature Importance Analysis")
        
        if 'Logistic Regression' in models:
            lr_model = models['Logistic Regression']
            feature_names = vectorizer.get_feature_names_out()
            
            if hasattr(lr_model, 'coef_') and len(lr_model.coef_.shape) > 1:
                # Multi-class classification
                classes = lr_model.classes_
                
                # Get top features for each class
                n_features = min(10, len(feature_names))
                
                for i, class_name in enumerate(classes[:3]):  # Show top 3 classes
                    if i < lr_model.coef_.shape[0]:
                        class_coef = lr_model.coef_[i]
                        top_indices = np.argsort(np.abs(class_coef))[-n_features:]
                        
                        top_features = [(feature_names[idx], class_coef[idx]) for idx in top_indices]
                        top_features.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        # Create feature importance plot
                        feature_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
                        
                        fig_feat = px.bar(
                            feature_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title=f"Top Features for '{class_name}' Class",
                            color='Importance',
                            color_continuous_scale='RdBu'
                        )
                        fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_feat, use_container_width=True)
        
        # Word frequency analysis
        st.markdown("#### 📝 Word Frequency Analysis")
        
        # Combine all text for word frequency
        all_text = ' '.join(train_df['processed_text'])
        words = all_text.split()
        word_freq = pd.Series(words).value_counts().head(30)
        
        fig_wordfreq = px.bar(
            x=word_freq.values,
            y=word_freq.index,
            orientation='h',
            title="Top 30 Most Frequent Words in Training Data",
            labels={'x': 'Frequency', 'y': 'Words'}
        )
        fig_wordfreq.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_wordfreq, use_container_width=True)
        
        # Export analytics
        st.markdown("#### 📁 Export Analytics")
        
        if st.button("Generate Complete Analytics Report"):
            analytics_report = {
                'dataset_info': {
                    'total_samples': len(train_df) + len(valid_df) + len(test_df),
                    'train_samples': len(train_df),
                    'valid_samples': len(valid_df),
                    'test_samples': len(test_df),
                    'unique_classes': sorted(train_df['label'].unique().tolist()),
                    'class_distribution': train_df['label'].value_counts().to_dict()
                },
                'model_performance': {
                    'accuracy_scores': scores,
                    'f1_scores': f1_scores
                },
                'word_frequency': word_freq.head(50).to_dict()
            }
            
            # Convert to JSON
            import json
            analytics_json = json.dumps(analytics_report, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download Analytics Report (JSON)",
                data=analytics_json,
                file_name="marathi_emotion_analytics.json",
                mime="application/json"
            )
            
            st.success("✅ Analytics report generated successfully!")

# Additional utility functions for the updated version
def create_requirements_file():
    """Generate requirements.txt content for the updated project"""
    requirements = """
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
transformers==4.33.0
torch==2.0.1
datasets==2.14.0
    """.strip()
    return requirements

def create_setup_instructions():
    """Generate setup instructions for L3Cube-MahaEmotions integration"""
    instructions = """
# Marathi Emotion & Sentiment Analysis - Setup Instructions

## Dataset Setup

1. **Download L3Cube-MahaEmotions Dataset**
   - Visit: https://github.com/l3cube-pune/MarathiNLP
   - Download the emotion and hate speech datasets

2. **Organize Dataset Structure**
   Create the following folder structure:
   ```
   Dataset/
   ├── emotion/
   │   ├── emotion_train.csv
   │   ├── emotion_valid.csv
   │   └── emotion_test.csv
   └── hate/
       ├── hate_bin_train.csv
       ├── civil_hate_augmented.csv
       ├── hate_bin_valid.csv
       └── hate_bin_test.csv
   ```

3. **CSV File Requirements**
   Each CSV file should have columns:
   - `text`: Marathi text content
   - `label`: Class labels (emotion/hate categories)

## Installation Steps

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Features

- **Multi-task Classification**: Emotion recognition + Hate speech detection
- **Combined Dataset Training**: Merges emotion and hate datasets
- **Model Comparison**: Multiple ML algorithms with ensemble learning
- **Interactive Analytics**: Comprehensive performance analysis
- **Batch Processing**: Handle multiple texts at once

## Model Performance

The application trains and compares:
- Logistic Regression (with balanced class weights)
- Naive Bayes (with alpha smoothing)
- SVM (linear kernel with balanced weights)
- Random Forest (with balanced weights)
- Ensemble Model (voting classifier)

## Dataset Statistics

Based on L3Cube-MahaEmotions structure:
- **Emotion Classes**: joy, sadness, anger, fear, surprise, love, optimism, pessimism
- **Hate Classes**: hate, not_hate
- **Combined Training**: Emotion + Hate datasets merged
- **Preprocessing**: Devanagari script cleaning and normalization

## Usage Tips

1. **Training**: Use full dataset for best results, sample for quick testing
2. **Prediction**: Single text for interactive analysis, batch for bulk processing
3. **Analytics**: Review model performance and feature importance
4. **Export**: Download results and analytics reports

## Troubleshooting

- **File Not Found**: Ensure dataset files are in correct paths
- **Memory Issues**: Use smaller sample sizes for training
- **Encoding Errors**: Ensure CSV files are UTF-8 encoded
    """.strip()
    return instructions

# Run the main application
if __name__ == "__main__":
    # Display setup information in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📚 Project Info")
        
        if st.button("📋 Setup Instructions"):
            st.text_area("Setup Instructions", 
                        create_setup_instructions(), 
                        height=400)
        
        if st.button("📄 Requirements"):
            st.text_area("Requirements.txt", 
                        create_requirements_file(), 
                        height=200)
        
        st.markdown("---")
        st.markdown("### 🔗 Useful Links")
        st.markdown("""
        - [L3Cube-MahaEmotions](https://github.com/l3cube-pune/MarathiNLP)
        - [Marathi BERT Models](https://huggingface.co/l3cube-pune)
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [Scikit-learn Guide](https://scikit-learn.org)
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Marathi Emotion & Sentiment Analysis**
        
        Built with:
        - 🐍 Python
        - 🚀 Streamlit  
        - 🤖 Scikit-learn
        - 🔤 Transformers
        - 📊 Plotly
        
        Using L3Cube-MahaEmotions dataset for comprehensive Marathi text analysis.
        """)
    
    # Run main application
    main()# Main app