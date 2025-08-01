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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# emotion_train_df = pd.read_csv("Dataset/emotion/emotion_train.csv")
# emotion_valid_df = pd.read_csv("Dataset/emotion/emotion_valid.csv")
# emotion_test_df = pd.read_csv("Dataset/emotion/emotion_test.csv")
# emotion_train_df = pd.concat([emotion_train_df, emotion_valid_df], ignore_index=True)
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

# test_df = test_df.reset_index(drop=True)


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

# Create sample Marathi dataset (since we can't directly download L3CubeMahaSent here)
@st.cache_data
def create_sample_dataset():
    """Create a sample Marathi sentiment dataset for demonstration"""
    data = {
        'text': [
            # Positive tweets
            'आज खूप छान दिवस आहे! सगळं काही चांगलं होत आहे.',
            'माझ्या मित्राला नवीन नोकरी मिळाली! खूप आनंद झाला!',
            'हा चित्रपट खूपच उत्तम आहे. सगळ्यांनी पाहावा.',
            'आज सकाळी खूप सुंदर वाटत होतं. मन प्रसन्न आहे.',
            'यश मिळालं! खूप खुशी झाली आज.',
            'माझं कुटुंब खूप आनंदी आहे आज.',
            'हे जेवण खूप स्वादिष्ट आहे. मला खूप आवडलं.',
            'आज खूप चांगला अनुभव मिळाला.',
            'माझा मित्र खूप चांगला आहे. त्याची मदत केली.',
            'आज खूप प्रेम मिळालं सगळ्यांकडून.',
            
            # Negative tweets
            'आज खूप वाईट दिवस गेला. सगळं काही चुकत गेलं.',
            'हे काय चालू आहे? खूप वाईट वाटत आहे.',
            'पाऊस पडत आहे आणि रस्ता बंद आहे. खूप त्रास होत आहे.',
            'माझं मन खराब आहे आज. काहीही चांगलं वाटत नाही.',
            'हे काम करणं खूप कठीण आहे. मला आवडत नाही.',
            'आज खूप दुःख झालं. काळजी वाटत आहे.',
            'या गोष्टीमुळे मला खूप राग आला आहे.',
            'हे परिस्थिती खूप वाईट आहे. समजत नाही काय करावं.',
            'आज खूप निराशा झाली. काहीही योग्य झालं नाही.',
            'मला या गोष्टीचा खूप त्रास होत आहे.',
            
            # Neutral tweets
            'आज ऑफिसमध्ये काम केलं. सकाळी चहा पिऊन बाहेर गेलो.',
            'उद्या शाळेत जाणार आहे. पुस्तक वाचत आहे.',
            'आज बाजारात गेलो होतो. काही सामान विकत घेतलं.',
            'रात्री १० वाजता जेवण केलं. आता झोपायला जातोय.',
            'आज संध्याकाळी मित्रांसोबत भेटलो. चर्चा केली.',
            'ट्रेनने प्रवास केला आज. वेळेत पोहोचलो.',
            'नवीन पुस्तक वाचत आहे. माहिती मिळत आहे.',
            'आज घरी राहिलो. टीव्ही पाहिला.',
            'सकाळी उठून योग केला. नहाऊन तयार झालो.',
            'आज कॉलेजमध्ये व्याख्यान होतं. नोट्स घेतल्या.'
        ],
        'sentiment': [
            # 10 positive
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            # 10 negative  
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            # 10 neutral
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
        ]
    }
    return pd.DataFrame(data)

# Text preprocessing for Marathi
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

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset"""
    df = create_sample_dataset()
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_marathi_text)
    
    return df

# Train models
@st.cache_resource
def train_models(df):
    """Train multiple ML models for sentiment analysis"""
    
    # Prepare features and labels
    X = df['processed_text']
    y = df['sentiment']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=None  # No built-in Marathi stop words in sklearn
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_tfidf, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        trained_models[name] = model
        model_scores[name] = accuracy
    
    # Create ensemble model
    ensemble = VotingClassifier([
        ('lr', models['Logistic Regression']),
        ('nb', models['Naive Bayes']),
        ('svm', models['SVM']),
        ('rf', models['Random Forest'])
    ], voting='soft')
    
    ensemble.fit(X_train_tfidf, y_train)
    ensemble_pred = ensemble.predict(X_test_tfidf)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    trained_models['Ensemble'] = ensemble
    model_scores['Ensemble'] = ensemble_accuracy
    
    return trained_models, vectorizer, model_scores, X_test, y_test

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
                               ["🏠 Home", "📊 Dataset Overview", "🤖 Model Training", "🔮 Sentiment Prediction", "📈 Analytics"])
    
    # Load data
    df = load_and_prepare_data()
    
    if page == "🏠 Home":
        st.markdown("### Welcome to Marathi Sentiment Analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Project Features")
            st.markdown("""
            - **NLP-based Analysis**: Uses advanced machine learning models
            - **Multiple Models**: Logistic Regression, SVM, Random Forest, Naive Bayes
            - **Ensemble Learning**: Combines multiple models for better accuracy
            - **BERT Integration**: Optional transformer-based analysis
            - **Interactive Dashboard**: Real-time predictions and analytics
            - **Marathi Text Support**: Proper Devanagari script handling
            """)
        
        with col2:
            st.markdown("#### 📚 Dataset Information")
            st.markdown(f"""
            - **Total Samples**: {len(df)} tweets
            - **Classes**: Positive, Negative, Neutral
            - **Language**: Marathi (मराठी)
            - **Based on**: L3CubeMahaSent dataset structure
            - **Text Processing**: Custom Devanagari preprocessing
            """)
        
        # Quick prediction demo
        st.markdown("### 🚀 Quick Demo")
        demo_text = st.text_area("Enter Marathi text for quick sentiment analysis:", 
                                value="आज खूप छान दिवस आहे!")
        
        if st.button("Analyze Sentiment"):
            if demo_text:
                # Train models for demo
                with st.spinner("Training models..."):
                    models, vectorizer, scores, _, _ = train_models(df)
                
                # Preprocess and predict
                processed_text = preprocess_marathi_text(demo_text)
                text_tfidf = vectorizer.transform([processed_text])
                
                # Get ensemble prediction
                prediction = models['Ensemble'].predict(text_tfidf)[0]
                probabilities = models['Ensemble'].predict_proba(text_tfidf)[0]
                
                # Display result
                sentiment_colors = {
                    'positive': '#d4edda', 
                    'negative': '#f8d7da', 
                    'neutral': '#fff3cd'
                }
                
                sentiment_emojis = {
                    'positive': '😊', 
                    'negative': '😞', 
                    'neutral': '😐'
                }
                
                st.markdown(f"""
                <div style="background-color: {sentiment_colors[prediction]}; 
                           padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <h3>{sentiment_emojis[prediction]} Sentiment: {prediction.title()}</h3>
                    <p><strong>Confidence:</strong> {max(probabilities):.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif page == "📊 Dataset Overview":
        st.markdown("### 📊 Dataset Overview")
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        
        sentiment_counts = df['sentiment'].value_counts()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>😊 Positive</h3>
                <h2>{sentiment_counts.get('positive', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>😞 Negative</h3>
                <h2>{sentiment_counts.get('negative', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>😐 Neutral</h3>
                <h2>{sentiment_counts.get('neutral', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(values=sentiment_counts.values, 
                           names=sentiment_counts.index,
                           title="Sentiment Distribution",
                           color_discrete_map={
                               'positive': '#28a745',
                               'negative': '#dc3545', 
                               'neutral': '#ffc107'
                           })
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(x=sentiment_counts.index, 
                           y=sentiment_counts.values,
                           title="Sentiment Counts",
                           color=sentiment_counts.index,
                           color_discrete_map={
                               'positive': '#28a745',
                               'negative': '#dc3545', 
                               'neutral': '#ffc107'
                           })
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Sample data
        st.markdown("### 📝 Sample Data")
        st.dataframe(df.head(10))
        
        # Text length analysis
        df['text_length'] = df['text'].str.len()
        fig_hist = px.histogram(df, x='text_length', color='sentiment',
                              title="Text Length Distribution by Sentiment",
                              nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    elif page == "🤖 Model Training":
        st.markdown("### 🤖 Model Training & Evaluation")
        
        with st.spinner("Training models... Please wait."):
            models, vectorizer, scores, X_test, y_test = train_models(df)
        
        st.success("✅ Models trained successfully!")
        
        # Model performance
        st.markdown("#### 📈 Model Performance")
        
        # Convert scores to DataFrame for plotting
        scores_df = pd.DataFrame(list(scores.items()), 
                               columns=['Model', 'Accuracy'])
        
        fig_scores = px.bar(scores_df, x='Model', y='Accuracy',
                          title="Model Accuracy Comparison",
                          color='Accuracy',
                          color_continuous_scale='viridis')
        fig_scores.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Display scores table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Model Scores")
            for model, score in scores.items():
                st.metric(model, f"{score:.3f}")
        
        with col2:
            st.markdown("#### 🎯 Best Model")
            best_model = max(scores, key=scores.get)
            st.success(f"**{best_model}** with accuracy: {scores[best_model]:.3f}")
        
        # Confusion Matrix for best model
        if len(X_test) > 0:
            st.markdown("#### 🔍 Confusion Matrix (Best Model)")
            
            best_model_obj = models[best_model]
            X_test_tfidf = vectorizer.transform(X_test)
            y_pred = best_model_obj.predict(X_test_tfidf)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, 
                             text_auto=True,
                             title=f"Confusion Matrix - {best_model}",
                             labels=dict(x="Predicted", y="Actual"),
                             x=['Negative', 'Neutral', 'Positive'],
                             y=['Negative', 'Neutral', 'Positive'])
            st.plotly_chart(fig_cm, use_container_width=True)
    
    elif page == "🔮 Sentiment Prediction":
        st.markdown("### 🔮 Sentiment Prediction")
        
        # Load models
        with st.spinner("Loading models..."):
            models, vectorizer, scores, _, _ = train_models(df)
            bert_model = load_bert_model() if TRANSFORMERS_AVAILABLE else None
        
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
                                          list(models.keys()))
            
            with col2:
                st.markdown("#### Available Models:")
                for model, score in scores.items():
                    st.write(f"**{model}**: {score:.3f}")
            
            if st.button("🔍 Analyze Sentiment", type="primary"):
                if user_text:
                    # Preprocess text
                    processed_text = preprocess_marathi_text(user_text)
                    
                    # Traditional ML prediction
                    text_tfidf = vectorizer.transform([processed_text])
                    selected_model = models[model_choice]
                    
                    prediction = selected_model.predict(text_tfidf)[0]
                    probabilities = selected_model.predict_proba(text_tfidf)[0]
                    confidence = max(probabilities)
                    
                    # Create probability distribution
                    prob_data = pd.DataFrame({
                        'Sentiment': ['Negative', 'Neutral', 'Positive'],
                        'Probability': probabilities
                    })
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Main prediction
                        sentiment_colors = {
                            'positive': '#28a745', 
                            'negative': '#dc3545', 
                            'neutral': '#ffc107'
                        }
                        
                        sentiment_emojis = {
                            'positive': '😊', 
                            'negative': '😞', 
                            'neutral': '😐'
                        }
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {sentiment_colors[prediction]}, #ffffff);
                                   padding: 2rem; border-radius: 15px; text-align: center;
                                   border: 2px solid {sentiment_colors[prediction]};">
                            <h2>{sentiment_emojis[prediction]} {prediction.upper()}</h2>
                            <h3>Confidence: {confidence:.2%}</h3>
                            <p>Model: {model_choice}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Probability chart
                        fig_prob = px.bar(prob_data, x='Sentiment', y='Probability',
                                        title="Sentiment Probabilities",
                                        color='Sentiment',
                                        color_discrete_map={
                                            'Positive': '#28a745',
                                            'Negative': '#dc3545',
                                            'Neutral': '#ffc107'
                                        })
                        st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # BERT prediction (if available)
                    if bert_model:
                        st.markdown("#### 🤖 BERT Model Prediction")
                        try:
                            bert_result = bert_model(user_text)[0]
                            bert_label = bert_result['label']
                            bert_score = bert_result['score']
                            
                            st.info(f"BERT Prediction: **{bert_label}** (Score: {bert_score:.3f})")
                        except Exception as e:
                            st.error(f"BERT prediction failed: {str(e)}")
                    
                    # Text analysis
                    st.markdown("#### 📊 Text Analysis")
                    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
                    
                    with analysis_col1:
                        st.metric("Word Count", len(processed_text.split()))
                    
                    with analysis_col2:
                        st.metric("Character Count", len(processed_text))
                    
                    with analysis_col3:
                        st.metric("Processed Text Length", len(processed_text.split()))
                
                else:
                    st.warning("Please enter some text to analyze.")
        
        else:
            # Batch processing
            st.markdown("#### 📁 Batch Processing")
            
            uploaded_file = st.file_uploader("Upload CSV file with 'text' column", 
                                           type=['csv'])
            
            if uploaded_file:
                batch_df = pd.read_csv(uploaded_file)
                
                if 'text' in batch_df.columns:
                    st.write(f"Loaded {len(batch_df)} texts for analysis")
                    st.dataframe(batch_df.head())
                    
                    if st.button("🚀 Process Batch"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        selected_model = models['Ensemble']  # Use ensemble for batch
                        
                        for idx, text in enumerate(batch_df['text']):
                            processed_text = preprocess_marathi_text(str(text))
                            text_tfidf = vectorizer.transform([processed_text])
                            
                            prediction = selected_model.predict(text_tfidf)[0]
                            probabilities = selected_model.predict_proba(text_tfidf)[0]
                            confidence = max(probabilities)
                            
                            results.append({
                                'original_text': text,
                                'processed_text': processed_text,
                                'predicted_sentiment': prediction,
                                'confidence': confidence
                            })
                            
                            progress_bar.progress((idx + 1) / len(batch_df))
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.success("✅ Batch processing completed!")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        # Batch analysis visualization
                        batch_sentiment_counts = results_df['predicted_sentiment'].value_counts()
                        fig_batch = px.pie(values=batch_sentiment_counts.values,
                                         names=batch_sentiment_counts.index,
                                         title="Batch Analysis - Sentiment Distribution")
                        st.plotly_chart(fig_batch, use_container_width=True)
                
                else:
                    st.error("CSV file must contain a 'text' column")
    
    elif page == "📈 Analytics":
        st.markdown("### 📈 Analytics Dashboard")
        
        # Load models for analytics
        models, vectorizer, scores, _, _ = train_models(df)
        
        # Model comparison
        st.markdown("#### 🏆 Model Performance Comparison")
        
        # Detailed performance metrics
        performance_data = []
        for model_name, model_obj in models.items():
            if model_name != 'Ensemble':  # Skip ensemble for individual analysis
                # Get sample predictions for analysis
                sample_texts = df['processed_text'].sample(min(50, len(df)))
                sample_tfidf = vectorizer.transform(sample_texts)
                predictions = model_obj.predict(sample_tfidf)
                probabilities = model_obj.predict_proba(sample_tfidf)
                
                avg_confidence = np.mean([max(prob) for prob in probabilities])
                
                performance_data.append({
                    'Model': model_name,
                    'Accuracy': scores[model_name],
                    'Avg Confidence': avg_confidence
                })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Performance visualization
        fig_perf = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Accuracy Comparison', 'Average Confidence'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_perf.add_trace(
            go.Bar(x=perf_df['Model'], y=perf_df['Accuracy'], 
                  name='Accuracy', marker_color='#1f77b4'),
            row=1, col=1
        )
        
        fig_perf.add_trace(
            go.Bar(x=perf_df['Model'], y=perf_df['Avg Confidence'], 
                  name='Avg Confidence', marker_color='#ff7f0e'),
            row=1, col=2
        )
        
        fig_perf.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Feature importance analysis
        st.markdown("#### 🔍 Feature Importance Analysis")
        
        # Get feature importance from Logistic Regression
        lr_model = models['Logistic Regression']
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients for each class
        if hasattr(lr_model, 'coef_'):
            coef_df = pd.DataFrame(
                lr_model.coef_.T,
                columns=['negative', 'neutral', 'positive'],
                index=feature_names
            )
            
            # Get top features for each sentiment
            top_positive = coef_df.nlargest(10, 'positive')[['positive']].reset_index()
            top_negative = coef_df.nsmallest(10, 'negative')[['negative']].reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Top Positive Features")
                fig_pos = px.bar(top_positive, x='positive', y='index',
                               orientation='h', title="Most Positive Words",
                               color='positive', color_continuous_scale='Greens')
                fig_pos.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_pos, use_container_width=True)
            
            with col2:
                st.markdown("##### Top Negative Features")
                fig_neg = px.bar(top_negative, x='negative', y='index',
                               orientation='h', title="Most Negative Words",
                               color='negative', color_continuous_scale='Reds')
                fig_neg.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_neg, use_container_width=True)
        
        # Dataset insights
        st.markdown("#### 📊 Dataset Insights")
        
        # Word frequency analysis
        all_text = ' '.join(df['processed_text'])
        words = all_text.split()
        word_freq = pd.Series(words).value_counts().head(20)
        
        fig_wordfreq = px.bar(x=word_freq.values, y=word_freq.index,
                            orientation='h', title="Top 20 Most Frequent Words",
                            labels={'x': 'Frequency', 'y': 'Words'})
        fig_wordfreq.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_wordfreq, use_container_width=True)
        
        # Sentiment-wise word analysis
        st.markdown("#### 🔤 Sentiment-wise Word Analysis")
        
        sentiment_words = {}
        for sentiment in df['sentiment'].unique():
            sentiment_text = ' '.join(df[df['sentiment'] == sentiment]['processed_text'])
            sentiment_words[sentiment] = len(sentiment_text.split())
        
        fig_sentiment_words = px.bar(x=list(sentiment_words.keys()),
                                   y=list(sentiment_words.values()),
                                   title="Total Words by Sentiment",
                                   color=list(sentiment_words.keys()),
                                   color_discrete_map={
                                       'positive': '#28a745',
                                       'negative': '#dc3545',
                                       'neutral': '#ffc107'
                                   })
        st.plotly_chart(fig_sentiment_words, use_container_width=True)
        
        # Model prediction distribution
        st.markdown("#### 🎯 Model Prediction Analysis")
        
        # Analyze predictions on entire dataset
        all_tfidf = vectorizer.transform(df['processed_text'])
        ensemble_predictions = models['Ensemble'].predict(all_tfidf)
        ensemble_probabilities = models['Ensemble'].predict_proba(all_tfidf)
        
        # Create prediction analysis DataFrame
        pred_analysis = pd.DataFrame({
            'actual': df['sentiment'],
            'predicted': ensemble_predictions,
            'confidence': [max(prob) for prob in ensemble_probabilities],
            'correct': df['sentiment'] == ensemble_predictions
        })
        
        # Prediction accuracy by sentiment
        accuracy_by_sentiment = pred_analysis.groupby('actual')['correct'].mean()
        
        fig_acc_sentiment = px.bar(x=accuracy_by_sentiment.index,
                                 y=accuracy_by_sentiment.values,
                                 title="Prediction Accuracy by Sentiment Class",
                                 color=accuracy_by_sentiment.index,
                                 color_discrete_map={
                                     'positive': '#28a745',
                                     'negative': '#dc3545',
                                     'neutral': '#ffc107'
                                 })
        st.plotly_chart(fig_acc_sentiment, use_container_width=True)
        
        # Confidence distribution
        fig_conf = px.histogram(pred_analysis, x='confidence', 
                              color='correct',
                              title="Confidence Distribution for Predictions",
                              nbins=20)
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Export analytics
        st.markdown("#### 📁 Export Analytics Data")
        
        if st.button("Generate Analytics Report"):
            analytics_report = {
                'model_performance': perf_df.to_dict('records'),
                'prediction_analysis': pred_analysis.to_dict('records'),
                'word_frequency': word_freq.head(50).to_dict(),
                'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
                'dataset_size': len(df),
                'accuracy_by_sentiment': accuracy_by_sentiment.to_dict()
            }
            
            # Convert to JSON for download
            import json
            analytics_json = json.dumps(analytics_report, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download Analytics Report (JSON)",
                data=analytics_json,
                file_name="marathi_sentiment_analytics.json",
                mime="application/json"
            )
            
            st.success("✅ Analytics report generated successfully!")

# Additional utility functions
def create_requirements_file():
    """Generate requirements.txt content"""
    requirements = """
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
wordcloud==1.9.2
transformers==4.33.0
torch==2.0.1
datasets==2.14.0
    """.strip()
    return requirements

def create_setup_instructions():
    """Generate setup instructions"""
    instructions = """
# Marathi Sentiment Analysis - Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation Steps

1. **Clone or download the project files**
   ```bash
   mkdir marathi_sentiment_analysis
   cd marathi_sentiment_analysis
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the L3CubeMahaSent dataset (optional)**
   - Visit: https://github.com/l3cube-pune/MarathiNLP
   - Download the sentiment analysis dataset
   - Place it in a 'data' folder

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Dataset Options

### Option 1: Use Sample Dataset (Default)
The application includes a sample Marathi dataset for demonstration.

### Option 2: Use L3CubeMahaSent Dataset
1. Download from: https://github.com/l3cube-pune/MarathiNLP
2. Update the `load_and_prepare_data()` function to load your dataset
3. Ensure your CSV has columns: 'text' and 'sentiment'

### Option 3: Create Your Own Dataset
Create a CSV file with columns:
- `text`: Marathi text/tweets
- `sentiment`: 'positive', 'negative', or 'neutral'

## Model Options

### Traditional ML Models (Default)
- Logistic Regression
- Naive Bayes
- SVM
- Random Forest
- Ensemble Model

### BERT Models (Optional)
Install transformers library and the application will automatically try to load:
- l3cube-pune/marathi-sentiment-tweets
- Other Marathi BERT models from Hugging Face

## Features

1. **Dataset Overview**: Visualize your data distribution
2. **Model Training**: Train and compare multiple models
3. **Sentiment Prediction**: Single text and batch processing
4. **Analytics Dashboard**: Detailed performance analysis

## Customization

### Adding New Models
Add your model to the `train_models()` function:
```python
models['Your Model'] = YourModelClass()
```

### Custom Preprocessing
Modify the `preprocess_marathi_text()` function for domain-specific preprocessing.

### UI Customization
Update the CSS in the `st.markdown()` sections to change the appearance.

## Troubleshooting

### Common Issues
1. **BERT model loading fails**: Ensure transformers library is installed
2. **Memory errors**: Reduce dataset size or model complexity
3. **Encoding issues**: Ensure your data is in UTF-8 format

### Performance Tips
1. Use smaller datasets for faster training during development
2. Cache models using `@st.cache_resource` decorator
3. Preprocess data once and save for reuse

## Contributing
Feel free to contribute by:
- Adding new models
- Improving preprocessing
- Enhancing visualizations
- Adding new features
    """.strip()
    return instructions

# Run the main application
if __name__ == "__main__":
    # Display setup information in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📚 Project Info")
        
        if st.button("📋 Show Setup Instructions"):
            st.text_area("Setup Instructions", 
                        create_setup_instructions(), 
                        height=300)
        
        if st.button("📄 Show Requirements"):
            st.text_area("Requirements.txt", 
                        create_requirements_file(), 
                        height=200)
        
        st.markdown("---")
        st.markdown("### 🔗 Useful Links")
        st.markdown("""
        - [L3CubeMahaSent Dataset](https://github.com/l3cube-pune/MarathiNLP)
        - [Marathi BERT Models](https://huggingface.co/l3cube-pune)
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [Scikit-learn Guide](https://scikit-learn.org)
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Marathi Sentiment Analysis**
        
        Built with:
        - 🐍 Python
        - 🚀 Streamlit
        - 🤖 Scikit-learn
        - 🔤 Transformers
        - 📊 Plotly
        
        For Marathi NLP research and applications.
        """)
    
    # Run main application
    main()