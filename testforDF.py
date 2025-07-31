

import pandas as pd
import numpy as np

import re
import os






def load_combined_data():
    # Load and clean emotion data
    emotion_train_df = pd.read_csv("Dataset/emotion/train.csv")
    emotion_valid_df = pd.read_csv("Dataset/emotion/validation.csv")
    emotion_test_df = pd.read_csv("Dataset/emotion/test.csv")

    # Standardize column names
    emotion_train_df = emotion_train_df.rename(columns={"text": "text", "label": "label"})
    emotion_valid_df = emotion_valid_df.rename(columns={"text": "text", "label": "label"})
    emotion_test_df = emotion_test_df.rename(columns={"text": "text", "label": "label"})

    # Drop NaNs and duplicates
    emotion_train_df = emotion_train_df.dropna().drop_duplicates().reset_index(drop=True)

    # Load and clean hate data
    hate_train_df = pd.read_csv("Dataset/hate/hate_bin_train.csv")
    civil_df = pd.read_csv("Dataset/hate/civil_hate_augmented.csv")
    hate_valid_df = pd.read_csv("Dataset/hate/hate_bin_valid.csv")
    hate_test_df = pd.read_csv("Dataset/hate/hate_bin_test.csv")

    # Standardize column names to match emotion data
    for df in [hate_train_df, hate_valid_df, hate_test_df, civil_df]:
        if 'sentence' in df.columns:
            df.rename(columns={'sentence': 'text'}, inplace=True)
        if 'category' in df.columns:
            df.rename(columns={'category': 'label'}, inplace=True)

    # Combine hate + civil for training
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


def prepare_combined_data():
    train_df, valid_df, test_df = load_combined_data()
    train_df['processed_text'] = train_df['text'].apply(preprocess_marathi_text)
    valid_df['processed_text'] = valid_df['text'].apply(preprocess_marathi_text)
    test_df['processed_text'] = test_df['text'].apply(preprocess_marathi_text)
    return train_df, valid_df, test_df




train_df, valid_df, test_df = prepare_combined_data()

print(len(train_df))
print(len(valid_df))
print(len(test_df))
print(train_df.head())
print(valid_df.head())
print(test_df.head())



print("LOAD--------------------------------")
train_df, valid_df, test_df = load_combined_data()
print(len(train_df))
print(len(valid_df))
print(len(test_df))
print(train_df.head())
print(valid_df.head())
print(test_df.head())

# # -----------------------------
# # Use merged dataset
# # -----------------------------
# @st.cache_data

# def load_combined_data():
#     # Load and clean emotion data
#     emotion_train_df = pd.read_csv("Dataset/emotion/train.csv")
#     emotion_valid_df = pd.read_csv("Dataset/emotion/validation.csv")
#     emotion_test_df = pd.read_csv("Dataset/emotion/test.csv")
#     emotion_train_df = emotion_train_df.dropna().drop_duplicates().reset_index(drop=True)

#     # Load and clean hate data
#     hate_train_df = pd.read_csv("Dataset/hate/hate_bin_train.csv")
#     civil_df = pd.read_csv("Dataset/hate/civil_hate_augmented.csv")
#     hate_valid_df = pd.read_csv("Dataset/hate/hate_bin_valid.csv")
#     hate_test_df = pd.read_csv("Dataset/hate/hate_bin_test.csv")
#     hate_train_df = pd.concat([hate_train_df, civil_df], ignore_index=True)
#     hate_train_df = hate_train_df.dropna().drop_duplicates().reset_index(drop=True)

#     # Merge train, valid, test datasets
#     train_df = pd.concat([emotion_train_df, hate_train_df], ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
#     valid_df = pd.concat([emotion_valid_df, hate_valid_df], ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
#     test_df = pd.concat([emotion_test_df, hate_test_df], ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)

#     return train_df, valid_df, test_df


# # Text preprocessing for Marathi (Barabar aahe as previous code)
# def preprocess_marathi_text(text):
#     """Preprocess Marathi text for sentiment analysis"""
#     # Remove URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
#     # Remove special characters but keep Devanagari script
#     text = re.sub(r'[^\u0900-\u097F\s]', ' ', text)
    
#     # Remove extra whitespace
#     text = ' '.join(text.split())
    
#     # Convert to lowercase (for Devanagari, this doesn't change much but good practice)
#     text = text.strip()
    
#     return text


# # -----------------------------
# # Prepare data(Navin try)
# # -----------------------------
# @st.cache_data

# def prepare_combined_data():
#     train_df, valid_df, test_df = load_combined_data()
#     train_df['processed_text'] = train_df['text'].apply(preprocess_marathi_text)
#     valid_df['processed_text'] = valid_df['text'].apply(preprocess_marathi_text)
#     test_df['processed_text'] = test_df['text'].apply(preprocess_marathi_text)
#     return train_df, valid_df, test_df



# # -----------------------------
# # Train models
# # -----------------------------
# @st.cache_resource

# def train_models(train_df, valid_df=None, test_df=None):
#     """
#     Train models using the provided datasets.
#     If valid_df and test_df are provided, use them for validation and testing.
#     Otherwise, split the train_df for validation and testing.
#     """
#     # Use train_df for training
#     X_train = train_df['processed_text']
#     y_train = train_df['label']
    
#     # If validation and test sets are provided, use them
#     if valid_df is not None and test_df is not None:
#         X_valid = valid_df['processed_text']
#         y_valid = valid_df['label']
#         X_test = test_df['processed_text']
#         y_test = test_df['label']
#     else:
#         # Split train_df for validation and testing
#         X_temp, X_test, y_temp, y_test = train_test_split(
#             X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
#         )
#         X_train, X_valid, y_train, y_valid = train_test_split(
#             X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
#         )

#     vectorizer = TfidfVectorizer(
#         max_features=5000,
#         ngram_range=(1, 2),
#         stop_words=None
#     )

#     X_train_tfidf = vectorizer.fit_transform(X_train)
#     X_valid_tfidf = vectorizer.transform(X_valid)
#     X_test_tfidf = vectorizer.transform(X_test)

#     models = {
#         'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
#         'Naive Bayes': MultinomialNB(),
#         'SVM': SVC(random_state=42, probability=True),
#         'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
#     }

#     trained_models = {}
#     model_scores = {}
#     model_f1_scores = {}

#     for name, model in models.items():
#         model.fit(X_train_tfidf, y_train)
#         y_pred = model.predict(X_test_tfidf)
#         accuracy = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         trained_models[name] = model
#         model_scores[name] = accuracy
#         model_f1_scores[name] = f1

#     ensemble = VotingClassifier([
#         ('lr', models['Logistic Regression']),
#         ('nb', models['Naive Bayes']),
#         ('svm', models['SVM']),
#         ('rf', models['Random Forest'])
#     ], voting='soft')

#     ensemble.fit(X_train_tfidf, y_train)
#     ensemble_pred = ensemble.predict(X_test_tfidf)
#     ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
#     ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')

#     trained_models['Ensemble'] = ensemble
#     model_scores['Ensemble'] = ensemble_accuracy
#     model_f1_scores['Ensemble'] = ensemble_f1

#     return trained_models, vectorizer, model_scores, model_f1_scores, X_test, y_test


