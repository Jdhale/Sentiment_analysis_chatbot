# # import streamlit as st

import pandas as pd
import numpy as np
# # import pickle
import re
import os
# # from sklearn.model_selection import train_test_split
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.naive_bayes import MultinomialNB
# # from sklearn.svm import SVC
# # from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # from wordcloud import WordCloud
# # import plotly.express as px
# # import plotly.graph_objects as go
# # from plotly.subplots import make_subplots


# -----------------------------
# Use merged dataset
# -----------------------------


def load_combined_data():
    # Load and clean emotion data
    emotion_train_df = pd.read_csv("Dataset/emotion/train.csv")
    emotion_valid_df = pd.read_csv("Dataset/emotion/validation.csv")
    emotion_test_df = pd.read_csv("Dataset/emotion/test.csv")
    
    # Rename for consistency
    emotion_train_df.rename(columns={"text": "text", "label": "label"}, inplace=True)
    emotion_valid_df.rename(columns={"text": "text", "label": "label"}, inplace=True)
    emotion_test_df.rename(columns={"text": "text", "label": "label"}, inplace=True)

    # Load and clean hate data
    hate_train_df = pd.read_csv("Dataset/hate/hate_bin_train.csv")
    civil_df = pd.read_csv("Dataset/hate/civil_hate_augmented.csv")
    hate_valid_df = pd.read_csv("Dataset/hate/hate_bin_valid.csv")
    hate_test_df = pd.read_csv("Dataset/hate/hate_bin_test.csv")

    # Rename for consistency
    hate_train_df.rename(columns={"Tweet": "text", "Label": "label"}, inplace=True)
    civil_df.rename(columns={"Tweet": "text", "Label": "label"}, inplace=True)
    hate_valid_df.rename(columns={"Tweet": "text", "Label": "label"}, inplace=True)
    hate_test_df.rename(columns={"Tweet": "text", "Label": "label"}, inplace=True)

    # Merge train, valid, test datasets
    hate_train_df = pd.concat([hate_train_df, civil_df], ignore_index=True)
    
    train_df = pd.concat([emotion_train_df, hate_train_df], ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
    valid_df = pd.concat([emotion_valid_df, hate_valid_df], ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
    test_df = pd.concat([emotion_test_df, hate_test_df], ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)

    return train_df, valid_df, test_df

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

print("--------------------------------")
print("emotion train exists:", os.path.exists("Dataset/emotion/train.csv"))
print("hate train exists:", os.path.exists("Dataset/hate/hate_bin_train.csv"))

print("--------------------------------")
train_df, valid_df, test_df = load_combined_data()
print(len(train_df))
print(len(valid_df))
print(len(test_df))
print(train_df.head())
print(valid_df.head())
print(test_df.head())

