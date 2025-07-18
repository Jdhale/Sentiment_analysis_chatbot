import pandas as pd
tweets_df = pd.read_csv("Dataset/tweets/tweets-valid.csv")
hate_df = pd.read_csv("Dataset/hate/hate_bin_valid.csv")
tweets_train= pd.read_csv("Dataset/tweets/tweets-extra.csv")
hate_train = pd.read_csv("Dataset/hate/hate_bin_train.csv")
emotion_df=pd.read_csv("Dataset/emotion/validation.csv")
emotion_train=pd.read_csv("Dataset/emotion/train.csv")




import re
import emoji
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

normalizer = IndicNormalizerFactory().get_normalizer("mr")
def clean_text(text):
    text = normalizer.normalize(text)
    text = re.sub(r"http\S+|@\w+|#[A-Za-z0-9_]+", "", text)
    return text.strip().lower()

emotion_train['Tweet'] =emotion_train['Tweet'].astype(str).apply(clean_text)
emotion_df['Tweet'] =emotion_df['Tweet'].astype(str).apply(clean_text)
tweets_df['tweet'] = tweets_df['tweet'].astype(str).apply(clean_text)
tweets_train['tweet'] = tweets_train['tweet'].astype(str).apply(clean_text)
hate_train['text'] = hate_train['text'].astype(str).apply(clean_text)
hate_df['text'] = hate_df['text'].astype(str).apply(clean_text)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# === Sentiment (Emotion) Classification Pipeline ===
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# === Hate Speech Classification Pipeline ===
hate_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# === Training the models ===
sentiment_pipeline.fit(emotion_train['Tweet'], emotion_train['Label'])
hate_pipeline.fit(hate_train['text'], hate_train['label'])

# === Making Predictions ===
sentiment_preds = sentiment_pipeline.predict(emotion_df['Tweet'])
hate_preds = hate_pipeline.predict(hate_df['text'])

# === Classification Reports ===
print("==== Sentiment Classification Report ====")
print(classification_report(emotion_df['Label'], sentiment_preds))

print("\n==== Hate Speech Classification Report ====")
print(classification_report(hate_df['label'], hate_preds))


def chatbot():
    print("\nMarathi Sentiment & Hate Detection Chatbot")
    print("Type 'exit' to stop.\n")
    while True:
        user_input = input("📥 तुमचं वाक्य: ")
        if user_input.lower() in ['exit','quit']:
            break
        
        cleaned = clean_text(user_input)
        sentiment_pred = sentiment_pipeline.predict([cleaned])[0]
        hate_pred = hate_pipeline.predict([cleaned])[0]

        print(f"➡️ भावना (Sentiment): {sentiment_pred}")
        print(f"➡️ द्वेष / गैरवर्तन (Hate): {hate_pred}\n")


chatbot()
