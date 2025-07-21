import pandas as pd
import re
import string
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

# === Load Data ===
train_df = pd.read_csv("Dataset/hate/hate_bin_train.csv")
valid_df = pd.read_csv("Dataset/hate/hate_bin_valid.csv")
test_df = pd.read_csv("Dataset/hate/hate_bin_test.csv")
civil_df = pd.read_csv("Dataset/hate/civil_hate_augmented.csv")
train_df = pd.concat([train_df, civil_df], ignore_index=True)

# === Normalizer for Marathi ===
normalizer = IndicNormalizerFactory().get_normalizer("mr")

# === Text Cleaning Function ===


def clean_text(text):
    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # 2. Remove mentions (@user) and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # 3. Remove emojis and symbols (including emojis in extended unicode)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # Flags
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 4. Remove all characters except Devanagari (Marathi) + spaces + basic punctuations
    # (Preserves full stops, commas, question marks etc.)
    text = re.sub(r'[^\u0900-\u097F\s.,!?]', '', text)

    # 5. Remove extra white spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply preprocessing
for df in [train_df, valid_df, test_df]:
    df['clean_text'] = df['text'].apply(clean_text)

# === Label Encoding ===
le = LabelEncoder()
train_df['label_enc'] = le.fit_transform(train_df['label'])
valid_df['label_enc'] = le.transform(valid_df['label'])
test_df['label_enc'] = le.transform(test_df['label'])

# === Build and Train Pipeline ===
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
    ngram_range=(1, 3),       # include unigrams, bigrams, trigrams
    max_features=10000,       # allow richer vocabulary
    min_df=2,                 # ignore rare terms
    sublinear_tf=True         # dampens term frequency influence
)
),
    ('clf', MultinomialNB())
])
pipeline.fit(train_df['clean_text'], train_df['label_enc'])

# === Evaluate ===
val_preds = pipeline.predict(valid_df['clean_text'])
test_preds = pipeline.predict(test_df['clean_text'])

print("📊 Validation Report:")
print(classification_report(valid_df['label_enc'], val_preds, target_names=le.classes_))

print("\n📊 Test Report:")
print(classification_report(test_df['label_enc'], test_preds, target_names=le.classes_))

marathi_hate_sentences = [
    "मला खोटेपणा मुळीच आवडत नाही, मी त्याचा तिरस्कार करतो.",
    "त्याचं वागणं पाहून चीडच येते.",
    "माझा राग अनावर होतो जेव्हा कोणी खोटं बोलतं.",
    "असे लोक माझ्या दृष्टीनं घृणास्पद आहेत.",
    "मला अशा लोकांचा तिरस्कार वाटतो जे इतरांना त्रास देतात.",
    "तिचं वागणं इतकं अहंकारी आहे की सहनच होत नाही.",
    "हे सगळं पाहून मला घृणा वाटते.",
    "तु खूपच असभ्य आणि उद्धट आहेस.",
    "माझं त्याच्यावर विश्वास नाही कारण तो एक बनवाबनवी करणारा आहे.",
    "हा माणूस समाजासाठी धोकादायक आहे, याला थांबवलं पाहिजे."
]

for sentence in marathi_hate_sentences:
    cleaned_sentence = clean_text(sentence)
    prediction = pipeline.predict([cleaned_sentence])[0]
    label_pred = le.inverse_transform([prediction])[0]
    print(f"Input Sentence: {sentence}")
    print(f"Predicted Label: {label_pred}")
    print("-" * 50)

# # Example: Check a custom Marathi sentence
# custom_sentence = "मला खोटेपणा मुळीच आवडत नाही, मी त्याचा तिरस्कार करतो."  # Replace with your test sentence
# cleaned_sentence = clean_text(custom_sentence)
# prediction = pipeline.predict([cleaned_sentence])[0]

# # Decode the prediction
# label_pred = le.inverse_transform([prediction])[0]
# print(f"Input Sentence: {custom_sentence}")
# print(f"Predicted Label: {label_pred}")
