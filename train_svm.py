import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_confusion_matrix 

OUTPUT_DIR = "results_svm_classifier" 
TEST_SIZE = 0.4
RANDOM_STATE = 42

LABEL_MAP = {
    'Irrelevant': 0,
    'Attack as a service': 1,
    'Attacker as a service': 2,
    'Malware as a service': 3
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()} 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_html(file_path):
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.extract()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def main(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    if 'file' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'file' and 'label' columns.")

    df['label_id'] = df['label'].map(LABEL_MAP)
    if df['label_id'].isnull().any():
        print("Warning: Some labels in the CSV were not found in LABEL_MAP and will be ignored.")
        df.dropna(subset=['label_id'], inplace=True)
    
    df = df[df['file'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]

    print("Cleaning HTML files and extracting text features...")

    df['cleaned_text'] = [clean_html(path) for path in tqdm(df['file'], desc="Cleaning HTML")]

    all_texts = df['cleaned_text'].tolist()
    all_labels = df['label_id'].astype(int).tolist()

    if not all_texts:
        raise ValueError(f"No valid text entries found in {csv_file} after processing.")  

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=all_labels
    )
    
    print(f"Total samples: {len(all_texts)}")
    print(f"Train samples: {len(X_train_text)}")
    print(f"Test samples: {len(X_test_text)}")

    print("Vectorizing text data with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) 
    X_train_features = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_features = tfidf_vectorizer.transform(X_test_text)
    
    print(f"TF-IDF features shape (Train): {X_train_features.shape}")
    print(f"TF-IDF features shape (Test): {X_test_features.shape}")

    print("Initializing and training SVM classifier...")
    svm_model = SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE, probability=True) 
    svm_model.fit(X_train_features, y_train)
    print("SVM training complete.")

    print("Evaluating SVM model...")
    y_pred = svm_model.predict(X_test_features)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print("\n--- SVM Model Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    model_path = os.path.join(OUTPUT_DIR, "svm_classifier.joblib")
    vectorizer_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(svm_model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"\nSVM model saved to: {model_path}")
    print(f"TF-IDF Vectorizer saved to: {vectorizer_path}")

    print(f"Saving confusion matrix to {OUTPUT_DIR}...")
    save_confusion_matrix(y_test, y_pred, LABEL_MAP, output_dir=OUTPUT_DIR)
    print("Confusion matrix saved successfully.")


if __name__ == "__main__":
    main('final.csv')