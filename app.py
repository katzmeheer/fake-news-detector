"""
COMPLETE FAKE NEWS DETECTOR - All-in-One Script
For a solo developer
Handles: Data → Training → Web App → Predictions
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import nltk
from nltk.corpus import stopwords
import string

# Web imports
from flask import Flask, render_template, request, jsonify

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ============================================================================
# PART 1: DATA PREPROCESSING
# ============================================================================

def clean_text(text):
    """Clean a single text string"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)


def load_and_preprocess_data(csv_path):
    """Load CSV and preprocess it"""
    print(f"\n📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} articles")
    
    # Combine title and text
    if 'title' in df.columns and 'text' in df.columns:
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'text' in df.columns:
        df['combined_text'] = df['text']
    else:
        raise ValueError("CSV must have 'text' column or 'title'+'text' columns")
    
    # Clean text
    print("🧹 Cleaning text...")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Create binary labels
    if 'label' in df.columns:
        df['label'] = df['label'].map( { 0: 'FAKE',1: 'REAL' } )
        df['label_binary'] = (df['label'] == 'FAKE').astype(int)
    elif 'category' in df.columns:
        df['label_binary'] = (df['category'].str.upper() == 'FAKE').astype(int)
    else:
        raise ValueError("CSV must have 'label' or 'category' column")
    
    print(f"✓ Cleaned data: {len(df)} articles")
    print(f"  - Fake: {df['label_binary'].sum()}")
    print(f"  - Real: {len(df) - df['label_binary'].sum()}")
    
    return df[['cleaned_text', 'label_binary']]


# ============================================================================
# PART 2: MODEL TRAINING
# ============================================================================

def train_fake_news_model(csv_path, model_name='fake_news_model'):
    """Train the ML model"""
    # Load and preprocess
    df = load_and_preprocess_data(csv_path)
    
    # Split
    print("\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['label_binary'],
        test_size=0.2,
        random_state=42,
        stratify=df['label_binary']
    )
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Vectorize
    print("\n🔢 Converting text to numbers (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"✓ Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train
    print("\n🤖 Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("✓ Model trained!")
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print("=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("=" * 50)
    
    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{model_name}.pkl')
    joblib.dump(vectorizer, f'models/{model_name}_vectorizer.pkl')
    print(f"\n✓ Model saved to models/{model_name}.pkl")
    
    return model, vectorizer, metrics


# ============================================================================
# PART 3: PREDICTION CLASS
# ============================================================================

class FakeNewsPredictor:
    """Make predictions on new articles"""
    
    def __init__(self, model_path='models/fake_news_model.pkl',
                 vectorizer_path='models/fake_news_model_vectorizer.pkl'):
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Model files not found. Train the model first!"
            )
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print("✓ Model loaded!")
    
    def predict(self, article_text):
        """Predict if article is fake or real"""
        cleaned = clean_text(article_text)
        
        if not cleaned:
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'error': 'Text too short'
            }
        
        text_vector = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(text_vector)[0]
        confidence = self.model.predict_proba(text_vector)[0]
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': float(max(confidence)) * 100,
            'fake_probability': float(confidence[1]) * 100,
            'real_probability': float(confidence[0]) * 100
        }


# ============================================================================
# PART 4: FLASK WEB APP
# ============================================================================

app = Flask(__name__)

# Load model when app starts
predictor = None


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form.get("news", "").strip()

        if len(text) < 10:
            result = {"prediction": "Text too short", "confidence": 0}
        else:
            result = predictor.predict(text)

    return render_template("index.html", result=result)
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short'}), 400
        
        result = predictor.predict(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# PART 5: ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTOR")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Train model from CSV file')
    parser.add_argument('--test', type=str, help='Test on a single article')
    parser.add_argument('--web', action='store_true', help='Start web server')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample data')
    
    args = parser.parse_args()
    
    if args.train:
        # Train mode
        model, vectorizer, metrics = train_fake_news_model(args.train)
    
    elif args.test:
        # Test mode
        try:
            pred = FakeNewsPredictor()
            result = pred.predict(args.test)
            print(f"\n📰 Article: {args.test[:100]}...")
            print(f"🎯 Prediction: {result['prediction']}")
            print(f"📊 Confidence: {result['confidence']:.2f}%")
            print(f"   - Fake: {result['fake_probability']:.2f}%")
            print(f"   - Real: {result['real_probability']:.2f}%")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
    
    elif args.web:
        # Web mode
        global predictor
        try:
            predictor = FakeNewsPredictor()
            print("\n🌐 Starting web server...")
            print("Visit: http://localhost:5000")
            app.run(debug=True, port=5000)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Run --train first!")
    
    elif args.demo:
        # Demo mode (with fake data)
        print("\n🎮 Running demo with sample data...")
        
        # Create fake data
        fake_texts = [
            "Aliens land in Times Square - SHOCKING!",
            "Celebrity reveals secret to immortality",
            "This one trick doctors don't want you to know",
        ] * 20
        
        real_texts = [
            "Scientists publish new study on climate change",
            "Government announces new policy",
            "Company releases quarterly earnings report",
        ] * 20
        
        df = pd.DataFrame({
            'text': fake_texts + real_texts,
            'label': ['FAKE'] * 60 + ['REAL'] * 60
        })
        
        # Save temporarily
        df.to_csv('temp_demo.csv', index=False)
        
        # Train
        train_fake_news_model('temp_demo.csv', 'fake_news_model')
        
        # Clean up
        os.remove('temp_demo.csv')
        
        print("\n✅ Demo training complete!")
        print("Run: python app.py --web")
    
    else:
        print("\nUsage:")
        print("  python app.py --train <csv_file>    # Train model")
        print("  python app.py --test 'article text' # Test prediction")
        print("  python app.py --web                 # Start web server")
        print("  python app.py --demo                # Run with sample data")
        print("\nExample:")
        print("  python app.py --demo")
        print("  python app.py --web")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
