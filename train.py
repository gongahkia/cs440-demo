import pandas as pd
from sklearn.model_selection import train_test_split
from src.normalize import TextNormalizer
from src.classify import TwoStageClassifier
import os

def main():
    print("CLiveSVD Demo - Training Pipeline")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('data/sample_transcripts.csv')
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Initialize normalizer
    print("\n2. Initializing text normalizer...")
    normalizer = TextNormalizer('data/prohibited_words.txt')
    
    # Normalize texts
    print("\n3. Normalizing texts...")
    df['normalized_text'] = df['text'].apply(normalizer.normalize)
    df['detected_morphs'] = df['text'].apply(normalizer.detect_morphs)
    
    print("\nSample normalization:")
    for i in range(min(3, len(df))):
        print(f"  Original: {df.iloc[i]['text']}")
        print(f"  Normalized: {df.iloc[i]['normalized_text']}")
        print(f"  Morphs: {df.iloc[i]['detected_morphs']}\n")
    
    # Split data
    print("\n4. Splitting data...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['normalized_text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Initialize classifier
    print("\n5. Initializing two-stage classifier...")
    classifier = TwoStageClassifier()
    
    # Train Stage 1
    print("\n6. Training Stage 1 (Binary Classification)...")
    os.makedirs('./models/stage1', exist_ok=True)
    classifier.train_stage1(train_texts, train_labels, val_texts, val_labels)
    
    # Train Stage 2
    print("\n7. Training Stage 2 (Severity Classification)...")
    os.makedirs('./models/stage2', exist_ok=True)
    classifier.train_stage2(train_texts, train_labels, val_texts, val_labels)
    
    print("\n" + "=" * 50)
    print("Training complete! Models saved to ./models/")
    print("Run 'python predict.py' to test the classifier")

if __name__ == "__main__":
    main()
