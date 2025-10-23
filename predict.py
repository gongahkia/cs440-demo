import argparse
from src.normalize import TextNormalizer
from src.classify import TwoStageClassifier
import sys

def main():
    parser = argparse.ArgumentParser(description='CLiveSVD Demo - Content Moderation Prediction')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--file', type=str, help='File with texts to classify (one per line)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize components
    print("Loading models...")
    normalizer = TextNormalizer('data/prohibited_words.txt')
    classifier = TwoStageClassifier()
    
    try:
        classifier.load_models('./models/stage1', './models/stage2')
        print("Models loaded successfully!\n")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run 'python train.py' first")
        sys.exit(1)
    
    # Process based on mode
    if args.interactive:
        interactive_mode(normalizer, classifier)
    elif args.text:
        process_text(args.text, normalizer, classifier)
    elif args.file:
        process_file(args.file, normalizer, classifier)
    else:
        print("Please provide --text, --file, or --interactive flag")
        parser.print_help()

def process_text(text, normalizer, classifier):
    """Process single text"""
    print(f"Original text: {text}")
    
    # Normalize
    normalized = normalizer.normalize(text)
    morphs = normalizer.detect_morphs(text)
    
    print(f"Normalized: {normalized}")
    if morphs:
        print(f"⚠️  Detected morphed words: {', '.join(morphs)}")
    
    # Classify
    label, description = classifier.predict(normalized)
    
    print(f"\n{'=' * 50}")
    print(f"RESULT: {description} (Label: {label})")
    print(f"{'=' * 50}\n")
    
    if label > 0:
        print("⚠️  FLAGGED FOR REVIEW")
        if label == 2:
            print("🚨 SERIOUS VIOLATION - IMMEDIATE ACTION REQUIRED")

def process_file(filepath, normalizer, classifier):
    """Process file with multiple texts"""
    with open(filepath, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(texts)} texts from {filepath}\n")
    
    results = []
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] Processing: {text[:50]}...")
        normalized = normalizer.normalize(text)
        label, description = classifier.predict(normalized)
        results.append((text, label, description))
        print(f"Result: {description}")
    
    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    compliant = sum(1 for _, label, _ in results if label == 0)
    suspected = sum(1 for _, label, _ in results if label == 1)
    serious = sum(1 for _, label, _ in results if label == 2)
    
    print(f"✓ Compliant: {compliant}")
    print(f"⚠️  Suspected violations: {suspected}")
    print(f"🚨 Serious violations: {serious}")
    
    if serious > 0:
        print(f"\n⚠️  {serious} serious violations detected - review required!")

def interactive_mode(normalizer, classifier):
    """Interactive prediction mode"""
    print("CLiveSVD Demo - Interactive Mode")
    print("Enter text to classify (or 'quit' to exit)")
    print("=" * 50)
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not text:
            continue
        
        process_text(text, normalizer, classifier)

if __name__ == "__main__":
    main()
