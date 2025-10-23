import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def create_sample_dataset(output_path='data/sample_transcripts.csv'):
    """Generate sample dataset for demo"""
    data = {
        'text': [
            "This vitamin supplement supports your immune system",
            "Our product is clinically tested and safe",
            "This cream will CURE all your skin problems permanently",
            "Guaranteed to treat diabetes without medication",
            "May help reduce inflammation based on studies",
            "This pill can replace your doctor's prescription",
            "Lose 50 pounds in one week guaranteed",
            "Supports healthy weight management when combined with diet",
            "Scientifically formulated to support joint health",
            "Miracle cure for cancer! No side effects!",
            "Our supplement contains natural ingredients",
            "This c*u*r*e works better than medicine",
            "FDA approved treatment for all diseases",
            "Helps support overall wellness",
            "Proven to eliminate all pain instantly",
            "Natural ingredients that may support health",
            "This m!r@cle pill fixes everything",
            "Consult your doctor before use",
            "Guaranteed c u r e with no risks",
            "May help as part of a balanced diet"
        ],
        'label': [0, 0, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created at {output_path}")
    return df

def create_prohibited_words(output_path='data/prohibited_words.txt'):
    """Generate prohibited words list"""
    words = [
        "cure",
        "miracle",
        "guaranteed cure",
        "treat disease",
        "replace medication",
        "no side effects",
        "fda approved",
        "eliminate",
        "fix everything"
    ]
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(words))
    
    print(f"Prohibited words list created at {output_path}")

def plot_results(y_true, y_pred, labels=['Compliant', 'Suspected', 'Serious']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    create_sample_dataset()
    create_prohibited_words()
