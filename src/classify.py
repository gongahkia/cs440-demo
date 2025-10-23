from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

class TwoStageClassifier:
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        """Initialize two-stage BERT classifier"""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Stage 1: Binary classification (compliant vs violation)
        self.stage1_model = None
        
        # Stage 2: Severity classification (suspected vs serious)
        self.stage2_model = None
    
    def tokenize(self, texts):
        """Tokenize input texts"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def prepare_stage1_data(self, texts, labels):
        """Prepare data for stage 1: binary classification"""
        # Convert labels: 0 -> 0 (compliant), 1,2 -> 1 (violation)
        binary_labels = [0 if label == 0 else 1 for label in labels]
        return texts, binary_labels
    
    def prepare_stage2_data(self, texts, labels):
        """Prepare data for stage 2: severity classification"""
        # Filter only violation samples (labels 1 and 2)
        violation_texts = [text for text, label in zip(texts, labels) if label > 0]
        violation_labels = [label - 1 for label in labels if label > 0]  # Remap to 0,1
        return violation_texts, violation_labels
    
    def train_stage1(self, train_texts, train_labels, val_texts, val_labels, output_dir='./models/stage1'):
        """Train Stage 1: Binary classifier"""
        print("\n=== Training Stage 1: Binary Classification ===")
        
        train_texts, train_labels = self.prepare_stage1_data(train_texts, train_labels)
        val_texts, val_labels = self.prepare_stage1_data(val_texts, val_labels)
        
        # Initialize model
        self.stage1_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2
        )
        
        # Create dataset
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=self.max_length)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=self.max_length)
        
        train_dataset = BertDataset(train_encodings, train_labels)
        val_dataset = BertDataset(val_encodings, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.stage1_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        print("Stage 1 training complete!")
        
        return trainer
    
    def train_stage2(self, train_texts, train_labels, val_texts, val_labels, output_dir='./models/stage2'):
        """Train Stage 2: Severity classifier"""
        print("\n=== Training Stage 2: Severity Classification ===")
        
        train_texts, train_labels = self.prepare_stage2_data(train_texts, train_labels)
        val_texts, val_labels = self.prepare_stage2_data(val_texts, val_labels)
        
        if len(train_texts) == 0:
            print("No violation samples for stage 2 training!")
            return None
        
        # Initialize model
        self.stage2_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2
        )
        
        # Create dataset
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=self.max_length)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=self.max_length)
        
        train_dataset = BertDataset(train_encodings, train_labels)
        val_dataset = BertDataset(val_encodings, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.stage2_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        print("Stage 2 training complete!")
        
        return trainer
    
    def predict(self, text):
        """Two-stage prediction pipeline"""
        if self.stage1_model is None:
            raise ValueError("Models not trained! Run train_stage1 and train_stage2 first.")
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)
        
        # Stage 1: Check if violation
        self.stage1_model.eval()
        with torch.no_grad():
            outputs = self.stage1_model(**inputs)
            stage1_pred = torch.argmax(outputs.logits, dim=1).item()
        
        if stage1_pred == 0:
            return 0, "Compliant"
        
        # Stage 2: Determine severity
        if self.stage2_model is None:
            return 1, "Suspected Violation (Stage 2 model not available)"
        
        self.stage2_model.eval()
        with torch.no_grad():
            outputs = self.stage2_model(**inputs)
            stage2_pred = torch.argmax(outputs.logits, dim=1).item()
        
        severity = stage2_pred + 1  # Remap: 0 -> 1 (suspected), 1 -> 2 (serious)
        label = "Suspected Violation" if severity == 1 else "Serious Violation"
        
        return severity, label
    
    def load_models(self, stage1_path, stage2_path):
        """Load pre-trained models"""
        self.stage1_model = BertForSequenceClassification.from_pretrained(stage1_path)
        try:
            self.stage2_model = BertForSequenceClassification.from_pretrained(stage2_path)
        except:
            print("Stage 2 model not found, will only do binary classification")
            self.stage2_model = None

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }
