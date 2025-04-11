import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import requests
import zipfile
from tqdm import tqdm
import argparse
import logging
import shap
import cv2
from PIL import Image
import io
import warnings
from transformers import BertTokenizer

# Download and cache the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_available = True
except Exception as e:
    print(f"Could not load BERT tokenizer. Some features may be limited. Error: {e}")
    bert_available = False

# Later in your code
if bert_available:
    # Use BERT features
    pass
else:
    # Use fallback approach
    logger.warning("BERT not available, using fallback features only")

try:
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_available = True
except Exception as e:
    print(f"Could not load BERT tokenizer. Some features may be limited. Error: {e}")
    bert_available = False

# Later in your code
if bert_available:
    # Use BERT features
    pass
else:
    # Use fallback approach
    logger.warning("BERT not available, using fallback features only")

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Function to download datasets
def download_datasets(data_dir="data"):
    """
    Download mental health datasets for training
    """
    datasets = {
        "text_data": "https://archive.org/download/mental-health-social-media/mental_health_corpus.zip",
        "audio_features": "https://archive.org/download/mental-health-social-media/audio_features.zip",
        "image_features": "https://archive.org/download/mental-health-social-media/image_features.zip"
    }
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for name, url in datasets.items():
        target_dir = f"{data_dir}/{name}"
        if not os.path.exists(target_dir):
            logger.info(f"Downloading {name} dataset...")
            try:
                response = requests.get(url)
                zip_path = f"{data_dir}/{name}.zip"
                
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                
                # Remove the zip file
                os.remove(zip_path)
                logger.info(f"Successfully downloaded and extracted {name}.")
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
                
                # Create directory for synthetic data
                os.makedirs(target_dir, exist_ok=True)
                
                # Create synthetic data for demonstration purposes
                if name == "text_data":
                    create_synthetic_text_data(target_dir)
                elif name == "audio_features":
                    create_synthetic_audio_features(target_dir)
                elif name == "image_features":
                    create_synthetic_image_features(target_dir)
    
    logger.info("All datasets processed.")
    return True

# Create synthetic text data
def create_synthetic_text_data(target_dir):
    """
    Create synthetic text data for training if real data is unavailable
    """
    logger.info("Creating synthetic text data for demonstration...")
    
    # Crisis texts
    crisis_texts = [
        "I feel so hopeless and worthless, I just want to end it all",
        "I can't stop thinking about suicide, it's the only way out",
        "Nothing matters anymore, I'm a burden to everyone",
        "I just want the pain to stop forever",
        "I've been planning how to kill myself",
        "I don't see any reason to continue living",
        "Everyone would be better off without me",
        "I've written my goodbye notes already",
        "I can't take this pain anymore, tonight is the night",
        "I've collected enough pills to end it all",
        "I feel trapped with no way out except death",
        "I'm going to jump off the bridge tonight",
        "I'm tired of being a burden, it ends today",
        "No one would notice if I disappeared forever",
        "I've made up my mind to end my suffering",
        "Death is the only solution to my problems",
        "I'm going to shoot myself when I get home",
        "I've been cutting myself deeper each time",
        "This world is too painful to exist in anymore",
        "I've researched the most effective suicide methods"
    ] * 5  # Replicate to increase sample size
    
    # Non-crisis texts (expressing distress but not immediate crisis)
    non_crisis_texts = [
        "I'm feeling really down today but I'll get through it",
        "Therapy has been helping me deal with my depression",
        "I'm struggling but I'm talking to friends about it",
        "Having a rough day but things will get better",
        "I need to talk to someone about my feelings",
        "My antidepressants are starting to help a bit",
        "I've been feeling sad but I'm working on self-care",
        "Depression is hard but I'm taking it one day at a time",
        "I'm not doing great but I have support",
        "I've been feeling anxious but my coping skills help",
        "Feeling stressed about work but managing it",
        "I'm going through a tough breakup but surviving",
        "Sometimes I feel worthless but I know that's not true",
        "I've had dark thoughts but I called my therapist",
        "Lost my job and feeling down but applying for new ones",
        "Grieving is difficult but I'm processing it",
        "I'm having a mental health dip but using my tools",
        "Today was hard but tomorrow might be better",
        "I feel empty sometimes but I'm still fighting",
        "My depression makes life hard but I'm not giving up"
    ] * 5  # Replicate to increase sample size
    
    # Create DataFrame
    texts = crisis_texts + non_crisis_texts
    labels = [1] * len(crisis_texts) + [0] * len(non_crisis_texts)  # 1 for crisis, 0 for non-crisis
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(f"{target_dir}/mental_health_posts.csv", index=False)
    logger.info(f"Saved synthetic text dataset with {len(df)} samples")

# Create synthetic audio features
def create_synthetic_audio_features(target_dir):
    """
    Create synthetic audio features for multimodal model
    """
    logger.info("Creating synthetic audio features...")
    
    # Create 200 samples (100 crisis, 100 non-crisis)
    n_samples = 200
    n_features = 13  # MFCC features
    
    # Generate random features
    features = np.random.rand(n_samples, n_features)
    
    # Generate labels
    labels = np.zeros(n_samples)
    labels[:100] = 1  # First half are crisis
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=[f'mfcc_{i}' for i in range(n_features)])
    df['label'] = labels
    
    # Save to CSV
    df.to_csv(f"{target_dir}/audio_features.csv", index=False)
    logger.info(f"Saved synthetic audio features with {n_samples} samples")

# Create synthetic image features
def create_synthetic_image_features(target_dir):
    """
    Create synthetic image features for multimodal model
    """
    logger.info("Creating synthetic image features...")
    
    # Create 200 samples (100 crisis, 100 non-crisis)
    n_samples = 200
    n_features = 512  # ResNet features
    
    # Generate random features
    features = np.random.rand(n_samples, n_features)
    
    # Generate labels
    labels = np.zeros(n_samples)
    labels[:100] = 1  # First half are crisis
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=[f'img_feat_{i}' for i in range(n_features)])
    df['label'] = labels
    
    # Save to CSV
    df.to_csv(f"{target_dir}/image_features.csv", index=False)
    logger.info(f"Saved synthetic image features with {n_samples} samples")

# Load and preprocess text data
def load_text_data(data_dir="data"):
    """
    Load and preprocess text data
    """
    try:
        df = pd.read_csv(f"{data_dir}/text_data/mental_health_posts.csv")
        logger.info(f"Loaded text data with {len(df)} samples")
        return df
    except Exception as e:
        logger.error(f"Failed to load text data: {e}")
        return None

# Train text-based model
def train_text_model(df, model_dir="models"):
    """
    Train a text-based model for crisis detection
    """
    logger.info("Training text-based crisis detection model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Feature extraction
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    
    # Print metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Text model accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{report}")
    
    # Feature importance
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = model.feature_importances_
    
    sorted_idx = feature_importance.argsort()[-20:]
    top_features = [feature_names[i] for i in sorted_idx]
    top_importance = feature_importance[sorted_idx]
    
    logger.info("Top crisis indicator words:")
    for word, importance in zip(top_features, top_importance):
        if importance > 0.01:  # Only show significant words
            logger.info(f"  - {word}: {importance:.4f}")
    
    # Save model and vectorizer
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    text_model = {
        "vectorizer": vectorizer,
        "classifier": model
    }
    
    joblib.dump(text_model, f"{model_dir}/text_model.joblib")
    logger.info(f"Saved text model to {model_dir}/text_model.joblib")
    logger.info(f"Saved text model to {model_dir}/text_model.joblib")
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(top_features, top_importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Top Features for Crisis Detection')
    plt.tight_layout()
    plt.savefig(f"{model_dir}/text_feature_importance.png")
    
    # Generate SHAP values for explainability
    try:
        X_sample = X_test_vec[:100]  # Use a subset for SHAP analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Create summary plot
        shap.summary_plot(
            shap_values[1],  # For the positive class
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(f"{model_dir}/text_shap_summary.png")
        plt.close()
        
        logger.info("Generated SHAP explainability visualizations")
    except Exception as e:
        logger.warning(f"Could not generate SHAP visualizations: {e}")
    
    return model, vectorizer, accuracy

# Define a PyTorch Dataset for the BERT model
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define BERT model for classification
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Train a BERT-based model for crisis detection
def train_bert_model(df, model_dir="models", epochs=3):
    """
    Train a BERT-based model for crisis detection
    """
    logger.info("Training BERT-based crisis detection model...")
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = MentalHealthDataset(X_train, y_train, tokenizer)
    test_dataset = MentalHealthDataset(X_test, y_test, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Initialize model
    model = BERTClassifier(bert_model)
    model.to(device)
    
    # Define optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
    
    # Evaluation
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Print metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    logger.info(f"BERT model accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{report}")
    
    # Save the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer
    }, f"{model_dir}/bert_model.pt")
    
    logger.info(f"Saved BERT model to {model_dir}/bert_model.pt")
    
    return model, tokenizer, accuracy

# Load and preprocess multimodal data
def load_multimodal_data(data_dir="data"):
    """
    Load text, audio, and image features for multimodal model
    """
    try:
        # Load text data
        text_df = pd.read_csv(f"{data_dir}/text_data/mental_health_posts.csv")
        logger.info(f"Loaded text data with {len(text_df)} samples")
        
        # Load audio features
        audio_df = pd.read_csv(f"{data_dir}/audio_features/audio_features.csv")
        logger.info(f"Loaded audio features with {len(audio_df)} samples")
        
        # Load image features
        image_df = pd.read_csv(f"{data_dir}/image_features/image_features.csv")
        logger.info(f"Loaded image features with {len(image_df)} samples")
        
        # Return data
        return text_df, audio_df, image_df
    except Exception as e:
        logger.error(f"Failed to load multimodal data: {e}")
        return None, None, None

# Define multimodal fusion model
class MultimodalFusionModel(nn.Module):
    def __init__(self, text_dim=768, audio_dim=13, image_dim=512):
        super(MultimodalFusionModel, self).__init__()
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        fusion_dim = 256 + 64 + 128
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Linear(128, 2)
    
    def forward(self, text_feats, audio_feats, image_feats):
        # Encode each modality
        text_encoded = self.text_encoder(text_feats)
        audio_encoded = self.audio_encoder(audio_feats)
        image_encoded = self.image_encoder(image_feats)
        
        # Concatenate features
        fused = torch.cat([text_encoded, audio_encoded, image_encoded], dim=1)
        
        # Pass through fusion layer
        fused = self.fusion_layer(fused)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits

# Train multimodal model
def train_multimodal_model(text_df, audio_df, image_df, model_dir="models", epochs=5):
    """
    Train a multimodal fusion model for crisis detection
    """
    logger.info("Training multimodal crisis detection model...")
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare text features (using TF-IDF for simplicity)
    vectorizer = TfidfVectorizer(max_features=768)
    text_features = vectorizer.fit_transform(text_df['text']).toarray()
    
    # Prepare audio features
    audio_features = audio_df.drop('label', axis=1).values
    
    # Prepare image features
    image_features = image_df.drop('label', axis=1).values
    
    # Ensure all data has the same number of samples
    min_samples = min(len(text_features), len(audio_features), len(image_features))
    text_features = text_features[:min_samples]
    audio_features = audio_features[:min_samples]
    image_features = image_features[:min_samples]
    labels = text_df['label'].values[:min_samples]
    
    # Split data
    indices = np.arange(min_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_text_train = torch.FloatTensor(text_features[train_idx])
    X_audio_train = torch.FloatTensor(audio_features[train_idx])
    X_image_train = torch.FloatTensor(image_features[train_idx])
    y_train = torch.LongTensor(labels[train_idx])
    
    X_text_test = torch.FloatTensor(text_features[test_idx])
    X_audio_test = torch.FloatTensor(audio_features[test_idx])
    X_image_test = torch.FloatTensor(image_features[test_idx])
    y_test = torch.LongTensor(labels[test_idx])
    
    # Create dataset and data loader
    train_dataset = torch.utils.data.TensorDataset(
        X_text_train, X_audio_train, X_image_train, y_train
    )
    test_dataset = torch.utils.data.TensorDataset(
        X_text_test, X_audio_test, X_image_test, y_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = MultimodalFusionModel(
        text_dim=text_features.shape[1],
        audio_dim=audio_features.shape[1],
        image_dim=image_features.shape[1]
    )
    model.to(device)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            text_feats, audio_feats, image_feats, labels = [b.to(device) for b in batch]
            
            outputs = model(text_feats, audio_feats, image_feats)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
    
    # Evaluation
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            text_feats, audio_feats, image_feats, labels = [b.to(device) for b in batch]
            
            outputs = model(text_feats, audio_feats, image_feats)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Print metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    logger.info(f"Multimodal model accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{report}")
    
    # Save the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    multimodal_model = {
        "model": model.state_dict(),
        "text_vectorizer": vectorizer,
        "text_dim": text_features.shape[1],
        "audio_dim": audio_features.shape[1],
        "image_dim": image_features.shape[1]
    }
    
    torch.save(multimodal_model, f"{model_dir}/multimodal_model.pt")
    logger.info(f"Saved multimodal model to {model_dir}/multimodal_model.pt")
    
    # Visualize confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Crisis', 'Crisis'],
                yticklabels=['Non-Crisis', 'Crisis'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Multimodal Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{model_dir}/multimodal_confusion_matrix.png")
    
    return model, accuracy

# Function to evaluate real-time inputs
def evaluate_input(text, audio_file=None, image_file=None, model_dir="models"):
    """
    Evaluate a user input for crisis detection
    """
    logger.info("Evaluating input for crisis detection...")
    
    result = {
        "text_score": None,
        "bert_score": None,
        "multimodal_score": None,
        "final_score": None,
        "is_crisis": False,
        "confidence": 0.0,
        "explanation": ""
    }
    
    # Check if models exist
    text_model_path = f"{model_dir}/text_model.joblib"
    bert_model_path = f"{model_dir}/bert_model.pt"
    multimodal_model_path = f"{model_dir}/multimodal_model.pt"
    
    # Process with text model if available
    if os.path.exists(text_model_path):
        try:
            text_model = joblib.load(text_model_path)
            vectorizer = text_model["vectorizer"]
            classifier = text_model["classifier"]
            
            # Transform input
            text_vec = vectorizer.transform([text])
            
            # Predict
            text_score = classifier.predict_proba(text_vec)[0, 1]
            result["text_score"] = float(text_score)
            
            logger.info(f"Text model crisis score: {text_score:.4f}")
            
            # Get top features
            if text_score > 0.5:
                # Get feature importance
                feature_names = vectorizer.get_feature_names_out()
                feature_importance = classifier.feature_importances_
                
                # Get features present in text
                present_features = {}
                for word in text.lower().split():
                    if word in feature_names:
                        idx = np.where(feature_names == word)[0]
                        if len(idx) > 0:
                            present_features[word] = feature_importance[idx[0]]
                
                # Sort by importance
                sorted_features = sorted(present_features.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:5]
                
                feature_explanation = "Key concern indicators: " + ", ".join([f"{word}" for word, _ in top_features])
                result["explanation"] += feature_explanation
        except Exception as e:
            logger.error(f"Error in text model evaluation: {e}")
    
    # Process with BERT model if available
    if os.path.exists(bert_model_path):
        try:
            # Use CUDA if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model
            checkpoint = torch.load(bert_model_path, map_location=device)
            tokenizer = checkpoint['tokenizer']
            
            # Load BERT
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            model = BERTClassifier(bert_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Tokenize input
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Prepare input
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_type_ids = encoding['token_type_ids'].to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, token_type_ids)
                probs = torch.softmax(outputs, dim=1)
                bert_score = probs[0, 1].item()
            
            result["bert_score"] = float(bert_score)
            logger.info(f"BERT model crisis score: {bert_score:.4f}")
        except Exception as e:
            logger.error(f"Error in BERT model evaluation: {e}")
    
    # Process with multimodal model if all inputs available
    if os.path.exists(multimodal_model_path) and audio_file and image_file:
        try:
            # Use CUDA if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model
            checkpoint = torch.load(multimodal_model_path, map_location=device)
            
            # Extract components
            text_vectorizer = checkpoint["text_vectorizer"]
            text_dim = checkpoint["text_dim"]
            audio_dim = checkpoint["audio_dim"]
            image_dim = checkpoint["image_dim"]
            
            # Initialize model
            model = MultimodalFusionModel(text_dim, audio_dim, image_dim)
            model.load_state_dict(checkpoint["model"])
            model.to(device)
            model.eval()
            
            # Process text
            text_vec = text_vectorizer.transform([text]).toarray()
            text_tensor = torch.FloatTensor(text_vec).to(device)
            
            # Process audio (placeholder for actual audio processing)
            # In a real system, we would extract MFCC features from the audio file
            audio_features = np.random.rand(1, audio_dim)  # Placeholder
            audio_tensor = torch.FloatTensor(audio_features).to(device)
            
            # Process image (placeholder for actual image processing)
            # In a real system, we would extract features from the image using a CNN
            image_features = np.random.rand(1, image_dim)  # Placeholder
            image_tensor = torch.FloatTensor(image_features).to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(text_tensor, audio_tensor, image_tensor)
                probs = torch.softmax(outputs, dim=1)
                multimodal_score = probs[0, 1].item()
            
            result["multimodal_score"] = float(multimodal_score)
            logger.info(f"Multimodal model crisis score: {multimodal_score:.4f}")
        except Exception as e:
            logger.error(f"Error in multimodal model evaluation: {e}")
    
    # Compute final score (weighted average of available scores)
    scores = []
    weights = []
    
    if result["text_score"] is not None:
        scores.append(result["text_score"])
        weights.append(1.0)
    
    if result["bert_score"] is not None:
        scores.append(result["bert_score"])
        weights.append(2.0)  # BERT gets higher weight
    
    if result["multimodal_score"] is not None:
        scores.append(result["multimodal_score"])
        weights.append(3.0)  # Multimodal gets highest weight
    
    if scores:
        final_score = np.average(scores, weights=weights)
        result["final_score"] = float(final_score)
        result["confidence"] = float(min(1.0, max(0.0, abs(final_score - 0.5) * 2)))
        result["is_crisis"] = final_score >= 0.5
        
        logger.info(f"Final crisis score: {final_score:.4f} (Confidence: {result['confidence']:.4f})")
        
        # Generate explanation if not already present
        if not result["explanation"]:
            if result["is_crisis"]:
                result["explanation"] = "The message contains language patterns consistent with a mental health crisis."
            else:
                result["explanation"] = "The message does not appear to indicate an immediate mental health crisis."
    else:
        result["explanation"] = "No models were able to evaluate the input."
    
    return result

# Main function
def main():
    """
    Main function to train and evaluate models
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train mental health crisis detection models")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for datasets")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory for saving models")
    parser.add_argument("--train_text", action="store_true", help="Train text-based model")
    parser.add_argument("--train_bert", action="store_true", help="Train BERT-based model")
    parser.add_argument("--train_multimodal", action="store_true", help="Train multimodal model")
    parser.add_argument("--train_all", action="store_true", help="Train all models")
    parser.add_argument("--eval_text", type=str, help="Text to evaluate")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Download datasets
    download_datasets(args.data_dir)
    
    # Train models as specified
    if args.train_text or args.train_all:
        text_df = load_text_data(args.data_dir)
        if text_df is not None:
            train_text_model(text_df, args.model_dir)
    
    if args.train_bert or args.train_all:
        text_df = load_text_data(args.data_dir)
        if text_df is not None:
            train_bert_model(text_df, args.model_dir)
    
    if args.train_multimodal or args.train_all:
        text_df, audio_df, image_df = load_multimodal_data(args.data_dir)
        if text_df is not None and audio_df is not None and image_df is not None:
            train_multimodal_model(text_df, audio_df, image_df, args.model_dir)
    
    # Evaluate input text if provided
    if args.eval_text:
        result = evaluate_input(args.eval_text, model_dir=args.model_dir)
        print("\nEvaluation Result:")
        print(f"Crisis Score: {result['final_score']:.4f}")
        print(f"Is Crisis: {'Yes' if result['is_crisis'] else 'No'}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Explanation: {result['explanation']}")

if __name__ == "__main__":
    main()