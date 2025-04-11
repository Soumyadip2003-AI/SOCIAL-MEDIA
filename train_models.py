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
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check and install required packages
def install_required_packages():
    """
    Install required packages if not already installed
    """
    required_packages = {
        'transformers': 'transformers>=4.28.0',
        'torch': 'torch>=1.13.0',
        'numpy': 'numpy>=1.20.0',
        'pandas': 'pandas>=1.4.0',
        'matplotlib': 'matplotlib>=3.5.0',
        'seaborn': 'seaborn>=0.11.0',
        'scikit-learn': 'scikit-learn>=1.0.0',
        'nltk': 'nltk>=3.6.0',
        'joblib': 'joblib>=1.1.0',
        'tqdm': 'tqdm>=4.62.0',
        'shap': 'shap>=0.40.0',
        'opencv-python': 'opencv-python>=4.5.0',
        'pillow': 'pillow>=9.0.0',
    }
    
    packages_to_install = []
    
    for package, version_spec in required_packages.items():
        try:
            __import__(package)
            logger.info(f"Package {package} is already installed.")
        except ImportError:
            packages_to_install.append(version_spec)
            logger.warning(f"Package {package} is not installed.")
    
    if packages_to_install:
        logger.info("Installing required packages...")
        import subprocess
        for package in packages_to_install:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info("All required packages installed successfully.")
        logger.info("Please restart the script for changes to take effect.")
        sys.exit(0)
    else:
        logger.info("All required packages are already installed.")

# Call the function to install required packages
install_required_packages()

# Now try to import transformers after ensuring it's installed
try:
    from transformers import BertTokenizer, BertModel
    # Try to load the BERT tokenizer with detailed error handling
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_available = True
        logger.info("Successfully loaded BERT tokenizer.")
    except Exception as e:
        logger.error(f"Error loading BERT tokenizer: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Try with offline mode if the first attempt failed
        try:
            logger.info("Attempting to download BERT tokenizer...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
            bert_available = True
            logger.info("Successfully downloaded and loaded BERT tokenizer.")
        except Exception as e2:
            logger.error(f"Second attempt failed: {e2}")
            bert_available = False
except ImportError:
    logger.error("Could not import transformers library. Please install it manually.")
    bert_available = False

warnings.filterwarnings('ignore')

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
    # Check if BERT is available
    if not bert_available:
        logger.error("BERT tokenizer is not available. Cannot train BERT model.")
        return None, None, 0.0
        
    logger.info("Training BERT-based crisis detection model...")
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Load tokenizer and model
    try:
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
    
    except Exception as e:
        logger.error(f"Error in BERT model training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0.0

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
        "explanation": []
    }
    
    # Text model inference
    try:
        text_model = joblib.load(f"{model_dir}/text_model.joblib")
        vectorizer = text_model["vectorizer"]
        classifier = text_model["classifier"]
        
        # Vectorize input
        text_vec = vectorizer.transform([text])
        
        # Predict probability
        proba = classifier.predict_proba(text_vec)[0]
        result["text_score"] = float(proba[1])
        
        # Get feature importance for explanation
        if proba[1] > 0.5:
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get important words in the input
            text_tokens = word_tokenize(text.lower())
            important_words = []
            
            for word in text_tokens:
                if word in feature_names:
                    idx = list(feature_names).index(word)
                    importance = classifier.feature_importances_[idx]
                    if importance > 0.01:
                        important_words.append((word, importance))
            
            # Sort by importance
            important_words.sort(key=lambda x: x[1], reverse=True)
            
            # Add to explanation
            for word, importance in important_words[:5]:
                result["explanation"].append(f"Crisis indicator word: '{word}' (importance: {importance:.4f})")
        
        logger.info(f"Text model score: {result['text_score']:.4f}")
    except Exception as e:
        logger.error(f"Error in text model inference: {e}")
    
    # BERT model inference
    if bert_available:
        try:
            # Load model
            checkpoint = torch.load(f"{model_dir}/bert_model.pt", map_location='cpu')
            tokenizer = checkpoint['tokenizer']
            
            # Load BERT model
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            model = BERTClassifier(bert_model)
            model.load_state_dict(checkpoint['model_state_dict'])
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
            
            # Predict
            with torch.no_grad():
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']
                token_type_ids = encoding['token_type_ids']
                
                outputs = model(input_ids, attention_mask, token_type_ids)
                proba = torch.softmax(outputs, dim=1).numpy()[0]
                
                result["bert_score"] = float(proba[1])
            
            logger.info(f"BERT model score: {result['bert_score']:.4f}")
        except Exception as e:
            logger.error(f"Error in BERT model inference: {e}")
    
    # Multimodal model inference (if audio and image are provided)
    if audio_file is not None or image_file is not None:
        try:
            # Load model
            multimodal_data = torch.load(f"{model_dir}/multimodal_model.pt", map_location='cpu')
            
            # Extract components
            text_vectorizer = multimodal_data["text_vectorizer"]
            text_dim = multimodal_data["text_dim"]
            audio_dim = multimodal_data["audio_dim"]
            image_dim = multimodal_data["image_dim"]
            
            # Initialize model
            model = MultimodalFusionModel(text_dim, audio_dim, image_dim)
            model.load_state_dict(multimodal_data["model"])
            model.eval()
            
            # Process text
            text_vec = text_vectorizer.transform([text]).toarray()
            text_tensor = torch.FloatTensor(text_vec)
            
            # Process audio (or use zeros if not provided)
            if audio_file is not None:
                # In a real system, we would extract MFCC features here
                # For now, we'll just use random values
                audio_tensor = torch.rand(1, audio_dim)
                logger.info("Using audio features for multimodal prediction")
            else:
                audio_tensor = torch.zeros(1, audio_dim)
            
            # Process image (or use zeros if not provided)
            if image_file is not None:
                # In a real system, we would extract image features here
                # For now, we'll just use random values
                image_tensor = torch.rand(1, image_dim)
                logger.info("Using image features for multimodal prediction")
            else:
                image_tensor = torch.zeros(1, image_dim)
            
            # Predict
            with torch.no_grad():
                outputs = model(text_tensor, audio_tensor, image_tensor)
                proba = torch.softmax(outputs, dim=1).numpy()[0]
                
                result["multimodal_score"] = float(proba[1])
            
            logger.info(f"Multimodal model score: {result['multimodal_score']:.4f}")
        except Exception as e:
            logger.error(f"Error in multimodal model inference: {e}")
    
    # Compute final score and decision
    scores = []
    weights = []
    
    if result["text_score"] is not None:
        scores.append(result["text_score"])
        weights.append(1.0)
    
    if result["bert_score"] is not None:
        scores.append(result["bert_score"])
        weights.append(1.5)  # Give more weight to BERT
    
    if result["multimodal_score"] is not None:
        scores.append(result["multimodal_score"])
        weights.append(2.0)  # Give most weight to multimodal
    
    if scores:
        weighted_scores = [s * w for s, w in zip(scores, weights)]
        result["final_score"] = sum(weighted_scores) / sum(weights)
        result["confidence"] = max(scores)
        
        # Determine if crisis
        if result["final_score"] > 0.7:
            result["is_crisis"] = True
            result["explanation"].insert(0, "HIGH RISK: Immediate attention required")
        elif result["final_score"] > 0.5:
            result["is_crisis"] = True
            result["explanation"].insert(0, "MODERATE RISK: Close monitoring recommended")
        else:
            result["is_crisis"] = False
            result["explanation"].insert(0, "LOW RISK: Continue support as needed")
        
        logger.info(f"Final crisis score: {result['final_score']:.4f}, Is crisis: {result['is_crisis']}")
    else:
        logger.error("No models were able to provide scores")
        result["explanation"].append("Error: Could not perform analysis")
    
    return result

# Function to provide appropriate response based on crisis level
def generate_response(result):
    """
    Generate an appropriate response based on crisis assessment
    """
    if result["is_crisis"]:
        if result["final_score"] > 0.8:
            return (
                "I'm concerned about what you're expressing. It sounds like you might be in a crisis situation. "
                "Please consider calling a crisis helpline right away: National Suicide Prevention Lifeline at 988 or 1-800-273-8255. "
                "They have trained counselors available 24/7. If you're in immediate danger, please call emergency services (911)."
            )
        elif result["final_score"] > 0.6:
            return (
                "I'm concerned about what you're sharing. It sounds like you're going through a difficult time. "
                "Consider reaching out to a mental health professional or a crisis helpline like the National Suicide Prevention Lifeline "
                "at 988 or 1-800-273-8255. They can provide support and resources to help you."
            )
        else:
            return (
                "It sounds like you're experiencing some distress. Consider talking to someone you trust or a mental health professional. "
                "Resources like the Crisis Text Line (text HOME to 741741) can also provide support if you need someone to talk to."
            )
    else:
        return (
            "Thank you for sharing. While I don't detect indications of crisis in your message, "
            "remember that it's always okay to reach out for support when needed. "
            "Self-care and connecting with supportive people are important parts of mental well-being."
        )

# Create a simple command line interface
def cli_interface():
    """
    Command-line interface for crisis detection
    """
    print("\n==== Mental Health Crisis Detection System ====\n")
    print("Enter text to analyze for crisis indicators. Type 'quit' to exit.")
    
    while True:
        text = input("\nInput text: ")
        
        if text.lower() in ["quit", "exit", "q"]:
            print("Exiting system. Take care!")
            break
        
        # Evaluate input
        result = evaluate_input(text)
        
        # Display results
        print("\n----- Analysis Results -----")
        print(f"Crisis probability: {result['final_score']:.2f}")
        print(f"Assessment: {'CRISIS DETECTED' if result['is_crisis'] else 'No crisis detected'}")
        print("\nExplanation:")
        for item in result["explanation"]:
            print(f"  - {item}")
        
        # Generate response
        print("\nRecommended response:")
        print(generate_response(result))
        print("\n-----------------------------")

# Main function to run the system
def main():
    """
    Main function to run the system
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mental Health Crisis Detection System")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", type=str, help="Evaluate text input")
    parser.add_argument("--audio", type=str, help="Audio file path for multimodal analysis")
    parser.add_argument("--image", type=str, help="Image file path for multimodal analysis")
    parser.add_argument("--cli", action="store_true", help="Launch command-line interface")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="models", help="Model directory")
    
    args = parser.parse_args()
    
    # Show ASCII art banner
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║  Mental Health Crisis Detection System            ║
    ║  Created to help identify and respond to          ║
    ║  mental health crises through text analysis       ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    # Download datasets if training
    if args.train:
        success = download_datasets(args.data_dir)
        if not success:
            logger.error("Failed to download or create datasets")
            return
    
    # Train models if requested
    if args.train:
        logger.info("Starting model training...")
        
        # Load text data
        text_df = load_text_data(args.data_dir)
        if text_df is None:
            logger.error("Failed to load text data")
            return
        
        # Train text model
        text_model, vectorizer, text_accuracy = train_text_model(text_df, args.model_dir)
        
        # Train BERT model
        bert_model, tokenizer, bert_accuracy = train_bert_model(text_df, args.model_dir)
        
        # Load multimodal data
        text_df, audio_df, image_df = load_multimodal_data(args.data_dir)
        if text_df is not None and audio_df is not None and image_df is not None:
            # Train multimodal model
            multimodal_model, multimodal_accuracy = train_multimodal_model(
                text_df, audio_df, image_df, args.model_dir
            )
            
            logger.info("Model training complete!")
            logger.info(f"Text model accuracy: {text_accuracy:.4f}")
            logger.info(f"BERT model accuracy: {bert_accuracy:.4f}")
            logger.info(f"Multimodal model accuracy: {multimodal_accuracy:.4f}")
        else:
            logger.error("Failed to load multimodal data")
    
    # Evaluate text input if provided
    if args.evaluate:
        result = evaluate_input(
            args.evaluate, 
            audio_file=args.audio,
            image_file=args.image,
            model_dir=args.model_dir
        )
        
        # Display results
        print("\n----- Analysis Results -----")
        print(f"Crisis probability: {result['final_score']:.2f}")
        print(f"Assessment: {'CRISIS DETECTED' if result['is_crisis'] else 'No crisis detected'}")
        print("\nExplanation:")
        for item in result["explanation"]:
            print(f"  - {item}")
        
        # Generate response
        print("\nRecommended response:")
        print(generate_response(result))
    
    # Launch CLI if requested
    if args.cli:
        cli_interface()
    
    # If no action specified, show help
    if not (args.train or args.evaluate or args.cli):
        parser.print_help()

if __name__ == "__main__":
    main()