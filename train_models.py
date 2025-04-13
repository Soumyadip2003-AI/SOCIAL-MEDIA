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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
import re
from collections import Counter

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
    from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
    from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
    # Try to load the BERT tokenizer with detailed error handling
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_available = True
        logger.info("Successfully loaded BERT tokenizer.")
    except Exception as e:
        logger.error(f"Error loading BERT tokenizer: {e}")
        
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

try:
    import spacy
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_md")
        spacy_available = True
        logger.info("Successfully loaded spaCy model.")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        try:
            logger.info("Attempting to download spaCy model...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
            nlp = spacy.load("en_core_web_md")
            spacy_available = True
            logger.info("Successfully downloaded and loaded spaCy model.")
        except Exception as e2:
            logger.error(f"Second attempt failed: {e2}")
            spacy_available = False
except ImportError:
    logger.error("Could not import spaCy library. Please install it manually.")
    spacy_available = False

try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    logger.error("Could not import XGBoost library. Some models will not be available.")
    xgboost_available = False

try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    logger.error("Could not import LightGBM library. Some models will not be available.")
    lightgbm_available = False

try:
    from gensim.models import Word2Vec, FastText
    gensim_available = True
except ImportError:
    logger.error("Could not import Gensim library. Some word embeddings will not be available.")
    gensim_available = False

warnings.filterwarnings('ignore')
# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

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

# Enhanced synthetic text data creation with more realistic patterns and larger dataset
def create_synthetic_text_data(target_dir):
    """
    Create enhanced synthetic text data for training if real data is unavailable
    """
    logger.info("Creating synthetic text data for demonstration...")
    
    # Crisis indicators
    suicide_words = ["suicide", "kill myself", "end my life", "take my life", "die", "death"]
    hopelessness_words = ["hopeless", "worthless", "useless", "burden", "can't go on", "no reason", "pointless"]
    planning_words = ["plan", "method", "pills", "rope", "gun", "jump", "bridge", "building", "cut", "wrist"]
    
    # Crisis texts - 500 samples with variations and realistic patterns
    crisis_texts = []
    
    # Generate 100 samples with more nuanced and realistic variations
    for i in range(500):
        # Use different patterns of crisis messages
        pattern = np.random.randint(0, 5)
        
        if pattern == 0:
            # Direct suicidal statement
            suicide_word = np.random.choice(suicide_words)
            text = f"I {'really ' if np.random.random() > 0.5 else ''}want to {suicide_word}{'.' if np.random.random() > 0.5 else ''}"
            
            # Add a context sometimes
            if np.random.random() > 0.6:
                context = np.random.choice([
                    "I can't take it anymore.",
                    "The pain is too much.",
                    "No one would miss me.",
                    "Everyone would be better off.",
                    "I'm a burden to everyone."
                ])
                text += f" {context}"
        
        elif pattern == 1:
            # Expressing hopelessness
            hopeless_word = np.random.choice(hopelessness_words)
            text = f"I feel so {hopeless_word}. " if np.random.random() > 0.5 else f"Everything is {hopeless_word}. "
            
            # Add suicidal thought sometimes
            if np.random.random() > 0.6:
                suicide_phrase = np.random.choice([
                    "I just want it all to end.",
                    "I don't see a way out.",
                    "I can't keep living like this.",
                    "I don't think I can go on."
                ])
                text += suicide_phrase
        
        elif pattern == 2:
            # Planning or method
            planning_word = np.random.choice(planning_words)
            text = f"I've been thinking about {'using ' if planning_word in ['pills', 'rope', 'gun'] else ''}{planning_word}{'ing' if planning_word in ['jump', 'cut'] else ''}."
            
            # Add more detail sometimes
            if np.random.random() > 0.7:
                detail = np.random.choice([
                    "I have it all planned out.",
                    "I've been researching methods.",
                    "I've already written my note.",
                    "No one knows what I'm planning.",
                    "Tonight might be the night."
                ])
                text += f" {detail}"
        
        elif pattern == 3:
            # Goodbye message
            text = np.random.choice([
                "I just wanted to say goodbye. Thank you for everything.",
                "This will be my last message. I'm sorry for everything.",
                "I can't do this anymore. This is goodbye.",
                "It's time for me to go. Forever.",
                "By the time you read this, I'll be gone."
            ])
            
            # Add reasoning sometimes
            if np.random.random() > 0.6:
                reason = np.random.choice([
                    "The pain is too much to bear.",
                    "I've been fighting for too long.",
                    "No one can help me now.",
                    "I'm too tired to keep going."
                ])
                text += f" {reason}"
        
        else:
            # Mixed elements
            element1 = np.random.choice([
                "I can't see any future for myself.",
                "Every day is just more pain.",
                "No one would miss me if I was gone.",
                "I feel like a failure at everything.",
                "I've been thinking about death a lot."
            ])
            
            element2 = np.random.choice([
                "I've made up my mind.",
                "I've found a way out.",
                "It will all be over soon.",
                "I won't have to suffer much longer.",
                "At least I won't be a burden anymore."
            ])
            
            text = f"{element1} {element2}" if np.random.random() > 0.5 else element1
        
        # Add some variations in capitalization and punctuation
        if np.random.random() > 0.8:
            text = text.lower()
        if np.random.random() > 0.9:
            text = text.replace(".", "")
        
        crisis_texts.append(text)
    
    # Non-crisis texts with various levels of distress but not immediate danger
    non_crisis_texts = []
    
    # Mental health vocabulary
    mental_health_terms = ["depressed", "anxious", "stress", "therapy", "counseling", "medication", "sad", "worried"]
    support_terms = ["friend", "family", "therapist", "doctor", "support group", "hotline", "resources", "self-care"]
    positive_action_terms = ["trying", "working on", "getting help", "improving", "coping", "managing", "progress"]
    
    # Generate 500 non-crisis samples
    for i in range(500):
        # Use different patterns of non-crisis messages
        pattern = np.random.randint(0, 5)
        
        if pattern == 0:
            # Expressing mental health issues but with coping
            mh_term = np.random.choice(mental_health_terms)
            positive_term = np.random.choice(positive_action_terms)
            text = f"I've been feeling {mh_term} lately, but I'm {positive_term}."
            
            # Add more context sometimes
            if np.random.random() > 0.6:
                context = np.random.choice([
                    "It's not easy, but I'm trying to stay positive.",
                    "One day at a time, right?",
                    "I know I'll get through this eventually.",
                    "There are still some good moments."
                ])
                text += f" {context}"
        
        elif pattern == 1:
            # Seeking help or advice
            support_term = np.random.choice(support_terms)
            text = f"I've been talking to my {support_term} about my feelings."
            
            # Add more detail sometimes
            if np.random.random() > 0.7:
                detail = np.random.choice([
                    "It's been helpful to have someone to talk to.",
                    "They suggested some good coping strategies.",
                    "I'm learning new ways to manage my thoughts.",
                    "It's a process, but I'm committed to getting better."
                ])
                text += f" {detail}"
        
        elif pattern == 2:
            # Bad day but with perspective
            text = np.random.choice([
                "Today was really rough, but tomorrow is another day.",
                "I'm struggling right now, but I know this will pass.",
                "I'm having a hard time, but I'm hanging in there.",
                "This week has been awful, but I'm taking it one day at a time.",
                "I feel overwhelmed, but I'm not giving up."
            ])
            
            # Add positive element sometimes
            if np.random.random() > 0.6:
                positive = np.random.choice([
                    "My friends have been really supportive.",
                    "I'm going to try some self-care tonight.",
                    "I have an appointment with my therapist next week.",
                    "At least I have my pet/hobby/etc to comfort me."
                ])
                text += f" {positive}"
        
        elif pattern == 3:
            # Specific stressors
            stressor = np.random.choice([
                "work", "school", "relationship", "family", "money", 
                "health", "pandemic", "social anxiety", "insomnia"
            ])
            text = f"My {stressor} issues are really getting to me lately."
            
            # Add coping strategy sometimes
            if np.random.random() > 0.7:
                strategy = np.random.choice([
                    "I'm trying to practice mindfulness.",
                    "I've started journaling about it.",
                    "Exercise has been helping a bit.",
                    "I'm limiting my social media time.",
                    "Talking about it helps."
                ])
                text += f" {strategy}"
        
        else:
            # Balanced perspective
            negative = np.random.choice([
                "I have dark thoughts sometimes",
                "I feel empty inside",
                "Life seems meaningless occasionally",
                "I wonder if things will ever get better",
                "I don't always see the point of trying"
            ])
            
            positive = np.random.choice([
                "but I would never act on those thoughts",
                "but I know that's just the depression talking",
                "but I have people who care about me",
                "but I'm stronger than these feelings",
                "but I'm using the coping skills I've learned"
            ])
            
            text = f"{negative}, {positive}."
        
        non_crisis_texts.append(text)
    
    # Create DataFrame
    texts = crisis_texts + non_crisis_texts
    labels = [1] * len(crisis_texts) + [0] * len(non_crisis_texts)  # 1 for crisis, 0 for non-crisis
    
    # Add an additional layer of complexity - create social media styles
    social_texts = []
    social_labels = []
    
    for text, label in zip(texts, labels):
        # Original text
        social_texts.append(text)
        social_labels.append(label)
        
        # Sometimes add social media style variations
        if np.random.random() > 0.7:
            # Twitter-like abbreviation
            abbrev_text = text
            for word, repl in [("you", "u"), ("are", "r"), ("for", "4"), ("to", "2"), 
                               ("before", "b4"), ("see", "c"), ("later", "l8r")]:
                if f" {word} " in f" {abbrev_text} ":
                    abbrev_text = re.sub(f"\\b{word}\\b", repl, abbrev_text)
            
            social_texts.append(abbrev_text)
            social_labels.append(label)
        
        if np.random.random() > 0.8:
            # Add emojis (represented as text)
            emoji_text = text
            if label == 1:  # Crisis
                emoji = np.random.choice(["😔", "😢", "😭", "💔", "😞", "😓", "😪", "🥀"])
                emoji_text += f" {emoji}"
            else:  # Non-crisis
                emoji = np.random.choice(["🙂", "🤔", "😕", "😐", "🤷", "⏳", "🌧️", "☀️"])
                emoji_text += f" {emoji}"
            
            social_texts.append(emoji_text)
            social_labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': social_texts,
        'label': social_labels
    })
    
    # Add a data augmentation technique - slight word replacements using synonyms
    augmented_texts = []
    augmented_labels = []
    
    # Simple synonym mappings
    synonyms = {
        "sad": ["unhappy", "depressed", "gloomy", "miserable", "down"],
        "happy": ["glad", "cheerful", "content", "pleased", "joyful"],
        "angry": ["mad", "furious", "upset", "annoyed", "enraged"],
        "scared": ["afraid", "frightened", "terrified", "fearful", "anxious"],
        "tired": ["exhausted", "weary", "fatigued", "drained", "sleepy"],
        "pain": ["suffering", "agony", "distress", "hurt", "discomfort"],
        "life": ["existence", "living", "being", "reality", "world"],
        "death": ["end", "demise", "passing", "expiration", "dying"],
        "friend": ["buddy", "pal", "companion", "mate", "ally"],
        "family": ["relatives", "kin", "folks", "loved ones", "relations"],
        "help": ["assistance", "aid", "support", "guidance", "advice"],
        "problem": ["issue", "trouble", "difficulty", "concern", "complication"],
        "think": ["believe", "consider", "ponder", "reflect", "contemplate"],
        "feel": ["sense", "experience", "perceive", "undergo", "detect"]
    }
    
    for text, label in zip(df['text'], df['label']):
        # Original text
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # Create synonym-replaced versions (30% chance)
        if np.random.random() > 0.7:
            words = text.split()
            new_text = []
            
            for word in words:
                word_lower = word.lower()
                if word_lower in synonyms and np.random.random() > 0.5:
                    # Replace with a synonym, preserving capitalization
                    synonym = np.random.choice(synonyms[word_lower])
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    new_text.append(synonym)
                else:
                    new_text.append(word)
            
            augmented_text = " ".join(new_text)
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)
    
    # Create final DataFrame
    final_df = pd.DataFrame({
        'text': augmented_texts,
        'label': augmented_labels
    })
    
    # Shuffle the data
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    final_df.to_csv(f"{target_dir}/mental_health_posts.csv", index=False)
    logger.info(f"Saved synthetic text dataset with {len(final_df)} samples")
    
    # Create a validation dataset with different patterns
    validation_crisis = [
        "I'm ready to end it all tonight",
        "I've got the pills and I'm going to take them all",
        "This world would be better without me in it",
        "I've written my goodbye letters",
        "I can't go on anymore, this is my final message",
        "Going to jump off the bridge tonight",
        "I'm done with this life, it's time to go",
        "I've made my decision and I'm finally at peace",
        "Tonight is the night I finally escape forever",
        "I'm gonna shoot myself when I get home",
        "I've made my peace with ending things",
        "Nobody would even notice if I was gone",
        "I'm tired of being a burden, it ends tonight",
        "I just can't see any other way out of this pain",
        "I don't want to wake up tomorrow",
        "I'm going to hang myself tonight",
        "Just wanted to say goodbye one last time",
        "The voices won't stop until I end it",
        "I already know how I'll do it, and when",
        "I've been researching the most painless methods"
    ]
    
    validation_non_crisis = [
        "I'm feeling better after starting my new medication",
        "Therapy has been really helping me work through my depression",
        "I'm struggling but my support group understands what I'm going through",
        "Even though I'm sad today, I know tomorrow might be better",
        "My anxiety is acting up but I'm practicing my breathing exercises",
        "Having a hard time but I called the crisis line and they helped",
        "I don't feel great but I'm not giving up on recovery",
        "Depression is tough but I'm fighting through each day",
        "I've been feeling down lately but my friends have been supportive",
        "Sometimes I wonder what's the point, but I still have hope",
        "Life feels heavy right now, but I'm taking it one moment at a time",
        "I'm thankful for my therapist helping me through this rough patch",
        "Having dark thoughts but I know not to trust them",
        "My mental health is a struggle but I'm committed to self-care",
        "Bad day today but I'm using my coping skills",
        "Felt really low this morning but went for a walk and it helped",
        "The depression comes in waves, but I've survived them before",
        "I miss feeling normal but I know healing isn't linear",
        "Grief is hitting me hard lately but I'm letting myself feel it",
        "Sometimes existing is exhausting but I'm still trying"
    ]
    
    # Create validation DataFrame
    val_texts = validation_crisis + validation_non_crisis
    val_labels = [1] * len(validation_crisis) + [0] * len(validation_non_crisis)
    
    val_df = pd.DataFrame({
        'text': val_texts,
        'label': val_labels
    })
    
    # Shuffle validation data
    val_df = val_df.sample(frac=1, random_state=43).reset_index(drop=True)
    
    # Save validation dataset
    val_df.to_csv(f"{target_dir}/validation_posts.csv", index=False)
    logger.info(f"Saved validation dataset with {len(val_df)} samples")
    
    return final_df

# Create enhanced synthetic audio features
def create_synthetic_audio_features(target_dir):
    """
    Create synthetic audio features for multimodal model
    """
    logger.info("Creating synthetic audio features...")
    
    # Create 1000 samples (500 crisis, 500 non-crisis)
    n_samples = 1000
    
    # More realistic MFCC features
    n_mfcc = 13
    n_frames = 10
    
    # Generate more realistic features
    features = []
    
    # Define different patterns for crisis vs non-crisis audio
    crisis_mean = np.array([-20, -15, 8, 5, -3, 2, 1, 0, -1, -2, 3, -5, 4])
    crisis_std = np.array([10, 8, 6, 5, 4, 3, 3, 2, 2, 3, 2, 2, 3])
    
    non_crisis_mean = np.array([-15, -10, 5, 3, -1, 1, 0, 1, 0, -1, 2, -3, 2])
    non_crisis_std = np.array([8, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2])
    
    # Generate crisis audio features
    for i in range(500):
        sample_features = []
        for frame in range(n_frames):
            # Add some temporal structure
            frame_factor = 1 + 0.1 * (frame / n_frames)
            
            # Generate frame features
            frame_features = np.random.normal(
                crisis_mean * frame_factor,
                crisis_std,
                size=n_mfcc
            )
            
            sample_features.extend(frame_features)
        
        features.append(sample_features)
    
    # Generate non-crisis audio features
    for i in range(500):
        sample_features = []
        for frame in range(n_frames):
            # Add some temporal structure
            frame_factor = 1 + 0.1 * (frame / n_frames)
            
            # Generate frame features
            frame_features = np.random.normal(
                non_crisis_mean * frame_factor,
                non_crisis_std,
                size=n_mfcc
            )
            
            sample_features.extend(frame_features)
        
        features.append(sample_features)
    
    # Create feature columns
    feature_cols = []
    for frame in range(n_frames):
        for mfcc in range(n_mfcc):
            feature_cols.append(f'mfcc_{frame}_{mfcc}')
    
    # Generate labels
    labels = np.zeros(n_samples)
    labels[:500] = 1  # First half are crisis
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_cols)
    df['label'] = labels
    
    # Add derived features
    # Energy (sum of squared MFCC0 values)
    df['energy'] = np.sum([df[f'mfcc_{frame}_0']**2 for frame in range(n_frames)], axis=0)
    
    # Spectral centroid approximation
    df['spectral_centroid'] = np.sum([df[f'mfcc_{frame}_{mfcc}'] * (mfcc + 1) 
                                      for frame in range(n_frames) 
                                      for mfcc in range(1, n_mfcc)], axis=0)
    
    # Spectral flux approximation (frame-to-frame differences)
    for frame in range(1, n_frames):
        for mfcc in range(n_mfcc):
            df[f'flux_{frame}_{mfcc}'] = df[f'mfcc_{frame}_{mfcc}'] - df[f'mfcc_{frame-1}_{mfcc}']
    
    # Speech rate approximation (zero crossing rate proxy)
    df['speech_rate'] = np.random.normal(
        5 + 2 * labels,  # Higher rate for crisis
        1.5,
        size=n_samples
    )
    
    # Save to CSV
    df.to_csv(f"{target_dir}/audio_features.csv", index=False)
    
    # Create validation set
    val_size = 100
    
    # Generate validation features
    val_features = []
    
    # Generate crisis audio features for validation
    for i in range(50):
        sample_features = []
        for frame in range(n_frames):
            frame_factor = 1 + 0.12 * (frame / n_frames)  # Slightly different
            
            frame_features = np.random.normal(
                crisis_mean * frame_factor * 1.05,  # Slightly different distribution
                crisis_std * 0.95,
                size=n_mfcc
            )
            
        sample_features.extend(frame_features)
            
        val_features.append(sample_features)
    
    # Generate non-crisis audio features for validation
    for i in range(50):
        sample_features = []
        for frame in range(n_frames):
            frame_factor = 1 + 0.09 * (frame / n_frames)  # Slightly different
            
            frame_features = np.random.normal(
                non_crisis_mean * frame_factor * 0.95,  # Slightly different distribution
                non_crisis_std * 1.05,
                size=n_mfcc
            )
            
        sample_features.extend(frame_features)
        
        val_features.append(sample_features)
    
    # Create validation DataFrame
    val_df = pd.DataFrame(val_features, columns=feature_cols)
    val_df['label'] = np.concatenate([np.ones(50), np.zeros(50)])
    
    # Add derived features for validation set
    val_df['energy'] = np.sum([val_df[f'mfcc_{frame}_0']**2 for frame in range(n_frames)], axis=0)
    val_df['spectral_centroid'] = np.sum([val_df[f'mfcc_{frame}_{mfcc}'] * (mfcc + 1) 
                                          for frame in range(n_frames) 
                                          for mfcc in range(1, n_mfcc)], axis=0)
    
    for frame in range(1, n_frames):
        for mfcc in range(n_mfcc):
            val_df[f'flux_{frame}_{mfcc}'] = val_df[f'mfcc_{frame}_{mfcc}'] - val_df[f'mfcc_{frame-1}_{mfcc}']
    
    val_df['speech_rate'] = np.random.normal(
        5 + 2 * val_df['label'],  # Higher rate for crisis
        1.5,
        size=val_size
    )
    
    # Save validation set
    val_df.to_csv(f"{target_dir}/audio_features_validation.csv", index=False)
    
    logger.info(f"Saved synthetic audio features with {len(df)} samples and {len(val_df)} validation samples")
    return df

# Create synthetic image features
def create_synthetic_image_features(target_dir):
    """
    Create synthetic image features for multimodal analysis
    """
    logger.info("Creating synthetic image features...")
    
    # Create 1000 samples (500 crisis, 500 non-crisis)
    n_samples = 1000
    
    # Define feature dimensions for a CNN feature extractor
    feature_dim = 512
    
    # Different distributions for crisis vs non-crisis images
    crisis_mean = np.zeros(feature_dim)
    crisis_std = np.ones(feature_dim)
    
    # Make specific features more pronounced for crisis images
    crisis_indices = np.random.choice(feature_dim, size=50, replace=False)
    crisis_mean[crisis_indices] = 0.5
    crisis_std[crisis_indices] = 1.5
    
    # Generate features
    features = []
    
    # Crisis features
    for i in range(500):
        feature_vector = np.random.normal(crisis_mean, crisis_std)
        # Add some non-linearity
        feature_vector = np.tanh(feature_vector)
        features.append(feature_vector)
    
    # Non-crisis features
    for i in range(500):
        feature_vector = np.random.normal(0, 1, size=feature_dim)
        # Add some non-linearity
        feature_vector = np.tanh(feature_vector)
        features.append(feature_vector)
    
    # Create feature columns
    feature_cols = [f'feature_{i}' for i in range(feature_dim)]
    
    # Generate labels
    labels = np.zeros(n_samples)
    labels[:500] = 1  # First half are crisis
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_cols)
    df['label'] = labels
    
    # Add derived features - dimensionality reduction approximation
    # PCA-like projection
    for i in range(10):
        weights = np.random.normal(0, 1, size=feature_dim)
        df[f'pca_{i}'] = df[feature_cols].values.dot(weights)
    
    # t-SNE-like clusters
    df['cluster_x'] = np.random.normal(5 * labels, 1 + 0.5 * labels)
    df['cluster_y'] = np.random.normal(0, 1 + 0.5 * labels)
    
    # Save to CSV
    df.to_csv(f"{target_dir}/image_features.csv", index=False)
    
    # Create validation set
    val_size = 100
    
    # Generate validation features
    val_features = []
    
    # Crisis features for validation
    for i in range(50):
        feature_vector = np.random.normal(crisis_mean * 1.05, crisis_std * 0.95)  # Slightly different
        feature_vector = np.tanh(feature_vector)
        val_features.append(feature_vector)
    
    # Non-crisis features for validation
    for i in range(50):
        feature_vector = np.random.normal(0, 1.05, size=feature_dim)  # Slightly different
        feature_vector = np.tanh(feature_vector)
        val_features.append(feature_vector)
    
    # Create validation DataFrame
    val_df = pd.DataFrame(val_features, columns=feature_cols)
    val_df['label'] = np.concatenate([np.ones(50), np.zeros(50)])
    
    # Add derived features for validation
    for i in range(10):
        weights = np.random.normal(0, 1, size=feature_dim)
        val_df[f'pca_{i}'] = val_df[feature_cols].values.dot(weights)
    
    val_df['cluster_x'] = np.random.normal(5 * val_df['label'], 1 + 0.5 * val_df['label'])
    val_df['cluster_y'] = np.random.normal(0, 1 + 0.5 * val_df['label'])
    
    # Save validation set
    val_df.to_csv(f"{target_dir}/image_features_validation.csv", index=False)
    
    logger.info(f"Saved synthetic image features with {len(df)} samples and {len(val_df)} validation samples")
    return df

# Data preprocessing classes
class TextPreprocessor:
    """
    Text preprocessing pipeline
    """
    def __init__(self, use_bert=False, use_spacy=False):
        """
        Initialize the text preprocessor
        
        Parameters:
        -----------
        use_bert : bool
            Whether to use BERT embeddings
        use_spacy : bool
            Whether to use spaCy for preprocessing
        """
        self.stop_words = set(stopwords.words('english'))
        self.use_bert = use_bert and bert_available
        self.use_spacy = use_spacy and spacy_available
        
        # Initialize tokenizers if available
        if self.use_bert:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()  # Set model to evaluation mode
        
        if self.use_spacy:
            self.nlp = nlp
    
    def basic_preprocess(self, text):
        """
        Basic preprocessing steps
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def spacy_preprocess(self, text):
        """
        SpaCy-based preprocessing
        """
        if not self.use_spacy:
            return self.basic_preprocess(text)
        
        doc = self.nlp(text)
        
        # Extract lemmas and filter out stop words and punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)
    
    def get_bert_embeddings(self, text, max_length=128):
        """
        Get BERT embeddings for a text
        """
        if not self.use_bert:
            return None
        
        tokens = self.bert_tokenizer(text, 
                                    return_tensors="pt", 
                                    max_length=max_length, 
                                    padding="max_length", 
                                    truncation=True)
        
        with torch.no_grad():
            outputs = self.bert_model(**tokens)
            # Use CLS token embedding as the sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings[0]  # Return as 1D array
    
    def transform(self, texts, method='basic', return_embeddings=False):
        """
        Transform a list of texts
        
        Parameters:
        -----------
        texts : list
            List of text strings
        method : str
            Preprocessing method: 'basic' or 'spacy'
        return_embeddings : bool
            Whether to return BERT embeddings
        
        Returns:
        --------
        processed_texts : list
            List of preprocessed texts
        embeddings : ndarray (optional)
            BERT embeddings if return_embeddings is True
        """
        if method == 'spacy' and self.use_spacy:
            processed_texts = [self.spacy_preprocess(text) for text in tqdm(texts, desc="SpaCy Processing")]
        else:
            processed_texts = [self.basic_preprocess(text) for text in tqdm(texts, desc="Basic Processing")]
        
        if return_embeddings and self.use_bert:
            embeddings = np.array([self.get_bert_embeddings(text) for text in tqdm(texts, desc="BERT Embeddings")])
            return processed_texts, embeddings
        
        return processed_texts

# Model classes
class TextClassifier:
    """
    Text classification model
    """
    def __init__(self, model_type='rf', use_bert=False, use_tfidf=True):
        """
        Initialize the text classifier
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'rf' (Random Forest), 'gb' (Gradient Boosting), 
            'bert' (BERT fine-tuning), or 'ensemble'
        use_bert : bool
            Whether to use BERT embeddings
        use_tfidf : bool
            Whether to use TF-IDF features
        """
        self.model_type = model_type
        self.use_bert = use_bert and bert_available
        self.use_tfidf = use_tfidf
        
        # Text preprocessor
        self.preprocessor = TextPreprocessor(use_bert=use_bert, use_spacy=True)
        
        # TF-IDF vectorizer
        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        # Model
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'bert' and self.use_bert:
            # BERT fine-tuning model
            self.bert_model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=2
            )
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Optimizer
            self.optimizer = AdamW(self.bert_model.parameters(), lr=2e-5)
        elif model_type == 'ensemble':
            # Ensemble of models
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            if xgboost_available:
                self.xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            else:
                self.xgb_model = None
                
            if lightgbm_available:
                self.lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
            else:
                self.lgb_model = None
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.is_fitted = False
    
    def fit(self, texts, labels):
        """
        Fit the model to the training data
        
        Parameters:
        -----------
        texts : list
            List of text strings
        labels : array-like
            Labels for the texts
        """
        # Preprocess the texts
        if self.use_bert and self.model_type == 'bert':
            # BERT fine-tuning requires special preprocessing
            encoded_texts = []
            for text in tqdm(texts, desc="Encoding texts for BERT"):
                encoded = self.bert_tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                encoded_texts.append(encoded)
            
            # Train BERT model
            self.bert_model.train()
            
            for epoch in range(3):  # Simple training loop
                total_loss = 0
                for i, encoded in enumerate(tqdm(encoded_texts, desc=f"Training BERT epoch {epoch+1}")):
                    input_ids = encoded['input_ids']
                    attention_mask = encoded['attention_mask']
                    label = torch.tensor([labels[i]])
                    
                    self.optimizer.zero_grad()
                    outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=label
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    loss.backward()
                    self.optimizer.step()
                
                avg_loss = total_loss / len(encoded_texts)
                logger.info(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        else:
            # Standard ML models
            preprocessed_texts = self.preprocessor.transform(texts, method='spacy')
            
            if self.use_tfidf:
                # Transform texts using TF-IDF
                X = self.vectorizer.fit_transform(preprocessed_texts)
            
            if self.use_bert:
                # Get BERT embeddings
                _, bert_embeddings = self.preprocessor.transform(texts, return_embeddings=True)
                
                if self.use_tfidf:
                    # Combine TF-IDF and BERT
                    X_tfidf = X.toarray()
                    X = np.hstack((X_tfidf, bert_embeddings))
                else:
                    X = bert_embeddings
            
            # Train the model
            if self.model_type == 'ensemble':
                logger.info("Training ensemble models...")
                self.rf_model.fit(X, labels)
                self.gb_model.fit(X, labels)
                
                if self.xgb_model is not None:
                    self.xgb_model.fit(X, labels)
                
                if self.lgb_model is not None:
                    self.lgb_model.fit(X, labels)
            else:
                logger.info(f"Training {self.model_type} model...")
                self.model.fit(X, labels)
        
        self.is_fitted = True
        return self
    
    def predict(self, texts):
        """
        Predict labels for new texts
        
        Parameters:
        -----------
        texts : list
            List of text strings
        
        Returns:
        --------
        predictions : ndarray
            Predicted labels
        """
        if not self.is_fitted:
            logger.error("Model not fitted yet")
            raise RuntimeError("Model not fitted yet")
        
        # Preprocess the texts
        if self.use_bert and self.model_type == 'bert':
            # BERT prediction
            self.bert_model.eval()
            predictions = []
            
            for text in tqdm(texts, desc="Predicting with BERT"):
                encoded = self.bert_tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                with torch.no_grad():
                    output = self.bert_model(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask']
                    )
                    logits = output.logits
                    pred = torch.argmax(logits, dim=1).item()
                    predictions.append(pred)
            
            return np.array(predictions)
        else:
            # Standard ML models
            preprocessed_texts = self.preprocessor.transform(texts, method='spacy')
            
            if self.use_tfidf:
                # Transform texts using TF-IDF
                X = self.vectorizer.transform(preprocessed_texts)
            
            if self.use_bert:
                # Get BERT embeddings
                _, bert_embeddings = self.preprocessor.transform(texts, return_embeddings=True)
                
                if self.use_tfidf:
                    # Combine TF-IDF and BERT
                    X_tfidf = X.toarray()
                    X = np.hstack((X_tfidf, bert_embeddings))
                else:
                    X = bert_embeddings
            
            # Make predictions
            if self.model_type == 'ensemble':
                # Get predictions from all models
                rf_preds = self.rf_model.predict_proba(X)[:, 1]
                gb_preds = self.gb_model.predict_proba(X)[:, 1]
                
                all_preds = [rf_preds, gb_preds]
                weights = [1, 1]  # Equal weights
                
                if self.xgb_model is not None:
                    xgb_preds = self.xgb_model.predict_proba(X)[:, 1]
                    all_preds.append(xgb_preds)
                    weights.append(1)
                
                if self.lgb_model is not None:
                    lgb_preds = self.lgb_model.predict_proba(X)[:, 1]
                    all_preds.append(lgb_preds)
                    weights.append(1)
                
                # Weighted average
                weighted_preds = np.zeros_like(rf_preds)
                for preds, weight in zip(all_preds, weights):
                    weighted_preds += weight * preds
                
                weighted_preds /= sum(weights)
                
                # Binary predictions
                return (weighted_preds >= 0.5).astype(int)
            else:
                return self.model.predict(X)
    
    def predict_proba(self, texts):
        """
        Predict probabilities for new texts
        
        Parameters:
        -----------
        texts : list
            List of text strings
        
        Returns:
        --------
        probabilities : ndarray
            Predicted probabilities for each class
        """
        if not self.is_fitted:
            logger.error("Model not fitted yet")
            raise RuntimeError("Model not fitted yet")
        
        # Preprocess the texts
        if self.use_bert and self.model_type == 'bert':
            # BERT prediction
            self.bert_model.eval()
            probabilities = []
            
            for text in tqdm(texts, desc="Predicting probabilities with BERT"):
                encoded = self.bert_tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                with torch.no_grad():
                    output = self.bert_model(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask']
                    )
                    logits = output.logits
                    probs = torch.softmax(logits, dim=1).numpy()
                    probabilities.append(probs[0])
            
            return np.array(probabilities)
        else:
            # Standard ML models
            preprocessed_texts = self.preprocessor.transform(texts, method='spacy')
            
            if self.use_tfidf:
                # Transform texts using TF-IDF
                X = self.vectorizer.transform(preprocessed_texts)
            
            if self.use_bert:
                # Get BERT embeddings
                _, bert_embeddings = self.preprocessor.transform(texts, return_embeddings=True)
                
                if self.use_tfidf:
                    # Combine TF-IDF and BERT
                    X_tfidf = X.toarray()
                    X = np.hstack((X_tfidf, bert_embeddings))
                else:
                    X = bert_embeddings
            
            # Make predictions
            if self.model_type == 'ensemble':
                # Get predictions from all models
                rf_preds = self.rf_model.predict_proba(X)
                gb_preds = self.gb_model.predict_proba(X)
                
                all_preds = [rf_preds, gb_preds]
                weights = [1, 1]  # Equal weights
                
                if self.xgb_model is not None:
                    xgb_preds = self.xgb_model.predict_proba(X)
                    all_preds.append(xgb_preds)
                    weights.append(1)
                
                if self.lgb_model is not None:
                    lgb_preds = self.lgb_model.predict_proba(X)
                    all_preds.append(lgb_preds)
                    weights.append(1)
                
                # Weighted average
                weighted_preds = np.zeros_like(rf_preds)
                for preds, weight in zip(all_preds, weights):
                    weighted_preds += weight * preds
                
                weighted_preds /= sum(weights)
                return weighted_preds
            else:
                return self.model.predict_proba(X)
    
    def save(self, filepath):
        """
        Save the model to a file
        """
        if self.model_type == 'bert' and self.use_bert:
            # Save BERT model
            self.bert_model.save_pretrained(filepath)
            self.bert_tokenizer.save_pretrained(filepath)
        else:
            # Save sklearn model
            model_data = {
                'model_type': self.model_type,
                'use_bert': self.use_bert,
                'use_tfidf': self.use_tfidf,
                'is_fitted': self.is_fitted
            }
            
            if self.use_tfidf:
                model_data['vectorizer'] = self.vectorizer
            
            if self.model_type == 'ensemble':
                model_data['rf_model'] = self.rf_model
                model_data['gb_model'] = self.gb_model
                
                if self.xgb_model is not None:
                    model_data['xgb_model'] = self.xgb_model
                
                if self.lgb_model is not None:
                    model_data['lgb_model'] = self.lgb_model
            else:
                model_data['model'] = self.model
            
            joblib.dump(model_data, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load the model from a file
        """
        try:
            # Try loading as BERT model
            bert_model = BertForSequenceClassification.from_pretrained(filepath)
            bert_tokenizer = BertTokenizer.from_pretrained(filepath)
            
            # Create instance
            instance = cls(model_type='bert', use_bert=True)
            instance.bert_model = bert_model
            instance.bert_tokenizer = bert_tokenizer
            instance.is_fitted = True
            
            logger.info(f"Loaded BERT model from {filepath}")
            return instance
        except:
            # Try loading as sklearn model
            try:
                model_data = joblib.load(filepath)
                
                # Create instance
                instance = cls(
                    model_type=model_data['model_type'],
                    use_bert=model_data['use_bert'],
                    use_tfidf=model_data['use_tfidf']
                )
                
                if model_data['use_tfidf']:
                    instance.vectorizer = model_data['vectorizer']
                
                if model_data['model_type'] == 'ensemble':
                    instance.rf_model = model_data['rf_model']
                    instance.gb_model = model_data['gb_model']
                    
                    if 'xgb_model' in model_data:
                        instance.xgb_model = model_data['xgb_model']
                    
                    if 'lgb_model' in model_data:
                        instance.lgb_model = model_data['lgb_model']
                else:
                    instance.model = model_data['model']
                
                instance.is_fitted = model_data.get('is_fitted', False)
                
                logger.info(f"Loaded sklearn model from {filepath}")
                return instance
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

# Multimodal model for combining text, audio, and image features
class MultimodalCrisisDetector:
    """
    Multimodal model for combining text, audio, and image features
    """
    def __init__(self, use_text=True, use_audio=False, use_image=False):
        """
        Initialize the multimodal model
        
        Parameters:
        -----------
        use_text : bool
            Whether to use text features
        use_audio : bool
            Whether to use audio features
        use_image : bool
            Whether to use image features
        """
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_image = use_image
        
        # Initialize models
        if use_text:
            self.text_model = TextClassifier(model_type='ensemble', use_bert=bert_available)
        
        # For audio and image features, we'll use random forest as the base classifier
        if use_audio:
            self.audio_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.audio_scaler = StandardScaler()
        
        if use_image:
            self.image_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.image_scaler = StandardScaler()
        
        # Final meta-classifier
        self.meta_classifier = LogisticRegression(random_state=42)
        
        self.is_fitted = False
    
    def fit(self, text_data=None, audio_data=None, image_data=None, labels=None):
        """
        Fit the multimodal model
        
        Parameters:
        -----------
        text_data : list or None
            List of text strings
        audio_data : DataFrame or None
            Audio features
        image_data : DataFrame or None
            Image features
        labels : array-like
            Labels for all modalities (must be the same for all)
        """
        if labels is None:
            logger.error("Labels must be provided")
            raise ValueError("Labels must be provided")
        
        # Train individual models
        if self.use_text and text_data is not None:
            logger.info("Training text model...")
            self.text_model.fit(text_data, labels)
        
        if self.use_audio and audio_data is not None:
            logger.info("Training audio model...")
            # Scale features
            audio_features = self.audio_scaler.fit_transform(audio_data)
            self.audio_model.fit(audio_features, labels)
        
        if self.use_image and image_data is not None:
            logger.info("Training image model...")
            # Scale features
            image_features = self.image_scaler.fit_transform(image_data)
            self.image_model.fit(image_features, labels)
        
        # Get predictions from each model for meta-classifier
        meta_features = []
        
        if self.use_text and text_data is not None:
            text_probs = self.text_model.predict_proba(text_data)
            meta_features.append(text_probs)
        
        if self.use_audio and audio_data is not None:
            audio_features = self.audio_scaler.transform(audio_data)
            audio_probs = self.audio_model.predict_proba(audio_features)
            meta_features.append(audio_probs)
        
        if self.use_image and image_data is not None:
            image_features = self.image_scaler.transform(image_data)
            image_probs = self.image_model.predict_proba(image_features)
            meta_features.append(image_probs)
        
        if not meta_features:
            logger.error("No features available for training")
            raise ValueError("No features available for training")
        
        # Combine features for meta-classifier
        X_meta = np.hstack(meta_features)
        
        # Train meta-classifier
        logger.info("Training meta-classifier...")
        self.meta_classifier.fit(X_meta, labels)
        
        self.is_fitted = True
        return self
    
    def predict(self, text_data=None, audio_data=None, image_data=None):
        """
        Predict labels for new data
        
        Parameters:
        -----------
        text_data : list or None
            List of text strings
        audio_data : DataFrame or None
            Audio features
        image_data : DataFrame or None
            Image features
        
        Returns:
        --------
        predictions : ndarray
            Predicted labels
        """
        if not self.is_fitted:
            logger.error("Model not fitted yet")
            raise RuntimeError("Model not fitted yet")
        
        # Get predictions from each model
        meta_features = []
        
        if self.use_text and text_data is not None:
            text_probs = self.text_model.predict_proba(text_data)
            meta_features.append(text_probs)
        
        if self.use_audio and audio_data is not None:
            audio_features = self.audio_scaler.transform(audio_data)
            audio_probs = self.audio_model.predict_proba(audio_features)
            meta_features.append(audio_probs)
        
        if self.use_image and image_data is not None:
           image_features = self.image_scaler.transform(image_data)
        image_probs = self.image_model.predict_proba(image_features)
        meta_features.append(image_probs)
        
        if not meta_features:
            logger.error("No features available for prediction")
            raise ValueError("No features available for prediction")
        
        # Combine features for meta-classifier
        X_meta = np.hstack(meta_features)
        
        # Make predictions
        return self.meta_classifier.predict(X_meta)
    
    def predict_proba(self, text_data=None, audio_data=None, image_data=None):
        """
        Predict probabilities for new data
        
        Parameters:
        -----------
        text_data : list or None
            List of text strings
        audio_data : DataFrame or None
            Audio features
        image_data : DataFrame or None
            Image features
        
        Returns:
        --------
        probabilities : ndarray
            Predicted probabilities for each class
        """
        if not self.is_fitted:
            logger.error("Model not fitted yet")
            raise RuntimeError("Model not fitted yet")
        
        # Get predictions from each model
        meta_features = []
        
        if self.use_text and text_data is not None:
            text_probs = self.text_model.predict_proba(text_data)
            meta_features.append(text_probs)
        
        if self.use_audio and audio_data is not None:
            audio_features = self.audio_scaler.transform(audio_data)
            audio_probs = self.audio_model.predict_proba(audio_features)
            meta_features.append(audio_probs)
        
        if self.use_image and image_data is not None:
            image_features = self.image_scaler.transform(image_data)
            image_probs = self.image_model.predict_proba(image_features)
            meta_features.append(image_probs)
        
        if not meta_features:
            logger.error("No features available for prediction")
            raise ValueError("No features available for prediction")
        
        # Combine features for meta-classifier
        X_meta = np.hstack(meta_features)
        
        # Make predictions
        return self.meta_classifier.predict_proba(X_meta)
    
    def save(self, filepath):
        """
        Save the multimodal model to a file
        """
        model_data = {
            'use_text': self.use_text,
            'use_audio': self.use_audio,
            'use_image': self.use_image,
            'is_fitted': self.is_fitted,
            'meta_classifier': self.meta_classifier
        }
        
        if self.use_text:
            # Save text model separately
            text_model_path = f"{filepath}_text_model"
            self.text_model.save(text_model_path)
            model_data['text_model_path'] = text_model_path
        
        if self.use_audio:
            model_data['audio_model'] = self.audio_model
            model_data['audio_scaler'] = self.audio_scaler
        
        if self.use_image:
            model_data['image_model'] = self.image_model
            model_data['image_scaler'] = self.image_scaler
        
        joblib.dump(model_data, filepath)
        logger.info(f"Multimodal model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load the multimodal model from a file
        """
        try:
            model_data = joblib.load(filepath)
            
            # Create instance
            instance = cls(
                use_text=model_data['use_text'],
                use_audio=model_data['use_audio'],
                use_image=model_data['use_image']
            )
            
            # Load text model if available
            if instance.use_text and 'text_model_path' in model_data:
                instance.text_model = TextClassifier.load(model_data['text_model_path'])
            
            # Load audio model if available
            if instance.use_audio:
                instance.audio_model = model_data['audio_model']
                instance.audio_scaler = model_data['audio_scaler']
            
            # Load image model if available
            if instance.use_image:
                instance.image_model = model_data['image_model']
                instance.image_scaler = model_data['image_scaler']
            
            # Load meta-classifier
            instance.meta_classifier = model_data['meta_classifier']
            
            instance.is_fitted = model_data.get('is_fitted', False)
            
            logger.info(f"Loaded multimodal model from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

# Evaluation utilities
def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities for the positive class
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_proba)
    
    return metrics

def plot_evaluation_metrics(metrics, title="Model Evaluation"):
    """
    Plot evaluation metrics
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    title : str
        Title for the plot
    """
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 0])
    axs[0, 0].set_title('Confusion Matrix')
    axs[0, 0].set_xlabel('Predicted')
    axs[0, 0].set_ylabel('True')
    
    # Plot metrics
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_values = [metrics[name] for name in metric_names]
    
    axs[0, 1].bar(metric_names, metric_values, color='skyblue')
    axs[0, 1].set_title('Performance Metrics')
    axs[0, 1].set_ylim(0, 1)
    
    # Plot ROC curve if available
    if 'roc_auc' in metrics:
        if 'fpr' in metrics and 'tpr' in metrics:
            fpr, tpr = metrics['fpr'], metrics['tpr']
        else:
            # Generate dummy data if not available
            fpr = [0, 0.5, 1]
            tpr = [0, 0.5, 1]
        
        axs[1, 0].plot(fpr, tpr, 'b-', label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
        axs[1, 0].plot([0, 1], [0, 1], 'r--')
        axs[1, 0].set_xlim([0, 1])
        axs[1, 0].set_ylim([0, 1])
        axs[1, 0].set_title('ROC Curve')
        axs[1, 0].set_xlabel('False Positive Rate')
        axs[1, 0].set_ylabel('True Positive Rate')
        axs[1, 0].legend(loc='lower right')
    else:
        axs[1, 0].text(0.5, 0.5, 'ROC Curve not available', 
                     horizontalalignment='center', verticalalignment='center')
    
    # Plot Precision-Recall curve if available
    if 'average_precision' in metrics:
        if 'precision_curve' in metrics and 'recall_curve' in metrics:
            precision, recall = metrics['precision_curve'], metrics['recall_curve']
        else:
            # Generate dummy data if not available
            precision = [1, 0.5, 0]
            recall = [0, 0.5, 1]
        
        axs[1, 1].plot(recall, precision, 'g-', 
                     label=f'PR (AP = {metrics["average_precision"]:.3f})')
        axs[1, 1].set_xlim([0, 1])
        axs[1, 1].set_ylim([0, 1])
        axs[1, 1].set_title('Precision-Recall Curve')
        axs[1, 1].set_xlabel('Recall')
        axs[1, 1].set_ylabel('Precision')
        axs[1, 1].legend(loc='lower left')
    else:
        axs[1, 1].text(0.5, 0.5, 'Precision-Recall Curve not available', 
                     horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

# Main training and evaluation functions
def train_and_evaluate_text_model(data_dir, model_type='ensemble', use_bert=False):
    """
    Train and evaluate a text classification model
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data
    model_type : str
        Type of model: 'rf', 'gb', 'bert', or 'ensemble'
    use_bert : bool
        Whether to use BERT embeddings
    
    Returns:
    --------
    model : TextClassifier
        Trained text classification model
    metrics : dict
        Evaluation metrics
    """
    logger.info(f"Training {model_type} text model (use_bert={use_bert})...")
    
    # Load data
    try:
        df = pd.read_csv(f"{data_dir}/text_features.csv")
        val_df = pd.read_csv(f"{data_dir}/text_features_validation.csv")
        
        # Extract texts and labels
        texts = df['text'].tolist()
        labels = df['label'].values
        
        val_texts = val_df['text'].tolist()
        val_labels = val_df['label'].values
    except FileNotFoundError:
        logger.warning("Text feature files not found, generating synthetic data...")
        df = create_synthetic_text_data(data_dir)
        val_df = pd.read_csv(f"{data_dir}/text_features_validation.csv")
        
        # Extract texts and labels
        texts = df['text'].tolist()
        labels = df['label'].values
        
        val_texts = val_df['text'].tolist()
        val_labels = val_df['label'].values
    
    # Create and train model
    model = TextClassifier(model_type=model_type, use_bert=use_bert)
    model.fit(texts, labels)
    
    # Evaluate model
    val_preds = model.predict(val_texts)
    val_proba = model.predict_proba(val_texts)[:, 1] if val_preds.ndim == 1 else val_proba[:, 1]
    
    metrics = evaluate_model(val_labels, val_preds, val_proba)
    
    # Save model
    model.save(f"{data_dir}/text_model_{model_type}_bert{int(use_bert)}.joblib")
    
    return model, metrics

def train_and_evaluate_multimodal_model(data_dir, use_text=True, use_audio=False, use_image=False):
    """
    Train and evaluate a multimodal model
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data
    use_text : bool
        Whether to use text features
    use_audio : bool
        Whether to use audio features
    use_image : bool
        Whether to use image features
    
    Returns:
    --------
    model : MultimodalCrisisDetector
        Trained multimodal model
    metrics : dict
        Evaluation metrics
    """
    logger.info(f"Training multimodal model (text={use_text}, audio={use_audio}, image={use_image})...")
    
    # Load data
    text_data = None
    audio_data = None
    image_data = None
    
    val_text_data = None
    val_audio_data = None
    val_image_data = None
    
    labels = None
    val_labels = None
    
    if use_text:
        try:
            df = pd.read_csv(f"{data_dir}/text_features.csv")
            val_df = pd.read_csv(f"{data_dir}/text_features_validation.csv")
            
            text_data = df['text'].tolist()
            val_text_data = val_df['text'].tolist()
            
            if labels is None:
                labels = df['label'].values
                val_labels = val_df['label'].values
        except FileNotFoundError:
            logger.warning("Text feature files not found, generating synthetic data...")
            df = create_synthetic_text_data(data_dir)
            val_df = pd.read_csv(f"{data_dir}/text_features_validation.csv")
            
            text_data = df['text'].tolist()
            val_text_data = val_df['text'].tolist()
            
            if labels is None:
                labels = df['label'].values
                val_labels = val_df['label'].values
    
    if use_audio:
        try:
            df = pd.read_csv(f"{data_dir}/audio_features.csv")
            val_df = pd.read_csv(f"{data_dir}/audio_features_validation.csv")
            
            # Drop label column for features
            audio_data = df.drop('label', axis=1) if 'label' in df.columns else df
            val_audio_data = val_df.drop('label', axis=1) if 'label' in val_df.columns else val_df
            
            if labels is None:
                labels = df['label'].values
                val_labels = val_df['label'].values
        except FileNotFoundError:
            logger.warning("Audio feature files not found, generating synthetic data...")
            df = create_synthetic_audio_features(data_dir)
            val_df = pd.read_csv(f"{data_dir}/audio_features_validation.csv")
            
            audio_data = df.drop('label', axis=1) if 'label' in df.columns else df
            val_audio_data = val_df.drop('label', axis=1) if 'label' in val_df.columns else val_df
            
            if labels is None:
                labels = df['label'].values
                val_labels = val_df['label'].values
    
    if use_image:
        try:
            df = pd.read_csv(f"{data_dir}/image_features.csv")
            val_df = pd.read_csv(f"{data_dir}/image_features_validation.csv")
            
            # Drop label column for features
            image_data = df.drop('label', axis=1) if 'label' in df.columns else df
            val_image_data = val_df.drop('label', axis=1) if 'label' in val_df.columns else val_df
            
            if labels is None:
                labels = df['label'].values
                val_labels = val_df['label'].values
        except FileNotFoundError:
            logger.warning("Image feature files not found, generating synthetic data...")
            df = create_synthetic_image_features(data_dir)
            val_df = pd.read_csv(f"{data_dir}/image_features_validation.csv")
            
            image_data = df.drop('label', axis=1) if 'label' in df.columns else df
            val_image_data = val_df.drop('label', axis=1) if 'label' in val_df.columns else val_df
            
            if labels is None:
                labels = df['label'].values
                val_labels = val_df['label'].values
    
    # Create and train model
    model = MultimodalCrisisDetector(use_text=use_text, use_audio=use_audio, use_image=use_image)
    model.fit(text_data=text_data, audio_data=audio_data, image_data=image_data, labels=labels)
    
    # Evaluate model
    val_preds = model.predict(
        text_data=val_text_data, 
        audio_data=val_audio_data, 
        image_data=val_image_data
    )
    
    val_proba = model.predict_proba(
        text_data=val_text_data, 
        audio_data=val_audio_data, 
        image_data=val_image_data
    )[:, 1]
    
    metrics = evaluate_model(val_labels, val_preds, val_proba)
    
    # Save model
    model.save(f"{data_dir}/multimodal_model_t{int(use_text)}_a{int(use_audio)}_i{int(use_image)}.joblib")
    
    return model, metrics

# Main function
def main():
    """
    Main function to run the crisis detection pipeline
    """
    parser = argparse.ArgumentParser(description="Crisis Detection System")
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory for storing data')
    parser.add_argument('--generate_data', action='store_true',
                        help='Generate synthetic data')
    parser.add_argument('--model_type', type=str, default='ensemble',
                        choices=['rf', 'gb', 'bert', 'ensemble'],
                        help='Type of text model to train')
    parser.add_argument('--use_bert', action='store_true',
                        help='Use BERT embeddings')
    parser.add_argument('--use_text', action='store_true',
                        help='Use text features')
    parser.add_argument('--use_audio', action='store_true',
                        help='Use audio features')
    parser.add_argument('--use_image', action='store_true',
                        help='Use image features')
    parser.add_argument('--multimodal', action='store_true',
                        help='Train multimodal model')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.data_dir}/crisis_detection.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Crisis Detection System")
    logger.info(f"Using BERT: {args.use_bert}")
    
    # Generate synthetic data if requested
    if args.generate_data:
        logger.info("Generating synthetic data...")
        create_synthetic_text_data(args.data_dir)
        
        if args.use_audio:
            create_synthetic_audio_features(args.data_dir)
        
        if args.use_image:
            create_synthetic_image_features(args.data_dir)
    
    # Train models
    if args.multimodal:
        model, metrics = train_and_evaluate_multimodal_model(
            args.data_dir,
            use_text=args.use_text,
            use_audio=args.use_audio,
            use_image=args.use_image
        )
        
        logger.info("Multimodal model evaluation metrics:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"{metric}: {value:.4f}")
        
        # Plot evaluation metrics
        fig = plot_evaluation_metrics(metrics, "Multimodal Model Evaluation")
        fig.savefig(f"{args.data_dir}/multimodal_evaluation.png")
    else:
        model, metrics = train_and_evaluate_text_model(
            args.data_dir,
            model_type=args.model_type,
            use_bert=args.use_bert
        )
        
        logger.info("Text model evaluation metrics:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"{metric}: {value:.4f}")
        
        # Plot evaluation metrics
        fig = plot_evaluation_metrics(metrics, f"Text Model ({args.model_type}) Evaluation")
        fig.savefig(f"{args.data_dir}/text_model_evaluation.png")
    
    logger.info("Crisis Detection System completed successfully.")

if __name__ == "__main__":
    main()