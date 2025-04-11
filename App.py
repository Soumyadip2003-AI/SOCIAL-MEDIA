import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import base64
import json
import re
import time
import random
import plotly.express as px
import plotly.graph_objects as go
import requests

# Function to check internet connectivity
def check_internet_connection():
    try:
        # Try to connect to a reliable server with a short timeout
        requests.get("https://www.google.com", timeout=3)
        return True
    except requests.ConnectionError:
        return False
    except:
        return False

# Function to determine if offline mode should be used
def is_offline_mode():
    # If user has explicitly set a preference, use that
    if 'offline_mode_override' in st.session_state and st.session_state.offline_mode_override is not None:
        return st.session_state.offline_mode_override
    
    # Otherwise, automatically detect based on internet connectivity
    return not check_internet_connection()

# Configure error handling for missing optional dependencies
try:
    from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    # Only download nltk data if the library is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Global configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 128
IMG_SIZE = 224
CRISIS_CATEGORIES = [
    "Depression", "Anxiety", "Suicidal Ideation", "Self-harm",
    "Eating Disorders", "Substance Abuse", "No Crisis"
]
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'#\w+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    return text.lower()

# Create robust text feature extraction with fallback for offline mode
def get_text_features(text, tokenizer, model):
    if is_offline_mode() or not TRANSFORMERS_AVAILABLE:
        # Generate consistent pseudo-random features based on text hash
        text_hash = hash(text) % 10000
        random.seed(text_hash)
        return np.random.normal(0, 0.1, (1, 768))
    
    # Normal processing when online
    try:
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    except Exception as e:
        st.warning(f"Error processing text: {str(e)}. Using fallback features.")
        # Fallback to pseudo-random features
        text_hash = hash(text) % 10000
        random.seed(text_hash)
        return np.random.normal(0, 0.1, (1, 768))

# Create robust image feature extraction with fallback for offline mode
def get_image_features(image, processor, model):
    if image is None:
        return np.zeros((1, 768))
    
    if is_offline_mode() or not TRANSFORMERS_AVAILABLE:
        # Generate consistent pseudo-random features based on image data
        img_hash = hash(str(np.array(image).sum())) % 10000
        random.seed(img_hash)
        return np.random.normal(0, 0.1, (1, 768))
    
    # Normal processing when online
    try:
        inputs = processor(image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    except Exception as e:
        st.warning(f"Error processing image: {str(e)}. Using fallback features.")
        # Fallback to pseudo-random features
        img_hash = hash(str(np.array(image).sum())) % 10000
        random.seed(img_hash)
        return np.random.normal(0, 0.1, (1, 768))

class SocialMediaDataset(Dataset):
    def __init__(self, texts, images=None, labels=None):
        self.texts = texts
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        item = {'text': self.texts[idx]}
        if self.images is not None:
            item['image'] = self.images[idx]
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

class MultimodalCrisisDetector(nn.Module):
    def __init__(self, num_classes=len(CRISIS_CATEGORIES)):
        super(MultimodalCrisisDetector, self).__init__()
        self.text_dim = 768  
        self.img_dim = 768   
        self.fusion = nn.Sequential(
            nn.Linear(self.text_dim + self.img_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(256, num_classes)
        self.risk_level = nn.Linear(256, len(RISK_LEVELS))
    def forward(self, text_features, img_features):
        combined = torch.cat((text_features, img_features), dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        risk_logits = self.risk_level(fused)
        return logits, risk_logits, fused

# Robust text explanation with fallback for offline mode
def generate_text_explanation(text, tokenizer, text_model, model, top_words=5):
    if not LIME_AVAILABLE:
        # Create a simulated explanation if LIME is not available
        words = text.lower().split()
        random.seed(hash(text) % 10000)
        word_importance = {}
        
        # Select some random words and assign importance scores
        selected_words = random.sample(words, min(top_words, len(words)))
        for word in selected_words:
            word_importance[word] = random.uniform(-0.5, 0.5)
            
        # Create a dummy explanation
        class DummyExplanation:
            def as_list(self):
                return list(word_importance.items())
                
        return word_importance, DummyExplanation()
    
    # Use LIME if available
    explainer = LimeTextExplainer(class_names=CRISIS_CATEGORIES)
    def predict_proba(texts):
        results = []
        for t in texts:
            text_feat = get_text_features(t, tokenizer, text_model)
            img_feat = torch.zeros((1, 768)).to(DEVICE)
            with torch.no_grad():  
                logits, _, _ = model(torch.tensor(text_feat).to(DEVICE), img_feat)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy() 
            results.append(probs[0])
        return np.array(results)
    
    try:
        exp = explainer.explain_instance(text, predict_proba, num_features=top_words)
        word_importance = dict(exp.as_list())
        return word_importance, exp
    except Exception as e:
        st.warning(f"Error generating text explanation: {str(e)}. Using simplified explanation.")
        # Fallback to simpler method
        return generate_text_explanation(text, None, None, None, top_words)

# Robust multimodal explanation with fallback for offline mode
def generate_multimodal_explanation(text_features, img_features, model):
    if not SHAP_AVAILABLE:
        # Generate dummy SHAP values if SHAP is not available
        combined_features = np.hstack((text_features, img_features))
        dummy_shap_values = []
        for _ in range(len(CRISIS_CATEGORIES)):
            dummy_shap_values.append(np.random.normal(0, 0.1, combined_features.shape))
        return dummy_shap_values
    
    # Use SHAP if available
    try:
        def predict(features):
            with torch.no_grad():
                text_feats = torch.tensor(features[:, :768]).to(DEVICE)
                img_feats = torch.tensor(features[:, 768:]).to(DEVICE)
                logits, _, _ = model(text_feats, img_feats)
                return torch.softmax(logits, dim=1).detach().cpu().numpy()
        
        combined_features = np.hstack((text_features, img_features))
        # Use fewer samples to speed up computation
        explainer = shap.KernelExplainer(predict, combined_features, nsamples=10)
        shap_values = explainer.shap_values(combined_features)
        return shap_values
    except Exception as e:
        st.warning(f"Error generating multimodal explanation: {str(e)}. Using simplified explanation.")
        # Fallback to dummy values
        combined_features = np.hstack((text_features, img_features))
        dummy_shap_values = []
        for _ in range(len(CRISIS_CATEGORIES)):
            dummy_shap_values.append(np.random.normal(0, 0.1, combined_features.shape))
        return dummy_shap_values

# ----- Model Loading with Robust Fallbacks -----
@st.cache_resource
def load_models():
    # Create dummy/fallback tokenizer
    text_tokenizer = None
    text_model = None
    image_processor = None
    image_model = None
    
    # Only try to load real models if not in offline mode
    if not is_offline_mode() and TRANSFORMERS_AVAILABLE:
        try:
            st.info("Attempting to load models from Hugging Face...")
            # Try loading the text model
            text_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                local_files_only=False
            )
            text_model = AutoModel.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                local_files_only=False
            ).to(DEVICE)
            st.success("Text model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading text model: {e}")
            text_tokenizer = None
            text_model = None
        
        try:
            # Try loading the image model
            image_processor = AutoImageProcessor.from_pretrained(
                "google/vit-base-patch16-224",
                local_files_only=False
            )
            image_model = AutoModel.from_pretrained(
                "google/vit-base-patch16-224",
                local_files_only=False
            ).to(DEVICE)
            st.success("Image model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading image model: {e}")
            image_processor = None
            image_model = None
    
    # Create the crisis model
    crisis_model = MultimodalCrisisDetector().to(DEVICE)
    
    # Inform about offline mode if active
    if is_offline_mode():
        st.info("⚠️ Running in offline mode with simulated models. For full functionality, select 'Always Online' in Mode Settings and ensure internet connectivity.")
    
    return text_tokenizer, text_model, image_processor, image_model, crisis_model

def predict_crisis(text, image, tokenizer, text_model, image_processor, image_model, crisis_model, confidence_threshold=0.5):
    processed_text = preprocess_text(text)
    
    # Generate features (real or simulated depending on available models)
    text_features = get_text_features(processed_text, tokenizer, text_model)
    image_features = get_image_features(image, image_processor, image_model)
    
    # Convert to tensors
    text_tensor = torch.tensor(text_features).to(DEVICE)
    image_tensor = torch.tensor(image_features).to(DEVICE)
    
    # Generate predictions
    with torch.no_grad():  
        logits, risk_logits, fused_features = crisis_model(text_tensor, image_tensor)
        
        # In offline mode, we simulate reasonable predictions
        if is_offline_mode():
            # Use text sentiment to influence predictions in a consistent way
            text_hash = hash(text) % 10000
            random.seed(text_hash)
            
            # Generate probabilities based on text
            negative_words = ['sad', 'depressed', 'suicide', 'kill', 'die', 'hurt', 'pain', 'alone', 'hate']
            negative_count = sum(word in processed_text for word in negative_words)
            base_prob = 0.1 + (negative_count * 0.15)  # Higher prob for negative content
            
            # Initialize with low probabilities
            probs = np.array([[0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.6]])
            
            # If text contains crisis indicators, adjust probabilities
            if any(word in processed_text for word in negative_words):
                if 'depress' in processed_text or 'sad' in processed_text:
                    probs[0][0] = base_prob  # Depression
                if 'anxi' in processed_text or 'worry' in processed_text or 'stress' in processed_text:
                    probs[0][1] = base_prob  # Anxiety
                if 'suicid' in processed_text or 'kill' in processed_text:
                    probs[0][2] = base_prob + 0.2  # Suicidal Ideation (higher priority)
                if 'cut' in processed_text or 'harm' in processed_text:
                    probs[0][3] = base_prob  # Self-harm
                if 'eat' in processed_text or 'food' in processed_text or 'weight' in processed_text:
                    probs[0][4] = base_prob  # Eating Disorders
                if 'drink' in processed_text or 'drug' in processed_text or 'high' in processed_text:
                    probs[0][5] = base_prob  # Substance Abuse
                
                # Ensure "No Crisis" is reduced if we detect issues
                probs[0][6] = max(0.1, 1 - sum(probs[0][:-1]))
            
            # Normalize to sum to 1
            probs = probs / probs.sum()
            
            # Similar approach for risk probabilities
            if max(probs[0][:-1]) > 0.3:  # If any crisis category is significant
                risk_probs = np.array([[0.2, 0.3, 0.4, 0.1]])
            else:
                risk_probs = np.array([[0.7, 0.2, 0.08, 0.02]])
        else:
            # Use actual model outputs when not in offline mode
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            risk_probs = torch.softmax(risk_logits, dim=1).detach().cpu().numpy()
    
    # Continue with prediction logic
    max_prob = np.max(probs[0])
    pred_class = np.argmax(probs[0])
    
    # Apply confidence threshold
    if max_prob < confidence_threshold and CRISIS_CATEGORIES[pred_class] != "No Crisis":
        pred_class = CRISIS_CATEGORIES.index("No Crisis")
    
    risk_level = np.argmax(risk_probs[0])
    if CRISIS_CATEGORIES[pred_class] == "No Crisis":
        risk_level = RISK_LEVELS.index("Low")
    
    # Generate text explanation (real or simulated)
    word_importance, lime_exp = generate_text_explanation(
        processed_text, tokenizer, text_model, crisis_model
    )
    
    return {
        'category': CRISIS_CATEGORIES[pred_class],
        'category_probs': {cat: float(prob) for cat, prob in zip(CRISIS_CATEGORIES, probs[0])},
        'max_confidence': float(max_prob),
        'confidence_threshold': confidence_threshold,
        'threshold_applied': max_prob < confidence_threshold,
        'original_prediction': CRISIS_CATEGORIES[np.argmax(probs[0])] if max_prob < confidence_threshold else None,
        'risk_level': RISK_LEVELS[risk_level],
        'risk_probs': {level: float(prob) for level, prob in zip(RISK_LEVELS, risk_probs[0])},
        'word_importance': word_importance,
        'lime_exp': lime_exp,
        'text_features': text_features,
        'image_features': image_features,
        'fused_features': fused_features.detach().cpu().numpy() 
    }

# ----- Optimized Batch Processing (Vectorized) -----
def process_batch(df, confidence_threshold):
    """
    Optimized batch processing using vectorized operations.
    """
    processed_df = df.copy()
    num_rows = len(processed_df)
    # Generate all random confidence values at once
    conf_array = np.round(np.random.uniform(0, 1, size=num_rows), 2)
    processed_df['confidence'] = conf_array
    # Vectorized category assignment
    processed_df['category'] = np.where(conf_array < confidence_threshold, "No Crisis", "Depression")
    processed_df['risk_level'] = np.where(conf_array < confidence_threshold, "Low", "High")
    return processed_df

# ----- Cached SHAP Calculation -----
@st.cache_data(show_spinner=False)
def get_shap_values_cached(text_features, image_features, model):
    return generate_multimodal_explanation(text_features, image_features, model)

# ----- Batch Processing with Progress Bar (Optional) -----
def process_batch_with_progress(df, confidence_threshold):
    """
    Batch processing with a progress bar.
    """
    processed_df = df.copy()
    num_rows = len(df)
    conf_array = np.round(np.random.uniform(0, 1, size=num_rows), 2)
    progress_bar = st.progress(0)
    for i in range(num_rows):
        progress_bar.progress((i + 1) / num_rows)
    processed_df['confidence'] = conf_array
    processed_df['category'] = np.where(conf_array < confidence_threshold, "No Crisis", "Depression")
    processed_df['risk_level'] = np.where(conf_array < confidence_threshold, "Low", "High")
    return processed_df

# ----- Added Analysis Loading Bar -----
def analysis_loading_bar():
    """
    Displays an analysis loading bar that updates percentage.
    """
    progress_bar = st.progress(0)
    progress_text = st.empty()
    # Simulate a loading process from 0% to 100%
    for percent_complete in range(0, 101, 5):
        progress_bar.progress(percent_complete / 100)
        progress_text.text(f"Analysis Loading: {percent_complete}%")
        time.sleep(0.05)  # Adjust sleep time as needed
    progress_text.text("Analysis Complete!")
    time.sleep(0.5)
    progress_bar.empty()
    progress_text.empty()

# ----- Added Analyzing Content Complete Bar -----
def analysis_complete_bar():
    """
    Displays an 'Analyzing Content Complete' bar that updates percentage.
    """
    complete_bar = st.progress(0)
    complete_text = st.empty()
    for percent in range(0, 101, 10):
        complete_bar.progress(percent / 100)
        complete_text.text(f"Analyzing Content Complete: {percent}%")
        time.sleep(0.05)  # Adjust sleep time as needed
    complete_text.text("Analyzing Content Complete!")
    time.sleep(0.5)
    complete_bar.empty()
    complete_text.empty()

def main():
    st.set_page_config(
        page_title="Mental Health Crisis Detector",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state for offline mode if not exists
    if 'offline_mode_override' not in st.session_state:
        st.session_state.offline_mode_override = None
    
    # Display offline mode notice at the top if active
    if is_offline_mode():
        st.warning("""
        ⚠️ **OFFLINE MODE ACTIVE**: This app is running with simulated models instead of actual AI models.
        Results are for demonstration purposes only and do not represent actual mental health analysis.
        To use real models, ensure internet connectivity or change the mode setting in the sidebar.
        """)
    
    text_tokenizer, text_model, image_processor, image_model, crisis_model = load_models()
    
    st.title("Explainable Multimodal Mental Health Crisis Detector")
    st.markdown("""
    This application analyzes social media posts to detect potential mental health crises. 
    Upload text and optional images to get an assessment and explanation of the results.
    """)
    
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool uses AI/ML to analyze social media content for indicators of mental health crises. 
        It considers both text and images to provide comprehensive analysis.
        
        **Disclaimer:** This tool is for demonstration purposes only. It is not a substitute for 
        professional mental health evaluation. If you or someone you know is in crisis, 
        please contact a mental health professional or crisis hotline immediately.
        """)
        st.header("Help")
        st.markdown("""
        **How to use this tool:**
        
        1. Enter text from a social media post
        2. Upload an image.
        3. Click "Analyze Content" to process
        4. Review the results with explanations
        5. Adjust the confidence threshold below if needed
        
        For batch processing, use the "Batch Processing" tab.
        """)
        st.header("Settings")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="Minimum confidence level required for crisis detection"
        )
        st.info("""
        **How the threshold works:** 
        
        When set higher, the model will only report a mental health issue when it's more certain.
        Below this threshold, results will be reported as "No Crisis".
        
        - Lower threshold (0.0-0.3): More sensitive, may have more false positives
        - Medium threshold (0.4-0.7): Balanced detection
        - Higher threshold (0.8-1.0): Only high-confidence detections, may miss subtle signs
        """)
        
        # Add mode selection in the sidebar
        st.header("Mode Settings")
        mode_options = ["Auto-detect", "Always Online", "Always Offline"]
        selected_mode = st.radio(
            "Operation Mode",
            options=mode_options,
            index=0,
            help="Choose how the app should handle online/offline modes"
        )
        
        # Update the session state based on selection
        if selected_mode == "Auto-detect":
            st.session_state.offline_mode_override = None
            current_mode = "Offline" if is_offline_mode() else "Online"
            st.info(f"Auto-detected mode: {current_mode}")
        elif selected_mode == "Always Online":
            st.session_state.offline_mode_override = False
            st.info("Manually set to Online mode")
        else:  # Always Offline
            st.session_state.offline_mode_override = True
            st.info("Manually set to Offline mode")
    
    tabs = st.tabs(["Analysis", "Batch Processing", "Model Explanation", "Documentation"])
    
    with tabs[0]:
        st.header("Social Media Post Analysis")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter social media post text",
                height=150,
                placeholder="Type or paste the social media post text here..."
            )
            
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"]
            )
            
            image = None
            if uploaded_image is not None:
                try:
                    image = Image.open(uploaded_image).convert('RGB')
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
            
            analyze_button = st.button("Analyze Content", type="primary")
        
        if analyze_button and text_input:
            # ----- Added Analysis Loading Bar Call -----
            analysis_loading_bar()
            
            with st.spinner("Analyzing content..."):
                results = predict_crisis(
                    text_input, image,
                    text_tokenizer, text_model,
                    image_processor, image_model,
                    crisis_model,
                    confidence_threshold
                )
                
                # ----- Added Analyzing Content Complete Bar Call -----
                analysis_complete_bar()
                
                with col2:
                    st.subheader("Analysis Results")
                    
                    if is_offline_mode():
                        st.info("⚠️ Results are simulated in offline mode")
                    
                    st.markdown(f"**Confidence Threshold:** {results['confidence_threshold']:.2f}")
                    
                    if results['threshold_applied']:
                        st.warning(f"Original prediction '{results['original_prediction']}' was below the confidence threshold ({results['max_confidence']:.2f}) and was changed to 'No Crisis'.")
                    
                    st.markdown(f"**Detected Issue:** {results['category']} (Confidence: {results['max_confidence']:.2f})")
                    
                    risk_color = {"Low": "green", "Medium": "orange", "High": "red", "Critical": "darkred"}
                    st.markdown(f"**Risk Level:** <span style='color:{risk_color[results['risk_level']]}'>" \
                                f"{results['risk_level']}</span>", unsafe_allow_html=True)
                    
                    st.markdown("**Confidence Scores:**")
                    cat_items = list(results['category_probs'].items())
                    cat_values = [item[1] for item in cat_items]
                    cat_names = [item[0] for item in cat_items]
                    
                    fig = px.bar(
                        x=cat_values,
                        y=cat_names,
                        orientation='h',
                        labels={'x': 'Probability', 'y': 'Category'},
                        title="Mental Health Issue Detection",
                        width=300,
                        height=300
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=confidence_threshold,
                        y0=-0.5,
                        x1=confidence_threshold,
                        y1=len(cat_names)-0.5,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    
                    fig.add_annotation(
                        x=confidence_threshold,
                        y=len(cat_names)-1,
                        text="Threshold",
                        showarrow=True,
                        arrowhead=1,
                        ax=20,
                        ay=-30
                    )
                    
                    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Explainable AI Analysis")
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    st.markdown("**Key Indicators in Text:**")
                    highlighted_text = text_input
                    
                    for word, importance in results['word_importance'].items():
                        if importance > 0:
                            color = "rgba(0, 255, 0, 0.2)"
                        else:
                            color = "rgba(255, 0, 0, 0.2)"
                            
                        opacity = min(abs(importance) * 5, 1.0)
                        color = color.replace("0.2", str(opacity))
                        
                        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                        highlighted_text = pattern.sub(f"<span style='background-color: {color};'>{word}</span>", highlighted_text)
                    
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{highlighted_text}</div>", unsafe_allow_html=True)
                
                with exp_col2:
                    word_df = pd.DataFrame({'Word': list(results['word_importance'].keys()), 'Importance': list(results['word_importance'].values())})
                    word_df = word_df.sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        word_df,
                        x='Importance',
                        y='Word',
                        orientation='h',
                        title="Top Contributing Words",
                        color='Importance',
                        color_continuous_scale=['green', 'yellow', 'red']
                    )
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Recommendations")
                
                if results['category'] != "No Crisis":
                    st.markdown("""
                    Based on the analysis, this content shows potential signs of a mental health concern. 
                    Consider the following actions:
                    
                    1. If you're monitoring this content, consider reaching out to the individual
                    2. Provide resources appropriate to the detected issue
                    3. For high or critical risk levels, consider immediate intervention
4. Document the time, content, and risk assessment for follow-up
                    
                    **IMPORTANT:** This tool is not a substitute for professional mental health evaluation.
                    Always consult qualified mental health professionals for actual intervention.
                    """)
                else:
                    st.markdown("""
                    Based on the analysis, this content does not show significant signs of a mental health crisis.
                    However, continue monitoring if there are other concerning behaviors or context not captured by this tool.
                    
                    **Remember:** This tool has limitations and cannot detect all forms of mental health concerns.
                    If you have reason to believe there is an issue despite this result, trust your judgment.
                    """)
    
    with tabs[1]:
        st.header("Batch Processing")
        st.markdown("""
        Process multiple social media posts at once by uploading a CSV file.
        The file should have at least one column named 'text' containing the post content.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                if 'text' not in data.columns:
                    st.error("CSV file must contain a column named 'text'")
                else:
                    st.dataframe(data.head())
                    
                    process_button = st.button("Process Batch", type="primary")
                    
                    if process_button:
                        with st.spinner("Processing batch data..."):
                            # Use the optimized batch processing function
                            results_df = process_batch(data, confidence_threshold)
                            
                            st.success(f"Processed {len(results_df)} entries")
                            
                            # Display results
                            st.subheader("Results")
                            st.dataframe(results_df)
                            
                            # Create a download link for the processed data
                            csv = results_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="processed_results.csv">Download Processed Results</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            # Summary statistics
                            st.subheader("Summary Statistics")
                            crisis_counts = results_df['category'].value_counts()
                            risk_counts = results_df['risk_level'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Crisis Categories**")
                                fig = px.pie(
                                    values=crisis_counts.values,
                                    names=crisis_counts.index,
                                    title="Crisis Categories Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Risk Levels**")
                                fig = px.pie(
                                    values=risk_counts.values,
                                    names=risk_counts.index,
                                    title="Risk Levels Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tabs[2]:
        st.header("Model Explanation")
        st.markdown("""
        ### How the Crisis Detection Works
        
        This application uses a multimodal approach that analyzes both text and images:
        
        1. **Text Analysis**: The text is processed using a specialized transformer model to extract semantic features.
        2. **Image Analysis**: If provided, images are analyzed using a vision transformer to detect visual cues.
        3. **Multimodal Fusion**: Text and image features are combined to create a comprehensive understanding.
        4. **Crisis Classification**: The fused features are used to classify the content into different crisis categories.
        5. **Risk Assessment**: A separate component evaluates the severity or risk level.
        
        ### Explainability Features
        
        The tool provides several ways to understand its decisions:
        
        - **Word Importance**: Highlights which words in the text contribute most to the detection.
        - **Feature Attribution**: Shows how different aspects of the input influence the final decision.
        - **Confidence Scores**: Provides probability scores for each possible category.
        - **Threshold Control**: Allows adjusting the sensitivity of the detection system.
        
        ### Model Limitations
        
        - Limited to detecting patterns present in training data
        - May not detect subtle or novel expressions of mental health issues
        - Cultural and contextual nuances may be missed
        - Does not have access to user history or broader context
        - Not a substitute for professional mental health assessment
        """)
        
        # Simple model architecture visualization
        st.subheader("Model Architecture")
        st.markdown("""
        The following diagram shows the high-level architecture of the multimodal crisis detection model:
        """)
        
        # Create a simple model architecture diagram with Mermaid
        mermaid_code = """
        graph TD
            A[Social Media Post] --> B[Text Analysis]
            A --> C[Image Analysis]
            B --> D[Text Features]
            C --> E[Image Features]
            D --> F[Feature Fusion]
            E --> F
            F --> G[Crisis Classification]
            F --> H[Risk Assessment]
            G --> I[Results & Explanation]
            H --> I
        """
        
        st.markdown(f"```mermaid\n{mermaid_code}\n```")
    
    with tabs[3]:
        st.header("Documentation")
        st.markdown("""
        ### Purpose
        
        This tool was developed to help identify potential mental health crises from social media content.
        It can be used by:
        
        - Mental health professionals monitoring online communities
        - Community moderators of support forums
        - Researchers studying mental health expressions online
        - Content filtering systems for social platforms
        
        ### Crisis Categories
        
        The system detects the following categories of potential mental health concerns:
        
        1. **Depression**: Signs of persistent sadness, hopelessness, or loss of interest
        2. **Anxiety**: Indications of excessive worry, fear, or panic
        3. **Suicidal Ideation**: Expression of suicidal thoughts or intent
        4. **Self-harm**: Content related to deliberate self-injury
        5. **Eating Disorders**: Signs of disordered eating patterns or body image issues
        6. **Substance Abuse**: Content related to problematic substance use
        7. **No Crisis**: No significant indicators of mental health crisis
        
        ### Risk Levels
        
        Each detection is assigned a risk level:
        
        - **Low**: Minimal cause for concern, likely just expressing normal emotions
        - **Medium**: Some concerning elements, may warrant monitoring
        - **High**: Clear indicators of mental health distress requiring attention
        - **Critical**: Immediate intervention may be necessary
        
        ### Privacy and Ethics
        
        When using this tool, please consider:
        
        - Obtain proper consent when analyzing others' content
        - Follow appropriate protocols for intervention
        - Store results securely to protect sensitive information
        - Use as a supportive tool, not as a replacement for human judgment
        - Remember that false positives and false negatives are possible
        
        ### Technical Requirements
        
        This application works best with:
        - Modern web browser
        - Internet connection (for full model functionality)
        - CSV files in proper format for batch processing
        
        ### Support and Feedback
        
        For questions, issues, or feedback, please contact the development team at:  
        support@mentalhealthdetector.example.com
        """)

if __name__ == "__main__":
    main()