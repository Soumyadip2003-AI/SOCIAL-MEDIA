import os
from pathlib import Path
import logging

def download_and_save_models():
    """Download the necessary models to a local directory."""
    try:
        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Define model names
        text_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        image_model_name = "google/vit-base-patch16-224"
        
        # Define cache directory
        cache_dir = Path("./model_cache")
        cache_dir.mkdir(exist_ok=True)
        text_model_dir = cache_dir / "text_model"
        image_model_dir = cache_dir / "image_model"
        
        # Download text model
        logger.info(f"Downloading text model: {text_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(text_model_name, cache_dir=text_model_dir)
        tokenizer.save_pretrained(text_model_dir)
        text_model = AutoModel.from_pretrained(text_model_name, cache_dir=text_model_dir)
        text_model.save_pretrained(text_model_dir)
        
        # Download image model
        logger.info(f"Downloading image model: {image_model_name}")
        image_processor = AutoImageProcessor.from_pretrained(image_model_name, cache_dir=image_model_dir)
        image_processor.save_pretrained(image_model_dir)
        image_model = AutoModel.from_pretrained(image_model_name, cache_dir=image_model_dir)
        image_model.save_pretrained(image_model_dir)
        
        logger.info("Models downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False
