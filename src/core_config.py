# -*- coding: utf-8 -*-
"""
Multi-Language Sentiment Analysis Configuration
"""

import os
from pathlib import Path

# =============================================================================
# 📁 Project Path Settings
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Language-specific output folders
LANGUAGE_OUTPUT_DIRS = {
    'k': OUTPUT_DIR / "korea",  # Korean
    'c': OUTPUT_DIR / "china",  # Chinese
    'e': OUTPUT_DIR / "english",  # English
    'j': OUTPUT_DIR / "japan"   # Japanese
}

# =============================================================================
# 🔧 Model Settings
# =============================================================================

# Korean models
KOREAN_MODELS = {
    'primary': "snunlp/KR-FinBert-SC",           # Main model
    'alternative': "beomi/KcELECTRA-base-v2022",  # Alternative model
    'fallback': "klue/roberta-base"              # Fallback model
}

# Chinese models
CHINESE_MODELS = {
    'primary': "IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment",  # Main model
    'alternative': "IDEAL-Future/bert-base-chinese-finetuned-douban-movie",  # Alternative model
    'fallback': "hfl/chinese-roberta-wwm-ext"                   # Fallback model
}

# English models
ENGLISH_MODELS = {
    'primary': "cardiffnlp/twitter-roberta-base-sentiment-latest",  # Main model
    'alternative': "nlptown/bert-base-multilingual-uncased-sentiment",  # Alternative model
    'fallback': "distilbert-base-uncased-finetuned-sst-2-english"      # Fallback model
}

# Japanese models
JAPANESE_MODELS = {
    'primary': "cl-tohoku/bert-base-japanese-v3",           # Main model
    'alternative': "rinna/japanese-roberta-base",            # Alternative model
    'fallback': "megagonlabs/roberta-base-japanese-sentiment"  # Fallback model
}

# Language-to-model mapping
LANGUAGE_MODELS = {
    'k': KOREAN_MODELS,
    'c': CHINESE_MODELS,
    'e': ENGLISH_MODELS,
    'j': JAPANESE_MODELS
}

# =============================================================================
# ⚙️ Processing Settings
# =============================================================================

# Sentiment analysis settings
SENTIMENT_CONFIG = {
    'batch_size': 16,           # Batch size
    'max_length': 128,          # Maximum text length
    'confidence_threshold': 0.8, # Confidence threshold
    'device': 'auto'            # Device auto-selection
}

# Language detection settings
LANGUAGE_DETECTION_CONFIG = {
    'min_confidence': 0.7,      # Minimum confidence for language detection
    'fallback_language': 'e'    # Default language (English)
}

# =============================================================================
# 📊 Visualization Settings
# =============================================================================

# WordCloud settings
WORDCLOUD_CONFIG = {
    'width': 1600,
    'height': 800,
    'background_color': 'white',
    'max_words': 300,
    'min_font_size': 12,
    'relative_scaling': 0.5,
    'colormap': 'plasma',
    'collocations': False
}

# Chart settings
CHART_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'colors': {
        'positive': ['lightgreen', 'green'],
        'neutral': ['lightgray', 'gray'],
        'negative': ['lightcoral', 'red']
    }
}

# =============================================================================
# 📝 File Settings
# =============================================================================

# Input file settings
INPUT_CONFIG = {
    'supported_formats': ['.xlsx', '.xls'],
    'text_columns': ['comment', '댓글', 'review', 'text', 'comments', '텍스트'],
    'encoding': 'utf-8'
}

# Output file settings
OUTPUT_CONFIG = {
    'format': 'xlsx',
    'include_original': True,
    'include_confidence': True,
    'include_language': True,
    'include_metadata': True
}

# =============================================================================
# 🌐 Language-specific Settings
# =============================================================================

# Korean settings
KOREAN_CONFIG = {
    'font_path': None,  # Use system fonts on macOS
    'preprocessing': True,
    'stopwords': True
}

# Chinese settings
CHINESE_CONFIG = {
    'use_jieba': True,
    'traditional_to_simplified': True,
    'remove_punctuation': True
}

# English settings
ENGLISH_CONFIG = {
    'use_nltk': True,
    'remove_stopwords': True,
    'lemmatization': True
}

# Japanese settings
JAPANESE_CONFIG = {
    'use_mecab': True,
    'remove_particles': True,
    'normalize_text': True
}

# Language-to-config mapping
LANGUAGE_CONFIGS = {
    'k': KOREAN_CONFIG,
    'c': CHINESE_CONFIG,
    'e': ENGLISH_CONFIG,
    'j': JAPANESE_CONFIG
}

# =============================================================================
# 🔍 Logging Settings
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': OUTPUT_DIR / 'logs' / 'sentiment_analysis.log'
}

# =============================================================================
# 🚀 Performance Settings
# =============================================================================

PERFORMANCE_CONFIG = {
    'use_cache': True,
    'cache_dir': OUTPUT_DIR / 'cache',
    'parallel_processing': True,
    'max_workers': 4,
    'memory_limit': '4GB'
}

# =============================================================================
# 📋 Utility Functions
# =============================================================================

def ensure_directories():
    """Create necessary directories."""
    for lang_dir in LANGUAGE_OUTPUT_DIRS.values():
        lang_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log and cache directories under output folder
    (OUTPUT_DIR / 'logs').mkdir(exist_ok=True)
    (OUTPUT_DIR / 'cache').mkdir(exist_ok=True)

def get_model_name(language, model_type='primary'):
    """Get model name for language and model type."""
    if language in LANGUAGE_MODELS:
        return LANGUAGE_MODELS[language].get(model_type, LANGUAGE_MODELS[language]['primary'])
    return None

def get_output_path(language, filename):
    """Get output path for language and filename."""
    if language in LANGUAGE_OUTPUT_DIRS:
        return LANGUAGE_OUTPUT_DIRS[language] / filename
    return OUTPUT_DIR / filename

# Initialize directories
ensure_directories()
