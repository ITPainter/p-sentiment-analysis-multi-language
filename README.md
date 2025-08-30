# 🌍 Multi-Language Sentiment Analysis System

A Python-based sentiment analysis system that supports Korean, Chinese, English, and Japanese text analysis with advanced AI models and comprehensive visualization.

## 🌟 Key Features

- **🌐 Multi-language Support**: Automatic detection and analysis of Korean, Chinese, English, and Japanese text
- **📊 Excel I/O**: Excel file reading/writing with comprehensive visualization output
- **🤖 Advanced AI Models**: State-of-the-art pre-trained models optimized for each language
- **🎨 Rich Visualization**: Bar charts, pie charts, wordclouds, confidence histograms, and comparison graphs
- **⚡ Batch Processing**: Efficient processing of multiple sheets and large datasets
- **🔧 Flexible Configuration**: Language-specific model selection and processing parameter customization
- **💻 Google Colab Ready**: Jupyter notebooks for cloud-based execution

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install required packages
pip install -r requirements.txt
```

### 2. Run Analysis
```bash
cd src

# Option 1: Create sample data and run analysis (recommended for first use)
python main.py

# Option 2: Analyze your Excel file (place in output/ folder first)
python main.py your_data.xlsx

# Option 3: Force create sample data
python main.py --sample

# Option 4: Show help
python main.py --help
```

## 📁 Project Structure

```
p-sentiment-analysis-multi-language/
├── 📁 src/                          # Source code
│   ├── 🚀 main.py                   # Main execution file (with sample data creation)
│   ├── ⚙️ core_config.py            # Configuration management
│   ├── 🤖 core_models.py            # Model management
│   ├── 🌐 core_language.py          # Language detection
│   ├── 📊 data_excel.py             # Excel data processing
│   ├── 🎨 data_charts.py            # Visualization
│   ├── 📝 data_sample.py            # Sample data creation
│   └── 📁 colab/                    # Google Colab notebooks
│       ├── 🇰🇷 korean_sentiment_analysis.ipynb
│       ├── 🇨🇳 chinese_sentiment_analysis.ipynb
│       ├── 🇺🇸 english_sentiment_analysis.ipynb
│       └── 🇯🇵 japanese_sentiment_analysis.ipynb
├── 📁 output/                       # Results and system files
│   ├── 🇰🇷 korea/                   # Korean results
│   ├── 🇨🇳 china/                   # Chinese results
│   ├── 🇺🇸 english/                 # English results
│   ├── 🇯🇵 japan/                   # Japanese results
│   ├── 📊 Language_Comparison_Chart.png
│   ├── 📁 logs/                     # System logs
│   └── 📁 cache/                    # Model cache
├── 📋 requirements.txt               # Dependency packages
├── 📖 README.md                     # Project documentation
└── 🚫 .gitignore                    # Git ignore file
```

## 🎯 Supported Languages & Models

| Language | Code | Primary Model | Alternative Model | Fallback Model |
|----------|------|---------------|-------------------|----------------|
| 🇰🇷 Korean | `k` | snunlp/KR-FinBert-SC | beomi/KcELECTRA-base-v2022 | klue/roberta-base |
| 🇨🇳 Chinese | `c` | IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment | IDEAL-Future/bert-base-chinese-finetuned-douban-movie | hfl/chinese-roberta-wwm-ext |
| 🇺🇸 English | `e` | cardiffnlp/twitter-roberta-base-sentiment-latest | nlptown/bert-base-multilingual-uncased-sentiment | distilbert-base-uncased-finetuned-sst-2-english |
| 🇯🇵 Japanese | `j` | cl-tohoku/bert-base-japanese-v3 | rinna/japanese-roberta-base | megagonlabs/roberta-base-japanese-sentiment |

## 📊 Output Results

### 📁 File Results
- **Excel Files**: Sentiment labels, confidence scores, and metadata
- **Charts**: Bar charts, pie charts, and confidence histograms
- **WordClouds**: Text frequency visualizations with language-specific fonts
- **Comparison Charts**: Cross-language sentiment analysis comparison

### 🎨 Visualization Examples
- Sentiment distribution bar charts
- Confidence distribution by sentiment
- Sentiment distribution pie charts
- Keyword wordclouds with proper font support
- Cross-language sentiment comparison charts

## 🔧 Configuration and Customization

### Core Configuration File
You can customize the following items in `src/core_config.py`:

- **Model Selection**: Language-specific model selection
- **Processing Parameters**: Batch size, maximum length, confidence threshold
- **Visualization Settings**: Chart size, wordcloud options, font settings
- **Output Options**: File format, save path, language-specific output folders

### Key Configuration Options
```python
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

# Font settings for visualization
FONT_CONFIG = {
    'korean': 'Apple SD Gothic Neo',    # Korean font
    'chinese': 'PingFang',              # Chinese font
    'japanese': 'ヒラギノ角ゴシック',      # Japanese font
    'english': 'DejaVu Sans'            # English font
}
```

## 📝 Usage Examples

### 1. First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with sample data to test the system
cd src
python main.py
```

### 2. Analyze Your Excel File
```bash
# 1. Place Excel file in output/ folder
# 2. Run analysis
python src/main.py your_data.xlsx
```

### 3. Advanced Options
```bash
# Specify output directory
python src/main.py input_file.xlsx -o /path/to/output

# Debug mode
python src/main.py input_file.xlsx --debug

# Show help
python src/main.py --help
```

## 🌐 Google Colab Support

The `src/colab/` folder contains Google Colab notebooks for each language:

- **🇰🇷 Korean**: `korean_sentiment_analysis.ipynb`
- **🇨🇳 Chinese**: `chinese_sentiment_analysis.ipynb`
- **🇺🇸 English**: `english_sentiment_analysis.ipynb`
- **🇯🇵 Japanese**: `japanese_sentiment_analysis.ipynb`

### Colab Usage
1. Upload the desired language notebook to Colab
2. Install required packages (automatic in notebook)
3. Upload Excel file
4. Execute code to perform sentiment analysis
5. Download results

## 📋 System Requirements

### Basic Requirements
- **Python**: 3.8+ (Recommended: 3.11+)
- **Memory**: 4GB+ RAM recommended
- **Internet**: Required for model download on first run
- **GPU**: Optional (CUDA support)

### Supported Operating Systems
- Windows 10+
- macOS 10.15+ (with optimized font support)
- Ubuntu 18.04+

### GPU Support (Optional)
```bash
# Install PyTorch with CUDA 11.8+ support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔍 Key Features in Detail

### 1. Automatic Language Detection
- Advanced pattern-based language detection
- Confidence-based language selection
- Fallback language support
- Batch language detection for efficiency

### 2. Multi-Model Support
- Language-optimized models for each language
- Alternative model and fallback model support
- Model caching for performance optimization
- Automatic model download and management

### 3. Advanced Text Processing
- Efficient batch processing of large datasets
- Memory usage optimization
- Progress display and comprehensive logging
- Error handling and recovery

### 4. Rich Visualization
- High-quality charts based on matplotlib
- Language-specific color themes and fonts
- Wordcloud generation with proper font support
- Comparative analysis charts
- Confidence distribution analysis

### 5. Font Management
- Automatic font detection for each language
- macOS-optimized font selection
- Fallback font support
- Proper rendering of CJK characters

## 🚨 Important Notes

- **First Run**: Transformers models are automatically downloaded (internet connection required)
- **Model Download**: Models are cached locally for future use
- **GPU Usage**: CUDA version verification required for GPU acceleration
- **Windows Environment**: Visual C++ build tools may be required for some package installations
- **Memory**: Sufficient RAM required for large dataset processing
- **Font Support**: System fonts are automatically detected and used

## 📦 Dependencies

### Core AI Libraries
- `torch>=1.12.0` - PyTorch deep learning framework
- `transformers>=4.20.0` - Hugging Face transformers
- `protobuf>=3.20.0` - Protocol buffers for model loading

### Data Processing
- `pandas>=1.5.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computing
- `openpyxl>=3.0.0` - Excel file handling

### Visualization
- `matplotlib>=3.5.0` - Chart generation
- `wordcloud>=1.8.0` - Word cloud creation
- `pillow>=11.3.0` - Image processing

### Language-Specific
- `jieba>=0.42.1` - Chinese tokenization
- `mecab-python3>=1.0.5` - Japanese tokenization
- `fugashi>=1.5.0` - Japanese tokenizer
- `unidic-lite>=1.0.8` - Japanese dictionary
- `langdetect>=1.0.9` - Language detection

## 🤝 Contributing

This system is designed for both beginners and experts. Feel free to modify and improve the code according to your needs.

### How to Contribute
1. Fork the project
2. Create a feature branch
3. Modify code and test
4. Create a Pull Request

## 📄 License

Open source project for educational and research purposes.

## 📞 Support and Issues

For project-related questions or bug reports, please submit them through GitHub Issues.

---

**🎉 Thank you for using the Multi-Language Sentiment Analysis System!**

**✨ All 4 languages (Korean, Chinese, English, Japanese) are fully supported and tested!**

