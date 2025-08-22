# 🌍 Multi-Language Sentiment Analysis System

A Python-based sentiment analysis system that supports Korean, Chinese, English, and Japanese text analysis.

## 🌟 Key Features

- **🌐 Multi-language Support**: Automatic detection and analysis of Korean, Chinese, English, and Japanese text
- **📊 Excel I/O**: Excel file reading/writing with visualization output
- **🤖 Advanced Models**: State-of-the-art pre-trained models for each language
- **🎨 Visualization**: Charts, wordclouds, and comparison graphs
- **⚡ Batch Processing**: Efficient processing of multiple sheets and large datasets
- **🔧 Flexible Configuration**: Language-specific model selection and processing parameter customization

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Analysis
```bash
cd src

# Option 1: Analyze Excel file (place in output/ folder first)
python main.py your_data.xlsx

# Option 2: Create sample data and run analysis
python main.py

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
│       ├── 🇯🇵 japanese_sentiment_analysis.ipynb
│       └── 🔧 create_notebooks.py   # Notebook generation script
├── 📁 output/                       # Results and system files
│   ├── 🇰🇷 korea/                   # Korean results
│   ├── 🇨🇳 china/                   # Chinese results
│   ├── 🇺🇸 english/                 # English results
│   ├── 🇯🇵 japan/                   # Japanese results
│   ├── 📊 Language_Comparison_Chart.png
│   └── 📁 logs/                     # System logs
└── 📋 requirements.txt               # Dependency packages
```

## 🎯 Supported Languages

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
- **WordClouds**: Text frequency visualizations
- **Comparison Charts**: Cross-language sentiment analysis comparison

### 🎨 Visualization Examples
- Sentiment distribution bar charts
- Confidence distribution by sentiment
- Sentiment distribution pie charts
- Keyword wordclouds
- Cross-language sentiment comparison charts

## 🔧 Configuration and Customization

### Core Configuration File
You can customize the following items in `src/core_config.py`:

- **Model Selection**: Language-specific model selection
- **Processing Parameters**: Batch size, maximum length, confidence threshold
- **Visualization Settings**: Chart size, wordcloud options
- **Output Options**: File format, save path

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
```

## 📝 Usage Examples

### 1. Analyze Excel File
```bash
# 1. Place Excel file in output/ folder
# 2. Run analysis
python src/main.py your_data.xlsx
```

### 2. Run Analysis with Sample Data
```bash
python src/main.py
```

### 3. Force Create Sample Data
```bash
python src/main.py --sample
```

### 4. Advanced Options
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
2. Install required packages
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
- macOS 10.15+
- Ubuntu 18.04+

### GPU Support (Optional)
```bash
# Install PyTorch with CUDA 11.8+ support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔍 Key Features in Detail

### 1. Automatic Language Detection
- Automatic language detection from text input
- Confidence-based language selection
- Fallback language support

### 2. Multi-Model Support
- Language-optimized models for each language
- Alternative model and fallback model support
- Model caching for performance optimization

### 3. Batch Processing
- Efficient processing of large datasets
- Memory usage optimization
- Progress display and logging

### 4. Result Visualization
- High-quality charts based on matplotlib
- Language-specific color themes
- Wordcloud generation
- Comparative analysis charts

## 🚨 Important Notes

- **First Run**: Transformers models are automatically downloaded (internet connection required)
- **GPU Usage**: CUDA version verification required
- **Windows Environment**: Visual C++ build tools may be required for some package installations
- **Memory**: Sufficient RAM required for large dataset processing

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

