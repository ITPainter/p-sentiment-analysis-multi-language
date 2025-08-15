# Multi-Language Sentiment Analysis System

A Python-based sentiment analysis system that supports Korean, Chinese, English, and Japanese text analysis.

## 🌍 Supported Languages

- 🇰🇷 Korean (한국어)
- 🇨🇳 Chinese (中文)
- 🇺🇸 English
- 🇯🇵 Japanese (日本語)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Analysis
```bash
cd src

# Option 1: Analyze your Excel file (place it in output/ folder first)
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
├── src/                    # Source code
│   ├── main.py            # Main execution (with sample data creation)
│   ├── core_config.py     # Configuration
│   ├── core_models.py     # Model management
│   ├── core_language.py   # Language detection
│   ├── data_excel.py      # Excel data processing
│   ├── data_charts.py     # Visualization
│   └── data_sample.py     # Sample data creation
├── output/                 # Results and system files
│   ├── korea/             # Korean results
│   ├── china/             # Chinese results
│   ├── english/            # English results
│   ├── japan/             # Japanese results
│   ├── logs/               # System logs
│   └── cache/              # Model cache
└── requirements.txt        # Dependencies
```

## 🎯 Features

- **Multi-language Support**: Automatically detects and analyzes text in 4 languages
- **Excel I/O**: Reads Excel files and outputs results with visualizations
- **Advanced Models**: Uses state-of-the-art pre-trained models for each language
- **Visualization**: Generates charts, wordclouds, and comparison graphs
- **Batch Processing**: Handles multiple sheets and large datasets efficiently

## 📊 Output

- **Excel Files**: Results with sentiment labels, confidence scores, and metadata
- **Charts**: Bar charts, pie charts, and confidence histograms
- **WordClouds**: Text frequency visualizations
- **Comparison Charts**: Cross-language sentiment analysis comparison

## 📁 File Organization

- **Input Files**: Place your Excel files in the `output/` folder
- **Output Results**: Automatically organized by language in `output/korea/`, `output/china/`, etc.
- **System Files**: Logs and cache stored in `output/logs/` and `output/cache/`

## 🔧 Configuration

Edit `src/core_config.py` to customize:
- Model selection for each language
- Processing parameters
- Visualization settings
- Output options

## 📝 Usage Examples

### Analyze Your Excel File
```bash
# 1. Place your Excel file in the output/ folder
# 2. Run analysis
python src/main.py your_data.xlsx
```

### Create Sample Data and Run Analysis
```bash
python src/main.py
```

### Force Create Sample Data
```bash
python src/main.py --sample
```

### Show Help
```bash
python src/main.py --help
```

### Advanced Options
```bash
python src/main.py input_file.xlsx -o /path/to/output
python src/main.py input_file.xlsx --debug
```

## ⚠️ Requirements

- Python 3.8+
- 4GB+ RAM recommended
- Internet connection for first run (model download)
- GPU support optional (CUDA)

## 🤝 Contributing

This system is designed for beginners and experts alike. Feel free to modify and improve the code according to your needs.

## 📄 License

Open source project for educational and research purposes.

