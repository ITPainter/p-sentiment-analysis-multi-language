#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create simple and clean Colab notebooks for sentiment analysis
"""

import json
import os

def create_korean_notebook():
    """Create Korean sentiment analysis notebook"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "title"},
                "source": [
                    "# 🇰🇷 Korean Sentiment Analysis\n\n",
                    "Simple Korean sentiment analysis in Google Colab.\n\n",
                    "## How to use:\n",
                    "1. Upload Excel file with text column\n",
                    "2. Run all cells\n",
                    "3. Download results"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "install"},
                "source": ["## 📦 Install Packages"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install-packages"},
                "outputs": [],
                "source": [
                    "!pip install torch transformers pandas openpyxl matplotlib seaborn wordcloud"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "imports"},
                "source": ["## 📚 Import Libraries"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "import-libs"},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
                    "from google.colab import files\n",
                    "from io import BytesIO\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "print('✅ Libraries loaded')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "model"},
                "source": ["## 🤖 Load Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "load-model"},
                "outputs": [],
                "source": [
                    "# Load Korean sentiment model\n",
                    "model_name = 'snunlp/KR-FinBert-SC'\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "model.to(device)\n",
                    "print(f'✅ Model loaded on {device}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "upload"},
                "source": ["## 📁 Upload File"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload-file"},
                "outputs": [],
                "source": [
                    "# Upload Excel file\n",
                    "uploaded = files.upload()\n",
                    "filename = list(uploaded.keys())[0]\n",
                    "df = pd.read_excel(BytesIO(uploaded[filename]))\n",
                    "print(f'✅ File loaded: {len(df)} rows')\n",
                    "print('Columns:', list(df.columns))\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "analyze"},
                "source": ["## 🧠 Analyze Sentiment"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "sentiment-analysis"},
                "outputs": [],
                "source": [
                    "# Set text column name (change this to match your file)\n",
                    "text_column = 'text'\n",
                    "\n",
                    "def analyze_sentiment(text):\n",
                    "    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)\n",
                    "    inputs = {k: v.to(device) for k, v in inputs.items()}\n",
                    "    with torch.no_grad():\n",
                    "        outputs = model(**inputs)\n",
                    "        probs = torch.softmax(outputs.logits, dim=1)\n",
                    "        sentiment_id = torch.argmax(probs, dim=1).item()\n",
                    "        confidence = probs[0][sentiment_id].item()\n",
                    "        labels = ['negative', 'neutral', 'positive']\n",
                    "        return labels[sentiment_id], confidence\n",
                    "\n",
                    "# Analyze all texts\n",
                    "results = []\n",
                    "for text in df[text_column]:\n",
                    "    if pd.isna(text):\n",
                    "        sentiment, conf = 'neutral', 0.0\n",
                    "    else:\n",
                    "        sentiment, conf = analyze_sentiment(str(text))\n",
                    "    results.append({'text': text, 'sentiment': sentiment, 'confidence': conf})\n",
                    "\n",
                    "results_df = pd.DataFrame(results)\n",
                    "print('✅ Analysis complete!')\n",
                    "print('\\nSentiment distribution:')\n",
                    "print(results_df['sentiment'].value_counts())\n",
                    "results_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "visualize"},
                "source": ["## 🎨 Visualize Results"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "create-charts"},
                "outputs": [],
                "source": [
                    "# Create charts\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                    "\n",
                    "# Pie chart\n",
                    "sentiment_counts = results_df['sentiment'].value_counts()\n",
                    "ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')\n",
                    "ax1.set_title('Sentiment Distribution')\n",
                    "\n",
                    "# Bar chart\n",
                    "ax2.bar(sentiment_counts.index, sentiment_counts.values)\n",
                    "ax2.set_title('Sentiment Counts')\n",
                    "ax2.set_ylabel('Count')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "download"},
                "source": ["## 💾 Download Results"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "save-results"},
                "outputs": [],
                "source": [
                    "# Save results\n",
                    "output_file = 'korean_sentiment_results.xlsx'\n",
                    "results_df.to_excel(output_file, index=False)\n",
                    "files.download(output_file)\n",
                    "print('✅ Results downloaded!')"
                ]
            }
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open('korean_sentiment_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Korean notebook created!")

def create_english_notebook():
    """Create English sentiment analysis notebook"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "title"},
                "source": [
                    "# 🇺🇸 English Sentiment Analysis\n\n",
                    "Simple English sentiment analysis in Google Colab.\n\n",
                    "## How to use:\n",
                    "1. Upload Excel file with text column\n",
                    "2. Run all cells\n",
                    "3. Download results"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "install"},
                "source": ["## 📦 Install Packages"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install-packages"},
                "outputs": [],
                "source": [
                    "!pip install torch transformers pandas openpyxl matplotlib seaborn wordcloud"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "imports"},
                "source": ["## 📚 Import Libraries"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "import-libs"},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
                    "from google.colab import files\n",
                    "from io import BytesIO\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "print('✅ Libraries loaded')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "model"},
                "source": ["## 🤖 Load Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "load-model"},
                "outputs": [],
                "source": [
                    "# Load English sentiment model\n",
                    "model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "model.to(device)\n",
                    "print(f'✅ Model loaded on {device}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "upload"},
                "source": ["## 📁 Upload File"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload-file"},
                "outputs": [],
                "source": [
                    "# Upload Excel file\n",
                    "uploaded = files.upload()\n",
                    "filename = list(uploaded.keys())[0]\n",
                    "df = pd.read_excel(BytesIO(uploaded[filename]))\n",
                    "print(f'✅ File loaded: {len(df)} rows')\n",
                    "print('Columns:', list(df.columns))\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "analyze"},
                "source": ["## 🧠 Analyze Sentiment"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "sentiment-analysis"},
                "outputs": [],
                "source": [
                    "# Set text column name (change this to match your file)\n",
                    "text_column = 'text'\n",
                    "\n",
                    "def analyze_sentiment(text):\n",
                    "    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)\n",
                    "    inputs = {k: v.to(device) for k, v in inputs.items()}\n",
                    "    with torch.no_grad():\n",
                    "        outputs = model(**inputs)\n",
                    "        probs = torch.softmax(outputs.logits, dim=1)\n",
                    "        sentiment_id = torch.argmax(probs, dim=1).item()\n",
                    "        confidence = probs[0][sentiment_id].item()\n",
                    "        labels = ['negative', 'neutral', 'positive']\n",
                    "        return labels[sentiment_id], confidence\n",
                    "\n",
                    "# Analyze all texts\n",
                    "results = []\n",
                    "for text in df[text_column]:\n",
                    "    if pd.isna(text):\n",
                    "        sentiment, conf = 'neutral', 0.0\n",
                    "    else:\n",
                    "        sentiment, conf = analyze_sentiment(str(text))\n",
                    "    results.append({'text': text, 'sentiment': sentiment, 'confidence': conf})\n",
                    "\n",
                    "results_df = pd.DataFrame(results)\n",
                    "print('✅ Analysis complete!')\n",
                    "print('\\nSentiment distribution:')\n",
                    "print(results_df['sentiment'].value_counts())\n",
                    "results_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "visualize"},
                "source": ["## 🎨 Visualize Results"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "create-charts"},
                "outputs": [],
                "source": [
                    "# Create charts\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                    "\n",
                    "# Pie chart\n",
                    "sentiment_counts = results_df['sentiment'].value_counts()\n",
                    "ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')\n",
                    "ax1.set_title('Sentiment Distribution')\n",
                    "\n",
                    "# Bar chart\n",
                    "ax2.bar(sentiment_counts.index, sentiment_counts.values)\n",
                    "ax2.set_title('Sentiment Counts')\n",
                    "ax2.set_ylabel('Count')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "download"},
                "source": ["## 💾 Download Results"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "save-results"},
                "outputs": [],
                "source": [
                    "# Save results\n",
                    "output_file = 'english_sentiment_results.xlsx'\n",
                    "results_df.to_excel(output_file, index=False)\n",
                    "files.download(output_file)\n",
                    "print('✅ Results downloaded!')"
                ]
            }
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open('english_sentiment_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ English notebook created!")

def create_chinese_notebook():
    """Create Chinese sentiment analysis notebook"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "title"},
                "source": [
                    "# 🇨🇳 Chinese Sentiment Analysis\n\n",
                    "Simple Chinese sentiment analysis in Google Colab.\n\n",
                    "## How to use:\n",
                    "1. Upload Excel file with text column\n",
                    "2. Run all cells\n",
                    "3. Download results"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "install"},
                "source": ["## 📦 Install Packages"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install-packages"},
                "outputs": [],
                "source": [
                    "!pip install torch transformers pandas openpyxl matplotlib seaborn wordcloud"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "imports"},
                "source": ["## 📚 Import Libraries"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "import-libs"},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
                    "from google.colab import files\n",
                    "from io import BytesIO\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "print('✅ Libraries loaded')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "model"},
                "source": ["## 🤖 Load Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "load-model"},
                "outputs": [],
                "source": [
                    "# Load Chinese sentiment model\n",
                    "model_name = 'IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment'\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "model.to(device)\n",
                    "print(f'✅ Model loaded on {device}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "upload"},
                "source": ["## 📁 Upload File"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload-file"},
                "outputs": [],
                "source": [
                    "# Upload Excel file\n",
                    "uploaded = files.upload()\n",
                    "filename = list(uploaded.keys())[0]\n",
                    "df = pd.read_excel(BytesIO(uploaded[filename]))\n",
                    "print(f'✅ File loaded: {len(df)} rows')\n",
                    "print('Columns:', list(df.columns))\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "analyze"},
                "source": ["## 🧠 Analyze Sentiment"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "sentiment-analysis"},
                "outputs": [],
                "source": [
                    "# Set text column name (change this to match your file)\n",
                    "text_column = 'text'\n",
                    "\n",
                    "def analyze_sentiment(text):\n",
                    "    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)\n",
                    "    inputs = {k: v.to(device) for k, v in inputs.items()}\n",
                    "    with torch.no_grad():\n",
                    "        outputs = model(**inputs)\n",
                    "        probs = torch.softmax(outputs.logits, dim=1)\n",
                    "        sentiment_id = torch.argmax(probs, dim=1).item()\n",
                    "        confidence = probs[0][sentiment_id].item()\n",
                    "        labels = ['negative', 'positive']\n",
                    "        return labels[sentiment_id], confidence\n",
                    "\n",
                    "# Analyze all texts\n",
                    "results = []\n",
                    "for text in df[text_column]:\n",
                    "    if pd.isna(text):\n",
                    "        sentiment, conf = 'neutral', 0.0\n",
                    "    else:\n",
                    "        sentiment, conf = analyze_sentiment(str(text))\n",
                    "    results.append({'text': text, 'sentiment': sentiment, 'confidence': conf})\n",
                    "\n",
                    "results_df = pd.DataFrame(results)\n",
                    "print('✅ Analysis complete!')\n",
                    "print('\\nSentiment distribution:')\n",
                    "print(results_df['sentiment'].value_counts())\n",
                    "results_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "visualize"},
                "source": ["## 🎨 Visualize Results"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "create-charts"},
                "outputs": [],
                "source": [
                    "# Create charts\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                    "\n",
                    "# Pie chart\n",
                    "sentiment_counts = results_df['sentiment'].value_counts()\n",
                    "ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')\n",
                    "ax1.set_title('Sentiment Distribution')\n",
                    "\n",
                    "# Bar chart\n",
                    "ax2.bar(sentiment_counts.index, sentiment_counts.values)\n",
                    "ax2.set_title('Sentiment Counts')\n",
                    "ax2.set_ylabel('Count')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "download"},
                "source": ["## 💾 Download Results"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "save-results"},
                "outputs": [],
                "source": [
                    "# Save results\n",
                    "output_file = 'chinese_sentiment_results.xlsx'\n",
                    "results_df.to_excel(output_file, index=False)\n",
                    "files.download(output_file)\n",
                    "print('✅ Results downloaded!')"
                ]
            }
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open('chinese_sentiment_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Chinese notebook created!")

def create_japanese_notebook():
    """Create Japanese sentiment analysis notebook"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "title"},
                "source": [
                    "# 🇯🇵 Japanese Sentiment Analysis\n\n",
                    "Simple Japanese sentiment analysis in Google Colab.\n\n",
                    "## How to use:\n",
                    "1. Upload Excel file with text column\n",
                    "2. Run all cells\n",
                    "3. Download results"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "install"},
                "source": ["## 📦 Install Packages"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install-packages"},
                "outputs": [],
                "source": [
                    "!pip install torch transformers pandas openpyxl matplotlib seaborn wordcloud"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "imports"},
                "source": ["## 📚 Import Libraries"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "import-libs"},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
                    "from google.colab import files\n",
                    "from io import BytesIO\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "print('✅ Libraries loaded')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "model"},
                "source": ["## 🤖 Load Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "load-model"},
                "outputs": [],
                "source": [
                    "# Load Japanese sentiment model\n",
                    "model_name = 'cl-tohoku/bert-base-japanese-v3'\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "model.to(device)\n",
                    "print(f'✅ Model loaded on {device}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "upload"},
                "source": ["## 📁 Upload File"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload-file"},
                "outputs": [],
                "source": [
                    "# Upload Excel file\n",
                    "uploaded = files.upload()\n",
                    "filename = list(uploaded.keys())[0]\n",
                    "df = pd.read_excel(BytesIO(uploaded[filename]))\n",
                    "print(f'✅ File loaded: {len(df)} rows')\n",
                    "print('Columns:', list(df.columns))\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "analyze"},
                "source": ["## 🧠 Analyze Sentiment"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "sentiment-analysis"},
                "outputs": [],
                "source": [
                    "# Set text column name (change this to match your file)\n",
                    "text_column = 'text'\n",
                    "\n",
                    "def analyze_sentiment(text):\n",
                    "    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)\n",
                    "    inputs = {k: v.to(device) for k, v in inputs.items()}\n",
                    "    with torch.no_grad():\n",
                    "        outputs = model(**inputs)\n",
                    "        probs = torch.softmax(outputs.logits, dim=1)\n",
                    "        sentiment_id = torch.argmax(probs, dim=1).item()\n",
                    "        confidence = probs[0][sentiment_id].item()\n",
                    "        labels = ['negative', 'positive']\n",
                    "        return labels[sentiment_id], confidence\n",
                    "\n",
                    "# Analyze all texts\n",
                    "results = []\n",
                    "for text in df[text_column]:\n",
                    "    if pd.isna(text):\n",
                    "        sentiment, conf = 'neutral', 0.0\n",
                    "    else:\n",
                    "        sentiment, conf = analyze_sentiment(str(text))\n",
                    "    results.append({'text': text, 'sentiment': sentiment, 'confidence': conf})\n",
                    "\n",
                    "results_df = pd.DataFrame(results)\n",
                    "print('✅ Analysis complete!')\n",
                    "print('\\nSentiment distribution:')\n",
                    "print(results_df['sentiment'].value_counts())\n",
                    "results_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "visualize"},
                "source": ["## 🎨 Visualize Results"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "create-charts"},
                "outputs": [],
                "source": [
                    "# Create charts\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                    "\n",
                    "# Pie chart\n",
                    "sentiment_counts = results_df['sentiment'].value_counts()\n",
                    "ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')\n",
                    "ax1.set_title('Sentiment Distribution')\n",
                    "\n",
                    "# Bar chart\n",
                    "ax2.bar(sentiment_counts.index, sentiment_counts.values)\n",
                    "ax2.set_title('Sentiment Counts')\n",
                    "ax2.set_ylabel('Count')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "download"},
                "source": ["## 💾 Download Results"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "save-results"},
                "outputs": [],
                "source": [
                    "# Save results\n",
                    "output_file = 'japanese_sentiment_results.xlsx'\n",
                    "results_df.to_excel(output_file, index=False)\n",
                    "files.download(output_file)\n",
                    "print('✅ Results downloaded!')"
                ]
            }
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open('japanese_sentiment_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Japanese notebook created!")

def main():
    """Create all notebooks"""
    print("🚀 Creating Colab notebooks...")
    
    # Create all notebooks
    create_korean_notebook()
    create_english_notebook()
    create_chinese_notebook()
    create_japanese_notebook()
    
    print("\n🎉 All notebooks created successfully!")
    print("\n📁 Files created:")
    print("- korean_sentiment_analysis.ipynb")
    print("- english_sentiment_analysis.ipynb")
    print("- chinese_sentiment_analysis.ipynb")
    print("- japanese_sentiment_analysis.ipynb")
    print("\n💡 Upload these files to Google Colab to use them!")

if __name__ == "__main__":
    main()
