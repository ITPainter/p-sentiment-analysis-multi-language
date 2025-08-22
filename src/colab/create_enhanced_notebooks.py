#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create enhanced Colab notebooks with model information and alternatives
"""

import json
import os

def create_korean_notebook():
    """Create Korean sentiment analysis notebook with model alternatives"""
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
                "metadata": {"id": "models-info"},
                "source": [
                    "## 🤖 Available Korean Models\n\n",
                    "### Primary Model (Current):\n",
                    "- **snunlp/KR-FinBert-SC** - Best performance for Korean sentiment\n\n",
                    "### Alternative Models:\n",
                    "- **beomi/KcELECTRA-base-v2022** - High accuracy alternative\n",
                    "- **klue/roberta-base** - General purpose Korean model\n\n",
                    "### Model Selection:\n",
                    "You can change the `model_name` variable below to use different models."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "install"},
                "source": ["## 📦 Install Required Packages"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install-packages"},
                "outputs": [],
                "source": [
                    "# Install required packages\n",
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
                    "print('✅ Libraries loaded successfully')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "model"},
                "source": ["## 🤖 Load Korean Sentiment Analysis Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "load-model"},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 🚀 Korean Sentiment Analysis Models\n",
                    "# =============================================================================\n",
                    "\n",
                    "# Primary model (best performance)\n",
                    "model_name = 'snunlp/KR-FinBert-SC'\n",
                    "\n",
                    "# Alternative models (uncomment to use)\n",
                    "# model_name = 'beomi/KcELECTRA-base-v2022'  # High accuracy alternative\n",
                    "# model_name = 'klue/roberta-base'           # General purpose Korean model\n",
                    "\n",
                    "print(f\"🔄 Loading model: {model_name}\")\n",
                    "print(\"\\n📋 Model information:\")\n",
                    "print(f\"   - Primary: snunlp/KR-FinBert-SC\")\n",
                    "print(f\"   - Alternative: beomi/KcELECTRA-base-v2022\")\n",
                    "print(f\"   - Fallback: klue/roberta-base\")\n",
                    "\n",
                    "# Load tokenizer and model\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                    "\n",
                    "# Use GPU if available\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "model.to(device)\n",
                    "model.eval()\n",
                    "\n",
                    "print(f\"\\n✅ Model loaded successfully on {device}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "upload"},
                "source": ["## 📁 Upload Excel File"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload-file"},
                "outputs": [],
                "source": [
                    "# Upload Excel file\n",
                    "print(\"📁 Please upload your Excel file (.xlsx)\")\n",
                    "uploaded = files.upload()\n",
                    "\n",
                    "# Get the first uploaded file\n",
                    "filename = list(uploaded.keys())[0]\n",
                    "print(f\"✅ File uploaded: {filename}\")\n",
                    "\n",
                    "# Read Excel file\n",
                    "df = pd.read_excel(BytesIO(uploaded[filename]))\n",
                    "print(f\"📊 Data loaded: {len(df)} rows, {len(df.columns)} columns\")\n",
                    "print(\"\\n📋 Columns:\", list(df.columns))\n",
                    "print(\"\\n🔍 First few rows:\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "select-column"},
                "source": ["## 📝 Select Text Column"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "column-selection"},
                "outputs": [],
                "source": [
                    "# Set text column name (modify this to match your column name)\n",
                    "text_column = 'text'  # Change this to your actual column name\n",
                    "\n",
                    "# Common column names for Korean text\n",
                    "# text_column = 'comment'    # 댓글\n",
                    "# text_column = 'review'     # 리뷰\n",
                    "# text_column = '댓글'       # Korean column name\n",
                    "# text_column = '리뷰'       # Korean column name\n",
                    "\n",
                    "# Check if column exists\n",
                    "if text_column not in df.columns:\n",
                    "    print(f\"❌ Column '{text_column}' not found. Available columns: {list(df.columns)}\")\n",
                    "    print(\"\\n💡 Please modify the 'text_column' variable above to match your column name\")\n",
                    "    print(\"\\n🔍 Common Korean column names: comment, review, 댓글, 리뷰, text\")\n",
                    "else:\n",
                    "    print(f\"✅ Text column selected: {text_column}\")\n",
                    "    print(f\"📝 Sample text: {df[text_column].iloc[0]}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "sentiment-analysis"},
                "source": ["## 🧠 Perform Sentiment Analysis"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "analysis"},
                "outputs": [],
                "source": [
                    "def analyze_sentiment(text):\n",
                    "    \"\"\"Analyze sentiment of Korean text\"\"\"\n",
                    "    try:\n",
                    "        # Tokenize text\n",
                    "        inputs = tokenizer(text, return_tensors=\"pt\", truncation=True, max_length=128, padding=True)\n",
                    "        inputs = {k: v.to(device) for k, v in inputs.items()}\n",
                    "        \n",
                    "        # Get prediction\n",
                    "        with torch.no_grad():\n",
                    "            outputs = model(**inputs)\n",
                    "            probabilities = torch.softmax(outputs.logits, dim=1)\n",
                    "            \n",
                    "        # Get sentiment and confidence\n",
                    "        sentiment_id = torch.argmax(probabilities, dim=1).item()\n",
                    "        confidence = probabilities[0][sentiment_id].item()\n",
                    "        \n",
                    "        # Map sentiment ID to label (Korean models typically use 3 classes)\n",
                    "        sentiment_labels = ['negative', 'neutral', 'positive']\n",
                    "        sentiment = sentiment_labels[sentiment_id]\n",
                    "        \n",
                    "        return sentiment, confidence\n",
                    "    except Exception as e:\n",
                    "        return 'error', 0.0\n",
                    "\n",
                    "# Analyze each text\n",
                    "print(\"🔄 Analyzing sentiments...\")\n",
                    "results = []\n",
                    "\n",
                    "for idx, text in enumerate(df[text_column]):\n",
                    "    if pd.isna(text) or str(text).strip() == '':\n",
                    "        sentiment, confidence = 'neutral', 0.0\n",
                    "    else:\n",
                    "        sentiment, confidence = analyze_sentiment(str(text))\n",
                    "    \n",
                    "    results.append({\n",
                    "        'text': text,\n",
                    "        'sentiment': sentiment,\n",
                    "        'confidence': confidence\n",
                    "    })\n",
                    "    \n",
                    "    # Show progress\n",
                    "    if (idx + 1) % 10 == 0:\n",
                    "        print(f\"Progress: {idx + 1}/{len(df)}\")\n",
                    "\n",
                    "print(\"✅ Sentiment analysis completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "create-results"},
                "source": ["## 📊 Create Results DataFrame"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "results-df"},
                "outputs": [],
                "source": [
                    "# Create results DataFrame\n",
                    "results_df = pd.DataFrame(results)\n",
                    "\n",
                    "# Add original data\n",
                    "for col in df.columns:\n",
                    "    if col != text_column:\n",
                    "        results_df[col] = df[col]\n",
                    "\n",
                    "# Reorder columns\n",
                    "cols = ['text', 'sentiment', 'confidence'] + [col for col in df.columns if col != text_column]\n",
                    "results_df = results_df[cols]\n",
                    "\n",
                    "print(\"📊 Results DataFrame created\")\n",
                    "print(f\"\\n📈 Sentiment distribution:\")\n",
                    "print(results_df['sentiment'].value_counts())\n",
                    "print(f\"\\n🔍 Sample results:\")\n",
                    "results_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "visualization"},
                "source": ["## 🎨 Create Visualizations"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "charts"},
                "outputs": [],
                "source": [
                    "# Create charts\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                    "\n",
                    "# Pie chart\n",
                    "sentiment_counts = results_df['sentiment'].value_counts()\n",
                    "ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')\n",
                    "ax1.set_title('Sentiment Distribution', fontweight='bold')\n",
                    "\n",
                    "# Bar chart\n",
                    "ax2.bar(sentiment_counts.index, sentiment_counts.values)\n",
                    "ax2.set_title('Sentiment Counts', fontweight='bold')\n",
                    "ax2.set_ylabel('Count')\n",
                    "for i, v in enumerate(sentiment_counts.values):\n",
                    "    ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')\n",
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
                    "output_file = f'korean_sentiment_results_{pd.Timestamp.now().strftime(\"%Y%m%d_%H%M%S\")}.xlsx'\n",
                    "results_df.to_excel(output_file, index=False)\n",
                    "files.download(output_file)\n",
                    "print(f\"✅ Results saved to: {output_file}\")\n",
                    "print(\"📥 File download started!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "summary"},
                "source": [
                    "## 🎉 Summary\n\n",
                    "✅ **Korean sentiment analysis completed successfully!**\n\n",
                    "### Model Used:\n",
                    "- **snunlp/KR-FinBert-SC** (Primary model)\n\n",
                    "### Alternative Models Available:\n",
                    "- **beomi/KcELECTRA-base-v2022** - High accuracy alternative\n",
                    "- **klue/roberta-base** - General purpose Korean model\n\n",
                    "### To Try Different Models:\n",
                    "1. Go to the 'Load Model' cell above\n",
                    "2. Comment out the current model_name\n",
                    "3. Uncomment one of the alternative models\n",
                    "4. Re-run the analysis\n\n",
                    "### Results:\n",
                    "- **Total texts analyzed**: Check the results above\n",
                    "- **Sentiment distribution**: Check the results above\n",
                    "- **Average confidence**: Check the results above\n\n",
                    "---\n\n",
                    "**💡 Tip**: Different models may give slightly different results. Try multiple models for best accuracy!"
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
    
    print("✅ Enhanced Korean notebook created!")

def create_english_notebook():
    """Create English sentiment analysis notebook with model alternatives"""
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
                "metadata": {"id": "models-info"},
                "source": [
                    "## 🤖 Available English Models\n\n",
                    "### Primary Model (Current):\n",
                    "- **cardiffnlp/twitter-roberta-base-sentiment-latest** - Best for social media text\n\n",
                    "### Alternative Models:\n",
                    "- **nlptown/bert-base-multilingual-uncased-sentiment** - Multilingual sentiment\n",
                    "- **distilbert-base-uncased-finetuned-sst-2-english** - Fast and efficient\n\n",
                    "### Model Selection:\n",
                    "You can change the `model_name` variable below to use different models."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "install"},
                "source": ["## 📦 Install Required Packages"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install-packages"},
                "outputs": [],
                "source": [
                    "# Install required packages\n",
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
                    "print('✅ Libraries loaded successfully')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "model"},
                "source": ["## 🤖 Load English Sentiment Analysis Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "load-model"},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 🚀 English Sentiment Analysis Models\n",
                    "# =============================================================================\n",
                    "\n",
                    "# Primary model (best for social media)\n",
                    "model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'\n",
                    "\n",
                    "# Alternative models (uncomment to use)\n",
                    "# model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'  # Multilingual\n",
                    "# model_name = 'distilbert-base-uncased-finetuned-sst-2-english'   # Fast & efficient\n",
                    "\n",
                    "print(f\"🔄 Loading model: {model_name}\")\n",
                    "print(\"\\n📋 Model information:\")\n",
                    "print(f\"   - Primary: cardiffnlp/twitter-roberta-base-sentiment-latest\")\n",
                    "print(f\"   - Alternative: nlptown/bert-base-multilingual-uncased-sentiment\")\n",
                    "print(f\"   - Fallback: distilbert-base-uncased-finetuned-sst-2-english\")\n",
                    "\n",
                    "# Load tokenizer and model\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                    "\n",
                    "# Use GPU if available\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "model.to(device)\n",
                    "model.eval()\n",
                    "\n",
                    "print(f\"\\n✅ Model loaded successfully on {device}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "upload"},
                "source": ["## 📁 Upload Excel File"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload-file"},
                "outputs": [],
                "source": [
                    "# Upload Excel file\n",
                    "print(\"📁 Please upload your Excel file (.xlsx)\")\n",
                    "uploaded = files.upload()\n",
                    "\n",
                    "# Get the first uploaded file\n",
                    "filename = list(uploaded.keys())[0]\n",
                    "print(f\"✅ File uploaded: {filename}\")\n",
                    "\n",
                    "# Read Excel file\n",
                    "df = pd.read_excel(BytesIO(uploaded[filename]))\n",
                    "print(f\"📊 Data loaded: {len(df)} rows, {len(df.columns)} columns\")\n",
                    "print(\"\\n📋 Columns:\", list(df.columns))\n",
                    "print(\"\\n🔍 First few rows:\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "select-column"},
                "source": ["## 📝 Select Text Column"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "column-selection"},
                "outputs": [],
                "source": [
                    "# Set text column name (modify this to match your column name)\n",
                    "text_column = 'text'  # Change this to your actual column name\n",
                    "\n",
                    "# Common column names for English text\n",
                    "# text_column = 'comment'    # Comments\n",
                    "# text_column = 'review'     # Reviews\n",
                    "# text_column = 'feedback'   # Feedback\n",
                    "# text_column = 'tweet'      # Social media posts\n",
                    "\n",
                    "# Check if column exists\n",
                    "if text_column not in df.columns:\n",
                    "    print(f\"❌ Column '{text_column}' not found. Available columns: {list(df.columns)}\")\n",
                    "    print(\"\\n💡 Please modify the 'text_column' variable above to match your column name\")\n",
                    "    print(\"\\n🔍 Common English column names: comment, review, feedback, tweet, text\")\n",
                    "else:\n",
                    "    print(f\"✅ Text column selected: {text_column}\")\n",
                    "    print(f\"📝 Sample text: {df[text_column].iloc[0]}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "sentiment-analysis"},
                "source": ["## 🧠 Perform Sentiment Analysis"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "analysis"},
                "outputs": [],
                "source": [
                    "def analyze_sentiment(text):\n",
                    "    \"\"\"Analyze sentiment of English text\"\"\"\n",
                    "    try:\n",
                    "        # Tokenize text\n",
                    "        inputs = tokenizer(text, return_tensors=\"pt\", truncation=True, max_length=128, padding=True)\n",
                    "        inputs = {k: v.to(device) for k, v in inputs.items()}\n",
                    "        \n",
                    "        # Get prediction\n",
                    "        with torch.no_grad():\n",
                    "            outputs = model(**inputs)\n",
                    "            probabilities = torch.softmax(outputs.logits, dim=1)\n",
                    "            \n",
                    "        # Get sentiment and confidence\n",
                    "        sentiment_id = torch.argmax(probabilities, dim=1).item()\n",
                    "        confidence = probabilities[0][sentiment_id].item()\n",
                    "        \n",
                    "        # Map sentiment ID to label (English models typically use 3 classes)\n",
                    "        sentiment_labels = ['negative', 'neutral', 'positive']\n",
                    "        sentiment = sentiment_labels[sentiment_id]\n",
                    "        \n",
                    "        return sentiment, confidence\n",
                    "    except Exception as e:\n",
                    "        return 'error', 0.0\n",
                    "\n",
                    "# Analyze each text\n",
                    "print(\"🔄 Analyzing sentiments...\")\n",
                    "results = []\n",
                    "\n",
                    "for idx, text in enumerate(df[text_column]):\n",
                    "    if pd.isna(text) or str(text).strip() == '':\n",
                    "        sentiment, confidence = 'neutral', 0.0\n",
                    "    else:\n",
                    "        sentiment, confidence = analyze_sentiment(str(text))\n",
                    "    \n",
                    "    results.append({\n",
                    "        'text': text,\n",
                    "        'sentiment': sentiment,\n",
                    "        'confidence': confidence\n",
                    "    })\n",
                    "    \n",
                    "    # Show progress\n",
                    "    if (idx + 1) % 10 == 0:\n",
                    "        print(f\"Progress: {idx + 1}/{len(df)}\")\n",
                    "\n",
                    "print(\"✅ Sentiment analysis completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "create-results"},
                "source": ["## 📊 Create Results DataFrame"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "results-df"},
                "outputs": [],
                "source": [
                    "# Create results DataFrame\n",
                    "results_df = pd.DataFrame(results)\n",
                    "\n",
                    "# Add original data\n",
                    "for col in df.columns:\n",
                    "    if col != text_column:\n",
                    "        results_df[col] = df[col]\n",
                    "\n",
                    "# Reorder columns\n",
                    "cols = ['text', 'sentiment', 'confidence'] + [col for col in df.columns if col != text_column]\n",
                    "results_df = results_df[cols]\n",
                    "\n",
                    "print(\"📊 Results DataFrame created\")\n",
                    "print(f\"\\n📈 Sentiment distribution:\")\n",
                    "print(results_df['sentiment'].value_counts())\n",
                    "print(f\"\\n🔍 Sample results:\")\n",
                    "results_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "visualization"},
                "source": ["## 🎨 Create Visualizations"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "charts"},
                "outputs": [],
                "source": [
                    "# Create charts\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                    "\n",
                    "# Pie chart\n",
                    "sentiment_counts = results_df['sentiment'].value_counts()\n",
                    "ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')\n",
                    "ax1.set_title('Sentiment Distribution', fontweight='bold')\n",
                    "\n",
                    "# Bar chart\n",
                    "ax2.bar(sentiment_counts.index, sentiment_counts.values)\n",
                    "ax2.set_title('Sentiment Counts', fontweight='bold')\n",
                    "ax2.set_ylabel('Count')\n",
                    "for i, v in enumerate(sentiment_counts.values):\n",
                    "    ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')\n",
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
                    "output_file = f'english_sentiment_results_{pd.Timestamp.now().strftime(\"%Y%m%d_%H%M%S\")}.xlsx'\n",
                    "results_df.to_excel(output_file, index=False)\n",
                    "files.download(output_file)\n",
                    "print(f\"✅ Results saved to: {output_file}\")\n",
                    "print(\"📥 File download started!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "summary"},
                "source": [
                    "## 🎉 Summary\n\n",
                    "✅ **English sentiment analysis completed successfully!**\n\n",
                    "### Model Used:\n",
                    "- **cardiffnlp/twitter-roberta-base-sentiment-latest** (Primary model)\n\n",
                    "### Alternative Models Available:\n",
                    "- **nlptown/bert-base-multilingual-uncased-sentiment** - Multilingual sentiment\n",
                    "- **distilbert-base-uncased-finetuned-sst-2-english** - Fast and efficient\n\n",
                    "### To Try Different Models:\n",
                    "1. Go to the 'Load Model' cell above\n",
                    "2. Comment out the current model_name\n",
                    "3. Uncomment one of the alternative models\n",
                    "4. Re-run the analysis\n\n",
                    "### Results:\n",
                    "- **Total texts analyzed**: Check the results above\n",
                    "- **Sentiment distribution**: Check the results above\n",
                    "- **Average confidence**: Check the results above\n\n",
                    "---\n\n",
                    "**💡 Tip**: Different models may give slightly different results. Try multiple models for best accuracy!"
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
    
    print("✅ Enhanced English notebook created!")

def create_chinese_notebook():
    """Create Chinese sentiment analysis notebook with model alternatives"""
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
                "metadata": {"id": "models-info"},
                "source": [
                    "## 🤖 Available Chinese Models\n\n",
                    "### Primary Model (Current):\n",
                    "- **IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment** - Best for Chinese sentiment\n\n",
                    "### Alternative Models:\n",
                    "- **IDEAL-Future/bert-base-chinese-finetuned-douban-movie** - Movie review sentiment\n",
                    "- **hfl/chinese-roberta-wwm-ext** - General purpose Chinese model\n\n",
                    "### Model Selection:\n",
                    "You can change the `model_name` variable below to use different models."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "install"},
                "source": ["## 📦 Install Required Packages"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install-packages"},
                "outputs": [],
                "source": [
                    "# Install required packages\n",
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
                    "print('✅ Libraries loaded successfully')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "model"},
                "source": ["## 🤖 Load Chinese Sentiment Analysis Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "load-model"},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 🚀 Chinese Sentiment Analysis Models\n",
                    "# =============================================================================\n",
                    "\n",
                    "# Primary model (best for Chinese sentiment)\n",
                    "model_name = 'IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment'\n",
                    "\n",
                    "# Alternative models (uncomment to use)\n",
                    "# model_name = 'IDEAL-Future/bert-base-chinese-finetuned-douban-movie'  # Movie reviews\n",
                    "# model_name = 'hfl/chinese-roberta-wwm-ext'                           # General purpose\n",
                    "\n",
                    "print(f\"🔄 Loading model: {model_name}\")\n",
                    "print(\"\\n📋 Model information:\")\n",
                    "print(f\"   - Primary: IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment\")\n",
                    "print(f\"   - Alternative: IDEAL-Future/bert-base-chinese-finetuned-douban-movie\")\n",
                    "print(f\"   - Fallback: hfl/chinese-roberta-wwm-ext\")\n",
                    "\n",
                    "# Load tokenizer and model\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                    "\n",
                    "# Use GPU if available\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "model.to(device)\n",
                    "model.eval()\n",
                    "\n",
                    "print(f\"\\n✅ Model loaded successfully on {device}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "upload"},
                "source": ["## 📁 Upload Excel File"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload-file"},
                "outputs": [],
                "source": [
                    "# Upload Excel file\n",
                    "print(\"📁 Please upload your Excel file (.xlsx)\")\n",
                    "uploaded = files.upload()\n",
                    "\n",
                    "# Get the first uploaded file\n",
                    "filename = list(uploaded.keys())[0]\n",
                    "print(f\"✅ File uploaded: {filename}\")\n",
                    "\n",
                    "# Read Excel file\n",
                    "df = pd.read_excel(BytesIO(uploaded[filename]))\n",
                    "print(f\"📊 Data loaded: {len(df)} rows, {len(df.columns)} columns\")\n",
                    "print(\"\\n📋 Columns:\", list(df.columns))\n",
                    "print(\"\\n🔍 First few rows:\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "select-column"},
                "source": ["## 📝 Select Text Column"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "column-selection"},
                "outputs": [],
                "source": [
                    "# Set text column name (modify this to match your column name)\n",
                    "text_column = 'text'  # Change this to your actual column name\n",
                    "\n",
                    "# Common column names for Chinese text\n",
                    "# text_column = 'comment'    # 评论\n",
                    "# text_column = 'review'     # 评论\n",
                    "# text_column = '评论'       # Chinese column name\n",
                    "# text_column = '内容'       # Chinese column name\n",
                    "\n",
                    "# Check if column exists\n",
                    "if text_column not in df.columns:\n",
                    "    print(f\"❌ Column '{text_column}' not found. Available columns: {list(df.columns)}\")\n",
                    "    print(\"\\n💡 Please modify the 'text_column' variable above to match your column name\")\n",
                    "    print(\"\\n🔍 Common Chinese column names: comment, review, 评论, 内容, text\")\n",
                    "else:\n",
                    "    print(f\"✅ Text column selected: {text_column}\")\n",
                    "    print(f\"📝 Sample text: {df[text_column].iloc[0]}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "sentiment-analysis"},
                "source": ["## 🧠 Perform Sentiment Analysis"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "analysis"},
                "outputs": [],
                "source": [
                    "def analyze_sentiment(text):\n",
                    "    \"\"\"Analyze sentiment of Chinese text\"\"\"\n",
                    "    try:\n",
                    "        # Tokenize text\n",
                    "        inputs = tokenizer(text, return_tensors=\"pt\", truncation=True, max_length=128, padding=True)\n",
                    "        inputs = {k: v.to(device) for k, v in inputs.items()}\n",
                    "        \n",
                    "        # Get prediction\n",
                    "        with torch.no_grad():\n",
                    "            outputs = model(**inputs)\n",
                    "            probabilities = torch.softmax(outputs.logits, dim=1)\n",
                    "            \n",
                    "        # Get sentiment and confidence\n",
                    "        sentiment_id = torch.argmax(probabilities, dim=1).item()\n",
                    "        confidence = probabilities[0][sentiment_id].item()\n",
                    "        \n",
                    "        # Map sentiment ID to label (Chinese models typically use 2 classes)\n",
                    "        sentiment_labels = ['negative', 'positive']\n",
                    "        sentiment = sentiment_labels[sentiment_id]\n",
                    "        \n",
                    "        return sentiment, confidence\n",
                    "    except Exception as e:\n",
                    "        return 'error', 0.0\n",
                    "\n",
                    "# Analyze each text\n",
                    "print(\"🔄 Analyzing sentiments...\")\n",
                    "results = []\n",
                    "\n",
                    "for idx, text in enumerate(df[text_column]):\n",
                    "    if pd.isna(text) or str(text).strip() == '':\n",
                    "        sentiment, confidence = 'neutral', 0.0\n",
                    "    else:\n",
                    "        sentiment, confidence = analyze_sentiment(str(text))\n",
                    "    \n",
                    "    results.append({\n",
                    "        'text': text,\n",
                    "        'sentiment': sentiment,\n",
                    "        'confidence': confidence\n",
                    "    })\n",
                    "    \n",
                    "    # Show progress\n",
                    "    if (idx + 1) % 10 == 0:\n",
                    "        print(f\"Progress: {idx + 1}/{len(df)}\")\n",
                    "\n",
                    "print(\"✅ Sentiment analysis completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "create-results"},
                "source": ["## 📊 Create Results DataFrame"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "results-df"},
                "outputs": [],
                "source": [
                    "# Create results DataFrame\n",
                    "results_df = pd.DataFrame(results)\n",
                    "\n",
                    "# Add original data\n",
                    "for col in df.columns:\n",
                    "    if col != text_column:\n",
                    "        results_df[col] = df[col]\n",
                    "\n",
                    "# Reorder columns\n",
                    "cols = ['text', 'sentiment', 'confidence'] + [col for col in df.columns if col != text_column]\n",
                    "results_df = results_df[cols]\n",
                    "\n",
                    "print(\"📊 Results DataFrame created\")\n",
                    "print(f\"\\n📈 Sentiment distribution:\")\n",
                    "print(results_df['sentiment'].value_counts())\n",
                    "print(f\"\\n🔍 Sample results:\")\n",
                    "results_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "visualization"},
                "source": ["## 🎨 Create Visualizations"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "charts"},
                "outputs": [],
                "source": [
                    "# Create charts\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                    "\n",
                    "# Pie chart\n",
                    "sentiment_counts = results_df['sentiment'].value_counts()\n",
                    "ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')\n",
                    "ax1.set_title('Sentiment Distribution', fontweight='bold')\n",
                    "\n",
                    "# Bar chart\n",
                    "ax2.bar(sentiment_counts.index, sentiment_counts.values)\n",
                    "ax2.set_title('Sentiment Counts', fontweight='bold')\n",
                    "ax2.set_ylabel('Count')\n",
                    "for i, v in enumerate(sentiment_counts.values):\n",
                    "    ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')\n",
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
                    "output_file = f'chinese_sentiment_results_{pd.Timestamp.now().strftime(\"%Y%m%d_%H%M%S\")}.xlsx'\n",
                    "results_df.to_excel(output_file, index=False)\n",
                    "files.download(output_file)\n",
                    "print(f\"✅ Results saved to: {output_file}\")\n",
                    "print(\"📥 File download started!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "summary"},
                "source": [
                    "## 🎉 Summary\n\n",
                    "✅ **Chinese sentiment analysis completed successfully!**\n\n",
                    "### Model Used:\n",
                    "- **IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment** (Primary model)\n\n",
                    "### Alternative Models Available:\n",
                    "- **IDEAL-Future/bert-base-chinese-finetuned-douban-movie** - Movie review sentiment\n",
                    "- **hfl/chinese-roberta-wwm-ext** - General purpose Chinese model\n\n",
                    "### To Try Different Models:\n",
                    "1. Go to the 'Load Model' cell above\n",
                    "2. Comment out the current model_name\n",
                    "3. Uncomment one of the alternative models\n",
                    "4. Re-run the analysis\n\n",
                    "### Results:\n",
                    "- **Total texts analyzed**: Check the results above\n",
                    "- **Sentiment distribution**: Check the results above\n",
                    "- **Average confidence**: Check the results above\n\n",
                    "---\n\n",
                    "**💡 Tip**: Different models may give slightly different results. Try multiple models for best accuracy!"
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
    
    print("✅ Enhanced Chinese notebook created!")

def create_japanese_notebook():
    """Create Japanese sentiment analysis notebook with model alternatives"""
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
                "metadata": {"id": "models-info"},
                "source": [
                    "## 🤖 Available Japanese Models\n\n",
                    "### Primary Model (Current):\n",
                    "- **cl-tohoku/bert-base-japanese-v3** - Best for Japanese text\n\n",
                    "### Alternative Models:\n",
                    "- **rinna/japanese-roberta-base** - High accuracy alternative\n",
                    "- **megagonlabs/roberta-base-japanese-sentiment** - Sentiment specific\n\n",
                    "### Model Selection:\n",
                    "You can change the `model_name` variable below to use different models."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "install"},
                "source": ["## 📦 Install Required Packages"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install-packages"},
                "outputs": [],
                "source": [
                    "# Install required packages\n",
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
                    "print('✅ Libraries loaded successfully')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "model"},
                "source": ["## 🤖 Load Japanese Sentiment Analysis Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "load-model"},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 🚀 Japanese Sentiment Analysis Models\n",
                    "# =============================================================================\n",
                    "\n",
                    "# Primary model (best for Japanese text)\n",
                    "model_name = 'cl-tohoku/bert-base-japanese-v3'\n",
                    "\n",
                    "# Alternative models (uncomment to use)\n",
                    "# model_name = 'rinna/japanese-roberta-base'                    # High accuracy\n",
                    "# model_name = 'megagonlabs/roberta-base-japanese-sentiment'   # Sentiment specific\n",
                    "\n",
                    "print(f\"🔄 Loading model: {model_name}\")\n",
                    "print(\"\\n📋 Model information:\")\n",
                    "print(f\"   - Primary: cl-tohoku/bert-base-japanese-v3\")\n",
                    "print(f\"   - Alternative: rinna/japanese-roberta-base\")\n",
                    "print(f\"   - Fallback: megagonlabs/roberta-base-japanese-sentiment\")\n",
                    "\n",
                    "# Load tokenizer and model\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                    "\n",
                    "# Use GPU if available\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "model.to(device)\n",
                    "model.eval()\n",
                    "\n",
                    "print(f\"\\n✅ Model loaded successfully on {device}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "upload"},
                "source": ["## 📁 Upload Excel File"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload-file"},
                "outputs": [],
                "source": [
                    "# Upload Excel file\n",
                    "print(\"📁 Please upload your Excel file (.xlsx)\")\n",
                    "uploaded = files.upload()\n",
                    "\n",
                    "# Get the first uploaded file\n",
                    "filename = list(uploaded.keys())[0]\n",
                    "print(f\"✅ File uploaded: {filename}\")\n",
                    "\n",
                    "# Read Excel file\n",
                    "df = pd.read_excel(BytesIO(uploaded[filename]))\n",
                    "print(f\"📊 Data loaded: {len(df)} rows, {len(df.columns)} columns\")\n",
                    "print(\"\\n📋 Columns:\", list(df.columns))\n",
                    "print(\"\\n🔍 First few rows:\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "select-column"},
                "source": ["## 📝 Select Text Column"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "column-selection"},
                "outputs": [],
                "source": [
                    "# Set text column name (modify this to match your column name)\n",
                    "text_column = 'text'  # Change this to your actual column name\n",
                    "\n",
                    "# Common column names for Japanese text\n",
                    "# text_column = 'comment'    # コメント\n",
                    "# text_column = 'review'     # レビュー\n",
                    "# text_column = 'コメント'     # Japanese column name\n",
                    "# text_column = 'レビュー'     # Japanese column name\n",
                    "\n",
                    "# Check if column exists\n",
                    "if text_column not in df.columns:\n",
                    "    print(f\"❌ Column '{text_column}' not found. Available columns: {list(df.columns)}\")\n",
                    "    print(\"\\n💡 Please modify the 'text_column' variable above to match your column name\")\n",
                    "    print(\"\\n🔍 Common Japanese column names: comment, review, コメント, レビュー, text\")\n",
                    "else:\n",
                    "    print(f\"✅ Text column selected: {text_column}\")\n",
                    "    print(f\"📝 Sample text: {df[text_column].iloc[0]}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "sentiment-analysis"},
                "source": ["## 🧠 Perform Sentiment Analysis"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "analysis"},
                "outputs": [],
                "source": [
                    "def analyze_sentiment(text):\n",
                    "    \"\"\"Analyze sentiment of Japanese text\"\"\"\n",
                    "    try:\n",
                    "        # Tokenize text\n",
                    "        inputs = tokenizer(text, return_tensors=\"pt\", truncation=True, max_length=128, padding=True)\n",
                    "        inputs = {k: v.to(device) for k, v in inputs.items()}\n",
                    "        \n",
                    "        # Get prediction\n",
                    "        with torch.no_grad():\n",
                    "            outputs = model(**inputs)\n",
                    "            probabilities = torch.softmax(outputs.logits, dim=1)\n",
                    "            \n",
                    "        # Get sentiment and confidence\n",
                    "        sentiment_id = torch.argmax(probabilities, dim=1).item()\n",
                    "        confidence = probabilities[0][sentiment_id].item()\n",
                    "        \n",
                    "        # Map sentiment ID to label (Japanese models typically use 2 classes)\n",
                    "        sentiment_labels = ['negative', 'positive']\n",
                    "        sentiment = sentiment_labels[sentiment_id]\n",
                    "        \n",
                    "        return sentiment, confidence\n",
                    "    except Exception as e:\n",
                    "        return 'error', 0.0\n",
                    "\n",
                    "# Analyze each text\n",
                    "print(\"🔄 Analyzing sentiments...\")\n",
                    "results = []\n",
                    "\n",
                    "for idx, text in enumerate(df[text_column]):\n",
                    "    if pd.isna(text) or str(text).strip() == '':\n",
                    "        sentiment, confidence = 'neutral', 0.0\n",
                    "    else:\n",
                    "        sentiment, confidence = analyze_sentiment(str(text))\n",
                    "    \n",
                    "    results.append({\n",
                    "        'text': text,\n",
                    "        'sentiment': sentiment,\n",
                    "        'confidence': confidence\n",
                    "    })\n",
                    "    \n",
                    "    # Show progress\n",
                    "    if (idx + 1) % 10 == 0:\n",
                    "        print(f\"Progress: {idx + 1}/{len(df)}\")\n",
                    "\n",
                    "print(\"✅ Sentiment analysis completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "create-results"},
                "source": ["## 📊 Create Results DataFrame"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "results-df"},
                "outputs": [],
                "source": [
                    "# Create results DataFrame\n",
                    "results_df = pd.DataFrame(results)\n",
                    "\n",
                    "# Add original data\n",
                    "for col in df.columns:\n",
                    "    if col != text_column:\n",
                    "        results_df[col] = df[col]\n",
                    "\n",
                    "# Reorder columns\n",
                    "cols = ['text', 'sentiment', 'confidence'] + [col for col in df.columns if col != text_column]\n",
                    "results_df = results_df[cols]\n",
                    "\n",
                    "print(\"📊 Results DataFrame created\")\n",
                    "print(f\"\\n📈 Sentiment distribution:\")\n",
                    "print(results_df['sentiment'].value_counts())\n",
                    "print(f\"\\n🔍 Sample results:\")\n",
                    "results_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "visualization"},
                "source": ["## 🎨 Create Visualizations"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "charts"},
                "outputs": [],
                "source": [
                    "# Create charts\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                    "\n",
                    "# Pie chart\n",
                    "sentiment_counts = results_df['sentiment'].value_counts()\n",
                    "ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')\n",
                    "ax1.set_title('Sentiment Distribution', fontweight='bold')\n",
                    "\n",
                    "# Bar chart\n",
                    "ax2.bar(sentiment_counts.index, sentiment_counts.values)\n",
                    "ax2.set_title('Sentiment Counts', fontweight='bold')\n",
                    "ax2.set_ylabel('Count')\n",
                    "for i, v in enumerate(sentiment_counts.values):\n",
                    "    ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')\n",
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
                    "output_file = f'japanese_sentiment_results_{pd.Timestamp.now().strftime(\"%Y%m%d_%H%M%S\")}.xlsx'\n",
                    "results_df.to_excel(output_file, index=False)\n",
                    "files.download(output_file)\n",
                    "print(f\"✅ Results saved to: {output_file}\")\n",
                    "print(\"📥 File download started!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "summary"},
                "source": [
                    "## 🎉 Summary\n\n",
                    "✅ **Japanese sentiment analysis completed successfully!**\n\n",
                    "### Model Used:\n",
                    "- **cl-tohoku/bert-base-japanese-v3** (Primary model)\n\n",
                    "### Alternative Models Available:\n",
                    "- **rinna/japanese-roberta-base** - High accuracy alternative\n",
                    "- **megagonlabs/roberta-base-japanese-sentiment** - Sentiment specific\n\n",
                    "### To Try Different Models:\n",
                    "1. Go to the 'Load Model' cell above\n",
                    "2. Comment out the current model_name\n",
                    "3. Uncomment one of the alternative models\n",
                    "4. Re-run the analysis\n\n",
                    "### Results:\n",
                    "- **Total texts analyzed**: Check the results above\n",
                    "- **Sentiment distribution**: Check the results above\n",
                    "- **Average confidence**: Check the results above\n\n",
                    "---\n\n",
                    "**💡 Tip**: Different models may give slightly different results. Try multiple models for best accuracy!"
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
    
    print("✅ Enhanced Japanese notebook created!")

def main():
    """Create enhanced notebooks with model information"""
    print("🚀 Creating enhanced Colab notebooks with model alternatives...")
    
    # Create enhanced notebooks
    create_korean_notebook()
    create_english_notebook()
    create_chinese_notebook()
    create_japanese_notebook()
    
    print("\n🎉 Enhanced notebooks created successfully!")
    print("\n📁 Files created:")
    print("- korean_sentiment_analysis.ipynb (with model alternatives)")
    print("- english_sentiment_analysis.ipynb (with model alternatives)")
    print("- chinese_sentiment_analysis.ipynb (with model alternatives)")
    print("- japanese_sentiment_analysis.ipynb (with model alternatives)")
    print("\n💡 These notebooks now include:")
    print("   - Primary model information")
    print("   - Alternative model options")
    print("   - Easy model switching instructions")
    print("   - Enhanced documentation")
    print("\n🔧 Model alternatives for each language:")
    print("   🇰🇷 Korean: snunlp/KR-FinBert-SC (primary), beomi/KcELECTRA-base-v2022, klue/roberta-base")
    print("   🇺🇸 English: cardiffnlp/twitter-roberta-base-sentiment-latest (primary), nlptown/bert-base-multilingual-uncased-sentiment, distilbert-base-uncased-finetuned-sst-2-english")
    print("   🇨🇳 Chinese: IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment (primary), IDEAL-Future/bert-base-chinese-finetuned-douban-movie, hfl/chinese-roberta-wwm-ext")
    print("   🇯🇵 Japanese: cl-tohoku/bert-base-japanese-v3 (primary), rinna/japanese-roberta-base, megagonlabs/roberta-base-japanese-sentiment")

if __name__ == "__main__":
    main()
