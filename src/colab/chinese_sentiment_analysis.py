# -*- coding: utf-8 -*-
"""
🇨🇳 중국어 감성분석 (Chinese Sentiment Analysis)
Google Colab에서 실행 가능한 중국어 텍스트 감성분석 시스템

📋 기능:
- Excel 파일에서 중국어 텍스트 읽기
- 중국어 전용 감성분석 모델 사용
- 결과를 Excel 파일로 저장

🚀 사용법:
1. Excel 파일을 업로드 (텍스트 컬럼 포함)
2. 코드 실행
3. 결과 다운로드
"""

# 필요한 패키지 설치
# !pip install torch transformers pandas openpyxl matplotlib seaborn wordcloud jieba

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import warnings
import logging
from google.colab import files
from io import BytesIO
import re
import jieba

# 경고 메시지 차단
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

# 중국어 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("✅ 라이브러리 로드 완료")

def load_chinese_model():
    """중국어 감성분석 모델 로드"""
    # 중국어 감성분석 모델 (감성분석 전용)
    model_name = "IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment"
    
    print(f"🔄 모델 로딩 중: {model_name}")
    
    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # GPU 사용 가능시 GPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"✅ 모델 로드 완료! 디바이스: {device}")
    print(f"📊 모델: {model_name}")
    
    return tokenizer, model, device

def upload_excel_file():
    """Excel 파일 업로드"""
    print("📁 Excel 파일을 업로드해주세요...")
    uploaded = files.upload()
    
    if not uploaded:
        print("❌ 파일이 업로드되지 않았습니다.")
        return None, None
    
    # 첫 번째 파일 사용
    filename = list(uploaded.keys())[0]
    print(f"✅ 파일 업로드 완료: {filename}")
    
    # Excel 파일 읽기
    try:
        df = pd.read_excel(BytesIO(uploaded[filename]))
        print(f"📊 Excel 파일 읽기 성공: {len(df)} 행, {len(df.columns)} 컬럼")
        print(f"📋 컬럼명: {list(df.columns)}")
        print("\n🔍 데이터 미리보기:")
        print(df.head())
        return df, filename
    except Exception as e:
        print(f"❌ Excel 파일 읽기 실패: {str(e)}")
        return None, None

def find_text_column(df):
    """텍스트 컬럼 찾기"""
    text_columns = ['comment', '评论', 'review', 'text', 'comments', '文本', '内容', '留言']
    text_column = None
    
    for col in df.columns:
        if col.lower() in [tc.lower() for tc in text_columns]:
            text_column = col
            break
    
    if text_column is None:
        # 컬럼 내용으로 추정
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_data = df[col].dropna().head(5)
                if len(sample_data) > 0:
                    avg_length = sample_data.astype(str).str.len().mean()
                    if avg_length > 5:  # 중국어는 짧아도 의미있음
                        text_column = col
                        break
    
    if text_column:
        print(f"✅ 텍스트 컬럼 선택: {text_column}")
        print(f"📝 텍스트 개수: {len(df[text_column].dropna())}")
        print("\n🔍 텍스트 샘플:")
        for i, text in enumerate(df[text_column].dropna().head(3)):
            print(f"{i+1}. {text[:100]}{'...' if len(str(text)) > 100 else ''}")
    else:
        print("❌ 텍스트 컬럼을 찾을 수 없습니다.")
        print("사용 가능한 컬럼:")
        for col in df.columns:
            print(f"- {col} ({df[col].dtype})")
    
    return text_column

def analyze_sentiment(texts, tokenizer, model, device):
    """중국어 텍스트 감성분석"""
    results = []
    batch_size = 16
    
    # 라벨 정의 (중국어 모델에 따라 다를 수 있음)
    labels = ["negative", "neutral", "positive"]
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 토크나이징
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          return_tensors="pt", max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            for j, prob in enumerate(probs):
                max_idx = torch.argmax(prob).item()
                confidence = prob[max_idx].item()
                
                results.append({
                    'text': batch_texts[j],
                    'sentiment_label': labels[max_idx],
                    'sentiment_confidence': confidence,
                    'negative_prob': prob[0].item(),
                    'neutral_prob': prob[1].item(),
                    'positive_prob': prob[2].item()
                })
    
    return results

def create_visualizations(results_df):
    """시각화 생성"""
    if results_df.empty:
        print("❌ 시각화할 결과가 없습니다.")
        return
    
    # 1. 감성 분포 막대 차트
    plt.figure(figsize=(10, 6))
    sentiment_counts = results_df['sentiment_label'].value_counts()
    colors = ['#F44336', '#9E9E9E', '#4CAF50']  # 빨강, 회색, 초록
    
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.8)
    plt.title('중국어 텍스트 감성분석 결과', fontsize=16, fontweight='bold')
    plt.xlabel('감성', fontsize=12)
    plt.ylabel('텍스트 수', fontsize=12)
    
    # 값 표시
    for bar, count in zip(bars, sentiment_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 2. 감성 분포 파이 차트
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('감성 분포 비율', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.show()
    
    # 3. 신뢰도 분포 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['sentiment_confidence'], bins=20, alpha=0.7, color='orange')
    plt.title('감성분석 신뢰도 분포', fontsize=16, fontweight='bold')
    plt.xlabel('신뢰도', fontsize=12)
    plt.ylabel('빈도', fontsize=12)
    
    # 평균선 추가
    mean_confidence = results_df['sentiment_confidence'].mean()
    plt.axvline(mean_confidence, color='red', linestyle='--', 
                label=f'평균: {mean_confidence:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 4. 워드클라우드 (긍정적 텍스트만)
    positive_texts = results_df[results_df['sentiment_label'] == 'positive']['text'].tolist()
    if positive_texts:
        combined_text = ' '.join(positive_texts)
        
        # 중국어 토큰화 및 전처리
        try:
            # jieba로 중국어 토큰화
            words = []
            for text in positive_texts:
                words.extend(jieba.cut(text))
            
            processed_text = ' '.join(words)
            
            if processed_text.strip():
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    min_font_size=12,
                    relative_scaling=0.5,
                    colormap='viridis'
                ).generate(processed_text)
                
                plt.figure(figsize=(16, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('긍정적 텍스트 워드클라우드', fontsize=20, fontweight='bold')
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"워드클라우드 생성 실패: {str(e)}")

def save_results(df, results_df, original_filename):
    """결과 저장"""
    if results_df.empty:
        print("❌ 저장할 결과가 없습니다.")
        return
    
    # 원본 데이터와 결과 합치기
    final_df = df.copy()
    
    # 결과 컬럼 추가
    for col in results_df.columns:
        if col != 'text':  # text 컬럼은 제외 (중복)
            final_df[col] = results_df[col]
    
    # 결과 파일명 생성
    output_filename = f"chinese_sentiment_analysis_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Excel 파일로 저장
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # 전체 결과
        final_df.to_excel(writer, sheet_name='情感分析_结果', index=False)
        
        # 감성별 요약
        sentiment_summary = results_df['sentiment_label'].value_counts().reset_index()
        sentiment_summary.columns = ['情感', '数量']
        sentiment_summary['比例(%)'] = (sentiment_summary['数量'] / len(results_df) * 100).round(1)
        sentiment_summary.to_excel(writer, sheet_name='情感_摘要', index=False)
        
        # 상세 결과
        results_df.to_excel(writer, sheet_name='详细_结果', index=False)
    
    print(f"✅ 결과 저장 완료: {output_filename}")
    print(f"📊 총 {len(final_df)} 행의 데이터가 분석되었습니다.")
    
    # 파일 다운로드
    files.download(output_filename)
    
    # 결과 미리보기
    print("\n🔍 최종 결과 미리보기:")
    print(final_df.head())
    
    return output_filename

def main():
    """메인 실행 함수"""
    print("🇨🇳 중국어 감성분석 시스템 시작!")
    print("=" * 50)
    
    # 1. 모델 로드
    tokenizer, model, device = load_chinese_model()
    
    # 2. Excel 파일 업로드
    df, filename = upload_excel_file()
    if df is None:
        return
    
    # 3. 텍스트 컬럼 찾기
    text_column = find_text_column(df)
    if text_column is None:
        return
    
    # 4. 감성분석 실행
    print("\n🔄 감성분석 실행 중...")
    
    # 텍스트 데이터 준비
    texts = df[text_column].dropna().astype(str).tolist()
    
    if texts:
        # 감성분석 실행
        results = analyze_sentiment(texts, tokenizer, model, device)
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(results)
        
        print(f"✅ 감성분석 완료: {len(results_df)} 텍스트 분석")
        
        # 결과 미리보기
        print("\n🔍 감성분석 결과 미리보기:")
        print(results_df.head())
        
        # 감성 분포
        sentiment_counts = results_df['sentiment_label'].value_counts()
        print("\n📊 감성 분포:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(results_df)) * 100
            print(f"- {sentiment}: {count}개 ({percentage:.1f}%)")
        
        # 5. 시각화
        print("\n📊 시각화 생성 중...")
        create_visualizations(results_df)
        
        # 6. 결과 저장
        print("\n💾 결과 저장 중...")
        output_filename = save_results(df, results_df, filename)
        
        # 7. 요약
        print("\n" + "=" * 50)
        print("🎯 중국어 감성분석 완료!")
        print(f"📊 총 텍스트 수: {len(results_df)}")
        print(f"📊 평균 신뢰도: {results_df['sentiment_confidence'].mean():.3f}")
        print(f"📁 다운로드 파일: {output_filename}")
        print("=" * 50)
        
    else:
        print("❌ 분석할 텍스트가 없습니다.")

if __name__ == "__main__":
    main()
