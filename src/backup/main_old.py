# ─────────────────────────────────────────────────────────────
# ✅ 파일 경로 설정 (PC 환경)
# ─────────────────────────────────────────────────────────────
import os

# 현재 스크립트가 있는 디렉토리를 기준으로 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'c_movie_review.xlsx')
save_path = os.path.join(current_dir, 'c_movie_review_result.xlsx')
#file_path = os.path.join(current_dir, 'k_movie_review.xlsx')
#save_path = os.path.join(current_dir, 'k_movie_review_result.xlsx')

# ─────────────────────────────────────────────────────────────
# ✅ 폰트 설정 (Windows 환경)
# ─────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Windows 기본 폰트 사용 (폰트 경고 제거)
plt.rcParams['font.family'] = ['Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

# 폰트 경로 설정 (Windows 환경)
font_path_ko = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
font_path_zh = 'C:/Windows/Fonts/simsun.ttc'  # SimSun

# 폰트 파일이 존재하는지 확인하고 설정
if os.path.exists(font_path_ko):
    nanum_font = fm.FontProperties(fname=font_path_ko, size=16)
else:
    # 기본 폰트 사용
    nanum_font = fm.FontProperties(size=16)

# ─────────────────────────────────────────────────────────────
# ✅ 경고 메시지 끄기
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# matplotlib 폰트 경고 완전 차단
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────
# ✅ 라이브러리
# ─────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────────────────────
# ✅ 한국어 모델 (업데이트)
# ─────────────────────────────────────────────────────────────
model_name_ko = "snunlp/KR-FinBert-SC"  # 가장 정확한 한국어 감정분석 모델

# 🏆 한국어 감정분석 모델 순위
# 1. snunlp/KR-FinBert-SC (금융 텍스트, 일반 텍스트 모두 우수)
# 2. beomi/KcELECTRA-base-v2022 (최신 업데이트)
# 3. klue/roberta-base (KLUE 프로젝트)
# 4. alsgyu/sentiment-analysis-fine-tuned-model (이전 사용 모델)
tokenizer_ko = AutoTokenizer.from_pretrained(model_name_ko)
model_ko = AutoModelForSequenceClassification.from_pretrained(model_name_ko)

# ─────────────────────────────────────────────────────────────
# ✅ 중국어 모델 (감성 분석 전용)
# ─────────────────────────────────────────────────────────────
model_name_zh = "IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment"

# 🎥 중국어 영화 리뷰용 모델 리스트 (감성 분석 전용)
# - IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment (감성 분석 전용, 추천)
# - hfl/chinese-roberta-wwm-ext (기본 모델, 감성 분석 부적합)
# - IDEAL-Future/bert-base-chinese-finetuned-douban-movie (영화 리뷰 전용)

tokenizer_zh = AutoTokenizer.from_pretrained(model_name_zh)
model_zh = AutoModelForSequenceClassification.from_pretrained(model_name_zh)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ko.to(device)
model_zh.to(device)

labels = ["negative", "neutral", "positive"]

# ─────────────────────────────────────────────────────────────
# ✅ 감성 분석 함수
# ─────────────────────────────────────────────────────────────
def analyze_sentiments(texts, tokenizer, model):
    results = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

        for pred, conf in zip(preds.tolist(), confs.tolist()):
            # 디버깅: 실제 예측값 출력 (처음 5개만)
            if len(results) < 5:
                print(f"예측값: {pred}, 신뢰도: {conf:.3f}")
            
            # 모든 모델에 신뢰도 기반 중립 분류 적용
            if conf < 0.8:  # 신뢰도 임계값을 높임 (0.6 → 0.8)
                label = "neutral"
            elif model == model_zh:
                # 중국어 모델 라벨 매핑
                if pred == 0:
                    label = "negative"
                elif pred == 1:
                    label = "positive"
                else:
                    label = "neutral"
            else:
                # 한국어 모델 라벨 매핑 (KR-FinBert-SC는 0:부정, 1:긍정)
                if pred == 0:
                    label = "negative"
                elif pred == 1:
                    # 긍정 예측이지만 일정 비율은 중립으로 조정
                    if conf < 0.95:  # 매우 높은 신뢰도가 아니면 중립
                        label = "neutral"
                    else:
                        label = "positive"
                else:
                    label = "neutral"
                
            results.append({
                'label': label,
                'confidence': round(conf, 2)
            })

    return results

# ─────────────────────────────────────────────────────────────
# ✅ 통합 시각화 생성 함수
# ─────────────────────────────────────────────────────────────
def create_integrated_visualizations(all_comments, all_labels, sheet_stats, current_dir):
    """전체 데이터를 통합하여 시각화 생성"""
    print("\n🎨 통합 시각화 생성 중...")
    
    # 1. 전체 워드클라우드
    all_text = ' '.join(all_comments)
    
    # 한국어/중국어 혼합 텍스트 전처리
    all_text = re.sub(r'[^가-힣一-龯a-zA-Z0-9\s]', '', all_text)
    
    if all_text.strip():
        # 한국어 폰트 경로 확인 및 설정
        if os.path.exists(font_path_ko):
            wordcloud_font = font_path_ko
            print(f"✅ 한국어 폰트 사용: {wordcloud_font}")
        else:
            wordcloud_font = None
            print("⚠️ 한국어 폰트를 찾을 수 없어 기본 폰트 사용")
        
        wordcloud_params = {
            'width': 1600,
            'height': 800,
            'background_color': 'white',
            'max_words': 300,
            'min_font_size': 12,
            'relative_scaling': 0.5,
            'colormap': 'plasma',
            'collocations': False,
            'font_path': wordcloud_font  # 한국어 폰트 명시적 설정
        }
        
        wordcloud = WordCloud(**wordcloud_params).generate(all_text)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("전체 영화 리뷰 워드클라우드", fontproperties=nanum_font, fontsize=20)
        
        wordcloud_path = os.path.join(current_dir, "전체_워드클라우드.png")
        plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
        print(f"💾 전체 워드클라우드 저장됨: 전체_워드클라우드.png")
        plt.show()
    
    # 2. 전체 감성 분석 통계
    total_label_counts = pd.Series(all_labels).value_counts()
    total_count = len(all_labels)
    
    print(f"\n📊 전체 감성 분석 통계 (총 {total_count}개 댓글)")
    for label in ['positive', 'neutral', 'negative']:
        count = total_label_counts.get(label, 0)
        percent = (count / total_count) * 100 if total_count > 0 else 0
        print(f"- {label}: {count}개 ({percent:.1f}%)")
    
    # 3. 전체 막대 그래프
    plt.figure(figsize=(12, 8))
    total_label_counts.reindex(['positive', 'neutral', 'negative'], fill_value=0).plot(
        kind='bar', color=['lightgreen', 'lightgray', 'lightcoral']
    )
    plt.title("전체 영화 리뷰 감성 분석 통계", fontproperties=nanum_font, fontsize=16)
    plt.ylabel("댓글 수", fontproperties=nanum_font)
    plt.xticks(rotation=0, fontproperties=nanum_font)
    
    bar_path = os.path.join(current_dir, "전체_막대그래프.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"💾 전체 막대 그래프 저장됨: 전체_막대그래프.png")
    plt.show()
    
    # 4. 전체 원형 차트
    plt.figure(figsize=(10, 8))
    plt.pie(total_label_counts, labels=total_label_counts.index, autopct='%1.1f%%', 
            colors=['lightgreen', 'lightgray', 'lightcoral'])
    plt.title("전체 영화 리뷰 감성 분석 비율", fontproperties=nanum_font, fontsize=16)
    
    pie_path = os.path.join(current_dir, "전체_원형차트.png")
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    print(f"💾 전체 원형 차트 저장됨: 전체_원형차트.png")
    plt.show()
    
    # 5. 시트별 비교 차트
    if len(sheet_stats) > 1:
        sheet_names = list(sheet_stats.keys())
        positive_counts = [sheet_stats[name]['positive'] for name in sheet_names]
        neutral_counts = [sheet_stats[name]['neutral'] for name in sheet_names]
        negative_counts = [sheet_stats[name]['negative'] for name in sheet_names]
        
        x = np.arange(len(sheet_names))
        width = 0.25
        
        plt.figure(figsize=(15, 8))
        plt.bar(x - width, positive_counts, width, label='Positive', color='lightgreen')
        plt.bar(x, neutral_counts, width, label='Neutral', color='lightgray')
        plt.bar(x + width, negative_counts, width, label='Negative', color='lightcoral')
        
        plt.xlabel('영화', fontproperties=nanum_font)
        plt.ylabel('댓글 수', fontproperties=nanum_font)
        plt.title('영화별 감성 분석 비교', fontproperties=nanum_font, fontsize=16)
        plt.xticks(x, sheet_names, fontproperties=nanum_font)
        plt.legend()
        plt.tight_layout()
        
        comparison_path = os.path.join(current_dir, "영화별_비교차트.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"💾 영화별 비교 차트 저장됨: 영화별_비교차트.png")
        plt.show()

# ─────────────────────────────────────────────────────────────
# ✅ 시트별 처리 함수
# ─────────────────────────────────────────────────────────────
def process_excel_sheets(file_path, save_path):
    # 현재 스크립트가 있는 디렉토리 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"⚠️ 파일을 찾을 수 없습니다: {file_path}")
        print("📁 현재 디렉토리에 'movie_review.xlsx' 파일이 있는지 확인해주세요.")
        return
    
    xlsx = pd.ExcelFile(file_path)
    sheet_names = xlsx.sheet_names
    print("📄 현재 시트 목록:", sheet_names)

    sheet_dict = {}
    
    # 전체 통계를 위한 데이터 수집
    all_comments = []
    all_labels = []
    all_confidences = []
    sheet_stats = {}

    for sheet in sheet_names:
        print(f"\n🎬 처리 중인 시트: {sheet}")

        try:
            # Read the sheet and assume the header is on the first row (row 0)
            df = pd.read_excel(file_path, sheet_name=sheet, header=0)
            print(f"📄 시트 '{sheet}'의 컬럼 목록:", df.columns.tolist()) # Add this line to print column names
        except Exception as e:
            print(f"⚠️ 시트 '{sheet}' 읽기 실패: {e}")
            continue

        target_column = None
        # Check if any column name matches the target names (case-insensitive)
        for col in df.columns:
            if col.lower() in ['comment', '댓글', 'review', 'text', 'comments']: # Added 'comments' to the list
                target_column = col
                break

        # If column name not found, try accessing by index (assuming 'D' is index 3)
        if target_column is None and len(df.columns) > 3:
             target_column = df.columns[3]
             print(f"⚠️ 댓글 컬럼 이름을 찾을 수 없어 D열 (인덱스 3)의 '{target_column}' 컬럼을 사용합니다.")


        if target_column is None:
            print("⚠️ 댓글 컬럼을 찾을 수 없습니다. 'comment', '댓글', 'review', 'text', 'comments' 중 하나가 필요하거나 D열에 데이터가 없습니다.")
            continue

        comments = df[target_column].dropna().tolist()

        if not comments:
            print("⚠️ 댓글 데이터가 없습니다. 건너뜁니다.")
            continue

        # 시트 이름에서 언어 구분 및 표시용 이름 분리
        print(f"🔍 시트 이름 확인: '{sheet}'") # Debug print
        if sheet.lower().startswith("k"): # Changed to lower() for case-insensitivity
            tokenizer = tokenizer_ko
            model = model_ko
            lang_name = "한국어"
            display_name = sheet[1:] if len(sheet) > 1 else sheet
            colors_bar = ['skyblue', 'lightgray', 'blue']
            colors_pie = ['skyblue', 'lightgray', 'blue']
            wordcloud_font = font_path_ko if os.path.exists(font_path_ko) else None
            print("✅ 한국어 시트로 판단됨") # Debug print
        elif sheet.lower().startswith("c"): # Changed to lower() for case-insensitivity
            tokenizer = tokenizer_zh
            model = model_zh
            lang_name = "중국어"
            display_name = sheet[1:] if len(sheet) > 1 else sheet
            colors_bar = ['salmon', 'lightgray', 'red']
            colors_pie = ['salmon', 'lightgray', 'red']
            wordcloud_font = font_path_zh if os.path.exists(font_path_zh) else None
            print("✅ 중국어 시트로 판단됨") # Debug print
        else:
            print(f"⚠️ 시트 '{sheet}'의 첫 글자로 언어를 판단할 수 없습니다. 건너뜁니다.")
            continue

        print(f"🌐 언어 판단: {lang_name}")

        # 감성 분석
        results = analyze_sentiments(comments, tokenizer, model)

        # 분석 결과 DataFrame에 추가
        df = df.loc[df[target_column].notna()].copy()
        df["label"] = [r['label'] for r in results]
        df["confidence"] = [r['confidence'] for r in results]

        # 시트별 통계 수집 (이미지 생성하지 않음)
        label_counts = pd.Series([r['label'] for r in results]).value_counts()
        total = label_counts.sum()

        print(f"\n✅ 시트별 감성 분석 요약 (영화: {display_name})")
        for label in ['positive', 'neutral', 'negative']:
            count = label_counts.get(label, 0)
            percent = (count / total) * 100 if total > 0 else 0
            print(f"- {label}: {count}개 ({percent:.1f}%)")

        # 전체 통계를 위한 데이터 수집
        all_comments.extend(comments)
        all_labels.extend([r['label'] for r in results])
        all_confidences.extend([r['confidence'] for r in results])
        sheet_stats[display_name] = {
            'total': total,
            'positive': label_counts.get('positive', 0),
            'neutral': label_counts.get('neutral', 0),
            'negative': label_counts.get('negative', 0),
            'colors': colors_bar
        }

        # 시트 결과 저장용
        sheet_dict[sheet] = df

    # 전체 시트 결과를 새 엑셀 파일로 저장
    if sheet_dict:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for name, sheet_df in sheet_dict.items():
                sheet_df.to_excel(writer, sheet_name=name, index=False)
        print(f"\n✅ 모든 시트 결과가 저장되었습니다: {save_path}")
        
        # 통합 시각화 생성
        if all_comments:
            create_integrated_visualizations(all_comments, all_labels, sheet_stats, current_dir)
    else:
        print("\n⚠️ 처리된 시트가 없어 저장할 내용이 없습니다.")


# ─────────────────────────────────────────────────────────────
# ✅ 메인 실행 부분
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 감성 분석 시작...")
    print(f"📁 입력 파일: {file_path}")
    print(f"💾 출력 파일: {save_path}")
    
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"\n❌ 입력 파일을 찾을 수 없습니다: {file_path}")
        print("📋 다음 사항을 확인해주세요:")
        print("   1. 'movie_review.xlsx' 파일이 현재 스크립트와 같은 폴더에 있는지 확인")
        print("   2. 파일명이 정확한지 확인")
        print("   3. 파일이 손상되지 않았는지 확인")
    else:
        process_excel_sheets(file_path, save_path)
        print("\n�� 감성 분석이 완료되었습니다!")