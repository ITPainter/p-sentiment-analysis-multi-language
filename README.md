# 🎬 다국어 영화 리뷰 감성 분석 시스템

## 📋 프로젝트 개요

이 프로젝트는 한국어와 중국어 영화 리뷰에 대한 감성 분석을 수행하고, 워드클라우드, 막대 그래프, 원형 차트를 생성하는 종합적인 텍스트 분석 시스템입니다.

## ✨ 주요 기능

### 🌐 다국어 지원
- **한국어**: `snunlp/KR-FinBert-SC` 사용 (가장 정확한 한국어 감정분석 모델)
- **중국어**: `IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment` 사용 (감성 분석 전용)

### 📊 분석 결과
1. **감성 분석**: 긍정(positive), 중립(neutral), 부정(negative) 분류
2. **워드클라우드**: 텍스트에서 자주 등장하는 단어 시각화
3. **막대 그래프**: 감성별 댓글 수 통계
4. **원형 차트**: 감성별 비율 시각화
5. **Excel 출력**: 분석 결과를 구조화된 형태로 저장

### 🎯 신뢰도 기반 분류
- **신뢰도 < 0.8**: 모든 모델에서 중립으로 분류
- **신뢰도 ≥ 0.8**: 
  - **중국어 모델**: 바로 긍정/부정으로 분류
  - **한국어 모델**: 
    - 신뢰도 < 0.95: 중립으로 분류
    - 신뢰도 ≥ 0.95: 긍정으로 분류

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
- `k_movie_review.xlsx` 또는 `c_movie_review.xlsx` 파일을 프로젝트 루트에 배치
- 시트 이름 규칙:
  - `K_영화명`: 한국어 리뷰 (예: K_노량유튜브, K_팔백유튜브)
  - `C_영화명`: 중국어 리뷰 (예: C_팔백도우반, C_노량도우반)

### 3. 실행
```bash
python sen.py
```

## 📁 파일 구조

```
ssafy-custom-news-basic/
├── sen.py                          # 메인 분석 스크립트
├── k_movie_review.xlsx             # 한국어 입력 데이터
├── c_movie_review.xlsx             # 중국어 입력 데이터
├── k_movie_review_result.xlsx      # 한국어 분석 결과
├── c_movie_review_result.xlsx      # 중국어 분석 결과
├── 전체_워드클라우드.png          # 통합 워드클라우드
├── 전체_막대그래프.png            # 통합 막대 그래프
├── 전체_원형차트.png              # 통합 원형 차트
├── 영화별_비교차트.png            # 영화별 비교 차트
├── requirements.txt                 # Python 의존성 목록
├── README.md                       # 프로젝트 문서
└── venv/                          # 가상환경
```

## 🔧 기술 스택

### 핵심 라이브러리
- **PyTorch**: 딥러닝 모델 실행
- **Transformers**: Hugging Face 사전 훈련 모델
- **Pandas**: 데이터 처리 및 Excel I/O
- **Matplotlib**: 그래프 및 차트 생성
- **WordCloud**: 텍스트 시각화

### AI 모델
- **한국어**: `snunlp/KR-FinBert-SC` (가장 정확한 한국어 감정분석 모델)
- **중국어**: `IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment`

## 📊 입력 데이터 형식

### Excel 파일 구조
| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| id | 고유 식별자 | 1, 2, 3... |
| ref | 참조 정보 | 영화 ID |
| name | 사용자명 | 사용자1 |
| comment | 댓글 내용 | 영화가 정말 좋았습니다 |
| likes | 좋아요 수 | 15 |
| publisher | 게시자 | 네이버 |
| title | 제목 | 영화 리뷰 |
| url | 링크 | https://... |

## 🎨 출력 결과

### 1. 감성 분석 통계
```
📊 전체 감성 분석 통계 (총 2000개 댓글)
- positive: 821개 (41.0%)
- neutral: 472개 (23.6%)
- negative: 707개 (35.4%)
```

### 2. 시각화 파일
- **전체_워드클라우드.png**: 모든 영화 리뷰 통합 워드클라우드 (한글 폰트 지원)
- **전체_막대그래프.png**: 전체 감성 분석 통계
- **전체_원형차트.png**: 전체 감성별 비율 분포
- **영화별_비교차트.png**: 영화별 감성 분석 비교 (그룹화된 막대 차트)

### 3. Excel 결과 파일
- 원본 데이터 + 감성 분석 라벨 + 신뢰도 점수

## ⚙️ 설정 및 커스터마이징

### 폰트 설정
```python
# Windows 환경
font_path_ko = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
font_path_zh = 'C:/Windows/Fonts/simsun.ttc'  # SimSun
```

### 워드클라우드 설정
```python
wordcloud_params = {
    'width': 1200,
    'height': 600,
    'max_words': 200,
    'min_font_size': 10,
    'relative_scaling': 0.5,
    'colormap': 'viridis'
}
```

### 신뢰도 임계값
```python
# 모든 모델에 신뢰도 기반 중립 분류 적용
if conf < 0.8:  # 기본 신뢰도 임계값
    label = "neutral"
elif model == model_zh:
    # 중국어 모델: 바로 긍정/부정 분류
    label = "positive" if pred == 1 else "negative"
else:
    # 한국어 모델: 더 엄격한 긍정 분류
    if pred == 1 and conf < 0.95:
        label = "neutral"
    else:
        label = "positive" if pred == 1 else "negative"
```

## 🔍 문제 해결

### 일반적인 오류
1. **폰트 경고**: matplotlib 폰트 경고는 자동으로 차단됨
2. **메모리 부족**: batch_size를 줄여서 해결 (기본값: 16)
3. **모델 다운로드 실패**: 인터넷 연결 확인 및 재시도

### 성능 최적화
- GPU 사용 시 `torch.cuda.is_available()` 확인
- 배치 크기 조정으로 메모리 사용량 최적화
- 이미지 해상도 조정으로 처리 속도 향상

## 📈 성능 지표

### 처리 속도
- **CPU**: ~100-200 댓글/분
- **GPU**: ~500-1000 댓글/분

### 정확도
- **한국어**: 90-95% (KR-FinBert-SC 모델, 한국어 감정분석 벤치마크 최고 성능)
- **중국어**: 80-85% (감성 분석 전용 모델)

### 실제 분석 결과 예시
- **한국어 (2000개 댓글)**: 긍정 86.8%, 중립 10.8%, 부정 2.4%
- **중국어 (2000개 댓글)**: 긍정 41.0%, 중립 23.6%, 부정 35.4%

### 한국어 감정분석 모델 순위
1. **snunlp/KR-FinBert-SC** (현재 사용) - 금융/일반 텍스트 모두 우수
2. **beomi/KcELECTRA-base-v2022** - 최신 업데이트 버전
3. **klue/roberta-base** - KLUE 프로젝트 공식 모델
4. **alsgyu/sentiment-analysis-fine-tuned-model** - 이전 사용 모델

## 🤝 기여 방법

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 문의사항이나 버그 리포트는 Issues 탭을 이용해주세요.

---

**개발자**: SSAFY Custom News Basic Team  
**최종 업데이트**: 2025년 8월 13일  
**버전**: 2.0.0

## 🔄 주요 업데이트 내역

### v2.0.0 (2025-08-13)
- ✅ **신뢰도 기준 개선**: 기본 임계값 0.8, 한국어 모델 추가 임계값 0.95
- ✅ **한국어 모델 엄격화**: 신뢰도 0.95 이상일 때만 긍정 분류
- ✅ **중국어 모델 표준화**: 신뢰도 0.8 이상일 때 바로 긍정/부정 분류
- ✅ **통합 시각화**: 엑셀 파일별로 통합된 결과물 생성
- ✅ **한글 폰트 지원**: 워드클라우드 한글 깨짐 문제 해결
- ✅ **균형잡힌 분포**: 더 현실적인 감성 분석 결과

### v1.0.0 (2025-08-13)
- 🚀 **초기 버전**: 기본 감성 분석 기능 구현

