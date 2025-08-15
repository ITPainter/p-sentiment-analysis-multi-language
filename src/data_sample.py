# -*- coding: utf-8 -*-
"""
Sample Data Creation Script
"""

import pandas as pd
from pathlib import Path

def create_sample_excel():
    """Creates test sample Excel file."""
    
    # Sample data
    sample_data = {
        'k': {  # Korean
            'comment': [
                "이 영화 정말 재미있었어요!",
                "스토리가 너무 지루했어요.",
                "배우들의 연기가 훌륭했습니다.",
                "특수효과가 인상적이었어요.",
                "시간이 아깝지 않은 영화였습니다.",
                "내용이 너무 복잡해서 이해하기 어려웠어요.",
                "감동적인 영화였습니다.",
                "기대했던 것보다 별로였어요.",
                "추천하고 싶은 영화입니다.",
                "다시 보고 싶지 않아요."
            ]
        },
        'c': {  # Chinese
            'comment': [
                "这部电影很有趣！",
                "故事情节太无聊了。",
                "演员的表演很出色。",
                "特效令人印象深刻。",
                "值得花时间看的电影。",
                "内容太复杂，难以理解。",
                "这是一部感人的电影。",
                "没有预期的那么好。",
                "推荐大家观看。",
                "不想再看第二遍。"
            ]
        },
        'e': {  # English
            'comment': [
                "This movie was really entertaining!",
                "The story was too boring.",
                "The acting was excellent.",
                "The special effects were impressive.",
                "It was worth the time.",
                "The plot was too complex to understand.",
                "It was a touching film.",
                "It wasn't as good as expected.",
                "I highly recommend it.",
                "I don't want to watch it again."
            ]
        },
        'j': {  # Japanese
            'comment': [
                "この映画は本当に面白かったです！",
                "ストーリーが退屈でした。",
                "俳優の演技が素晴らしかったです。",
                "特殊効果が印象的でした。",
                "時間をかける価値のある映画でした。",
                "内容が複雑すぎて理解しにくかったです。",
                "感動的な映画でした。",
                "期待していたほど良くなかったです。",
                "おすすめの映画です。",
                "もう一度見たくありません。"
            ]
        }
    }
    
    # Create Excel file in output directory
    output_file = Path(__file__).parent.parent / "output" / "sample_data.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for lang_code, data in sample_data.items():
            # Create DataFrame
            df = pd.DataFrame({
                'comment': data['comment'],
                'timestamp': pd.Timestamp.now(),
                'language': lang_code
            })
            
            # Set sheet name
            sheet_name = f"{lang_code}_sample"
            
            # Save to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"✅ {lang_code} sample data created: {len(data['comment'])} comments")
    
    print(f"\n🎉 Sample Excel file created: {output_file}")
    print("📋 Usage:")
    print("   python src/main.py output/sample_data.xlsx")
    
    return output_file

if __name__ == "__main__":
    create_sample_excel()
