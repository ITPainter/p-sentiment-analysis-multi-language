# -*- coding: utf-8 -*-
"""
Data Processing Module
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os

from core_config import INPUT_CONFIG, OUTPUT_CONFIG, LANGUAGE_OUTPUT_DIRS
from core_language import batch_detect_languages

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes Excel file data"""
    
    def __init__(self):
        self.supported_formats = INPUT_CONFIG['supported_formats']
        self.text_columns = INPUT_CONFIG['text_columns']
        self.encoding = INPUT_CONFIG['encoding']
        
        logger.info("DataProcessor initialized")
    
    def read_excel_file(self, file_path: str) -> Optional[pd.ExcelFile]:
        """
        Reads Excel file.
        
        Args:
            file_path (str): Excel file path
            
        Returns:
            Optional[pd.ExcelFile]: Excel file object or None
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                logger.error(f"Unsupported file format: {file_ext}")
                return None
            
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            logger.info(f"Excel file read successfully: {file_path} (sheets: {len(excel_file.sheet_names)})")
            
            return excel_file
            
        except Exception as e:
            logger.error(f"Excel file reading failed: {str(e)}")
            return None
    
    def find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Finds text column in DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to search
            
        Returns:
            Optional[str]: Text column name or None
        """
        # Search by column name
        for col in df.columns:
            if col.lower() in [col_name.lower() for col_name in self.text_columns]:
                logger.info(f"Text column found: {col}")
                return col
        
        # If column name not found, estimate by content
        for col in df.columns:
            if df[col].dtype == 'object':  # String type
                # Check if it's text by sample data
                sample_data = df[col].dropna().head(10)
                if len(sample_data) > 0:
                    # Consider as text if average length > 10 characters
                    avg_length = sample_data.astype(str).str.len().mean()
                    if avg_length > 10:
                        logger.info(f"Estimated as text column: {col} (avg length: {avg_length:.1f})")
                        return col
        
        logger.warning("Text column not found")
        return None
    
    def process_sheet(self, df: pd.DataFrame, sheet_name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Processes individual sheet.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            sheet_name (str): Sheet name
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: (Processed DataFrame, language code)
        """
        try:
            # Find text column
            text_column = self.find_text_column(df)
            if text_column is None:
                logger.warning(f"Text column not found in sheet '{sheet_name}'")
                return None, None
            
            # Extract text data
            texts = df[text_column].dropna().astype(str).tolist()
            if not texts:
                logger.warning(f"No text data in sheet '{sheet_name}'")
                return None, None
            
            logger.info(f"Processing sheet '{sheet_name}': {len(texts)} texts found")
            
            # Language detection
            detected_languages = batch_detect_languages(texts)
            
            # Select most detected language
            language_counts = {}
            for lang_code, confidence in detected_languages:
                if confidence > 0.5:  # Confidence threshold
                    language_counts[lang_code] = language_counts.get(lang_code, 0) + 1
            
            if not language_counts:
                # If language detection fails, estimate from sheet name
                primary_language = self._infer_language_from_sheet_name(sheet_name)
                logger.info(f"Language detection failed, estimated from sheet name: {primary_language}")
            else:
                primary_language = max(language_counts, key=language_counts.get)
                logger.info(f"Language detection completed: {primary_language} ({language_counts[primary_language]} texts detected)")
            
            # Create result DataFrame
            result_df = df.copy()
            result_df['detected_language'] = [lang for lang, _ in detected_languages]
            result_df['language_confidence'] = [conf for _, conf in detected_languages]
            result_df['primary_language'] = primary_language
            
            return result_df, primary_language
            
        except Exception as e:
            logger.error(f"Sheet '{sheet_name}' processing failed: {str(e)}")
            return None, None
    
    def _infer_language_from_sheet_name(self, sheet_name: str) -> str:
        """
        Infers language from sheet name.
        
        Args:
            sheet_name (str): Sheet name
            
        Returns:
            str: Language code
        """
        sheet_lower = sheet_name.lower()
        
        if sheet_lower.startswith('k') or 'korean' in sheet_lower or '한국' in sheet_name:
            return 'k'
        elif sheet_lower.startswith('c') or 'chinese' in sheet_lower or '중국' in sheet_name:
            return 'c'
        elif sheet_lower.startswith('j') or 'japanese' in sheet_lower or '일본' in sheet_name:
            return 'j'
        elif sheet_lower.startswith('e') or 'english' in sheet_lower or '영어' in sheet_name:
            return 'e'
        else:
            return 'e'  # Default: English
    
    def process_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        Processes Excel file completely.
        
        Args:
            file_path (str): Excel file path
            
        Returns:
            Dict[str, Any]: Processing results
        """
        # Read Excel file
        excel_file = self.read_excel_file(file_path)
        if excel_file is None:
            return {'success': False, 'error': 'Cannot read Excel file.'}
        
        results = {
            'success': True,
            'file_path': file_path,
            'sheets': {},
            'summary': {
                'total_sheets': len(excel_file.sheet_names),
                'processed_sheets': 0,
                'failed_sheets': 0,
                'language_distribution': {}
            }
        }
        
        # Process each sheet
        for sheet_name in excel_file.sheet_names:
            try:
                # Read sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Process sheet
                processed_df, language = self.process_sheet(df, sheet_name)
                
                if processed_df is not None and language is not None:
                    results['sheets'][sheet_name] = {
                        'data': processed_df,
                        'language': language,
                        'text_count': len(processed_df.dropna(subset=[self.find_text_column(processed_df)]))
                    }
                    results['summary']['processed_sheets'] += 1
                    
                    # Language statistics
                    if language not in results['summary']['language_distribution']:
                        results['summary']['language_distribution'][language] = 0
                    results['summary']['language_distribution'][language] += 1
                    
                    logger.info(f"Sheet '{sheet_name}' processing completed: {language}")
                else:
                    results['summary']['failed_sheets'] += 1
                    logger.warning(f"Sheet '{sheet_name}' processing failed")
                    
            except Exception as e:
                results['summary']['failed_sheets'] += 1
                logger.error(f"Error processing sheet '{sheet_name}': {str(e)}")
        
        logger.info(f"Excel file processing completed: {results['summary']['processed_sheets']}/{results['summary']['total_sheets']} sheets successful")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> Dict[str, str]:
        """
        Saves processing results to files.
        
        Args:
            results (Dict[str, Any]): Processing results
            output_dir (str): Output directory (default: config output directory)
            
        Returns:
            Dict[str, str]: Saved file paths
        """
        if not results['success']:
            logger.error("No results to save")
            return {}
        
        saved_files = {}
        
        try:
            for sheet_name, sheet_data in results['sheets'].items():
                language = sheet_data['language']
                df = sheet_data['data']
                
                # Determine language-specific output directory
                if output_dir:
                    lang_output_dir = Path(output_dir) / language
                else:
                    lang_output_dir = LANGUAGE_OUTPUT_DIRS.get(language, Path('output') / language)
                
                # Create directory
                lang_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filename
                input_filename = Path(results['file_path']).stem
                output_filename = f"{input_filename}_{sheet_name}_result.xlsx"
                output_path = lang_output_dir / output_filename
                
                # Save as Excel file
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Sentiment_Analysis_Results', index=False)
                    
                    # Add metadata sheet
                    metadata = pd.DataFrame({
                        'Item': ['Processing_Time', 'Language', 'Text_Count', 'Source_File'],
                        'Value': [
                            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                            language,
                            sheet_data['text_count'],
                            results['file_path']
                        ]
                    })
                    metadata.to_excel(writer, sheet_name='Metadata', index=False)
                
                saved_files[sheet_name] = str(output_path)
                logger.info(f"Results saved: {output_path}")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Results saving failed: {str(e)}")
            return {}
    
    def get_processing_summary(self, results: Dict[str, Any]) -> str:
        """
        Returns processing results summary as string.
        
        Args:
            results (Dict[str, Any]): Processing results
            
        Returns:
            str: Summary string
        """
        if not results['success']:
            return "Processing failed"
        
        summary = results['summary']
        lang_dist = summary['language_distribution']
        
        summary_text = f"""
📊 Excel File Processing Complete

📁 File: {results['file_path']}
📋 Total Sheets: {summary['total_sheets']}
✅ Success: {summary['processed_sheets']}
❌ Failed: {summary['failed_sheets']}

🌍 Language Distribution:
"""
        
        for lang, count in lang_dist.items():
            lang_name = {'k': 'Korean', 'c': 'Chinese', 'j': 'Japanese', 'e': 'English'}.get(lang, lang)
            summary_text += f"  - {lang_name}: {count} sheets\n"
        
        return summary_text.strip()
