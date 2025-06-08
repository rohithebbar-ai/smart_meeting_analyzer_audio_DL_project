"""
Audio transcription using Whisper
"""

import whisper
import numpy as np
import torch
import tempfile
import soundfile as sf
from typing import List, Dict, Optional
import os

from config import WHISPER_CONFIG, AUDIO_CONFIG
from src.utils import setup_logging, format_timestamp

class MeetingTranscriber:
    """Handles audio transcription using OpenAI Whisper"""
    
    def __init__(self):
        self.logger = setup_logging("MeetingTranscriber")
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        
        # Initialize Whisper model
        self.model_size = WHISPER_CONFIG['model_size']
        self.language = WHISPER_CONFIG['language']
        
        # Determine device
        if WHISPER_CONFIG['device'] == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = WHISPER_CONFIG['device']
        
        self.logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}")
        
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    
    def transcribe_segments(self, audio: np.ndarray, 
                           speaker_segments: List[Dict]) -> List[Dict]:
        """
        Transcribe speaker-separated audio segments
        
        Args:
            audio: Full audio array
            speaker_segments: List of speaker-labeled segments
            
        Returns:
            List of transcription results with speaker attribution
        """
        self.logger.info(f"Transcribing {len(speaker_segments)} speaker segments...")
        
        transcription_results = []
        
        for i, segment in enumerate(speaker_segments):
            try:
                self.logger.debug(f"Transcribing segment {i+1}/{len(speaker_segments)}: "
                                f"{segment['speaker']} ({segment['start_time']:.2f}s-{segment['end_time']:.2f}s)")
                
                # Extract audio segment
                start_sample = int(segment['start_time'] * self.sample_rate)
                end_sample = int(segment['end_time'] * self.sample_rate)
                segment_audio = audio[start_sample:end_sample]
                
                # Transcribe segment
                transcription = self._transcribe_audio_segment(segment_audio)
                
                # Prepare result
                result = {
                    'speaker': segment['speaker'],
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['duration'],
                    'text': transcription['text'].strip(),
                    'confidence': transcription.get('confidence', 0.0),
                    'language': transcription.get('language', 'unknown'),
                    'timestamp': format_timestamp(segment['start_time'])
                }
                
                transcription_results.append(result)
                
                # Log progress
                if (i + 1) % 5 == 0:
                    self.logger.info(f"Transcribed {i+1}/{len(speaker_segments)} segments")
                
            except Exception as e:
                self.logger.warning(f"Failed to transcribe segment {i}: {str(e)}")
                
                # Add error placeholder
                error_result = {
                    'speaker': segment['speaker'],
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['duration'],
                    'text': '[Transcription failed]',
                    'confidence': 0.0,
                    'language': 'unknown',
                    'timestamp': format_timestamp(segment['start_time']),
                    'error': str(e)
                }
                transcription_results.append(error_result)
        
        self.logger.info(f"Transcription completed: {len(transcription_results)} segments processed")
        
        return transcription_results
    
    def _transcribe_audio_segment(self, audio_segment: np.ndarray) -> Dict:
        """
        Transcribe a single audio segment using Whisper
        
        Args:
            audio_segment: Audio array for transcription
            
        Returns:
            Dictionary with transcription results
        """
        # Create temporary file for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
            try:
                # Save audio segment to temporary file
                sf.write(temp_path, audio_segment, self.sample_rate)
                
                # Transcribe using Whisper
                options = {
                    'language': self.language if self.language != 'auto' else None,
                    'task': 'transcribe',
                    'fp16': self.device == 'cuda',  # Use fp16 on GPU for speed
                }
                
                result = self.model.transcribe(temp_path, **options)
                
                # Extract confidence score (if available)
                confidence = self._calculate_confidence(result)
                
                return {
                    'text': result['text'],
                    'language': result.get('language', 'unknown'),
                    'confidence': confidence
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """
        Calculate confidence score from Whisper result
        
        Args:
            whisper_result: Raw Whisper transcription result
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Extract segment-level probabilities if available
            if 'segments' in whisper_result:
                segment_probs = []
                for segment in whisper_result['segments']:
                    if 'avg_logprob' in segment:
                        # Convert log probability to probability
                        prob = np.exp(segment['avg_logprob'])
                        segment_probs.append(prob)
                
                if segment_probs:
                    return float(np.mean(segment_probs))
            
            # Fallback: simple heuristic based on text characteristics
            text = whisper_result.get('text', '').strip()
            
            if not text:
                return 0.0
            
            # Simple heuristics for confidence
            confidence = 0.5  # Base confidence
            
            # Longer texts tend to be more reliable
            if len(text) > 20:
                confidence += 0.2
            
            # Presence of common words
            common_words = ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has']
            word_count = len(text.split())
            common_count = sum(1 for word in text.lower().split() if word in common_words)
            
            if word_count > 0:
                common_ratio = common_count / word_count
                confidence += min(0.3, common_ratio * 0.5)
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.debug(f"Error calculating confidence: {str(e)}")
            return 0.5  # Default confidence
    
    def transcribe_full_audio(self, audio: np.ndarray) -> Dict:
        """
        Transcribe full audio without speaker separation (for comparison)
        
        Args:
            audio: Full audio array
            
        Returns:
            Dictionary with full transcription result
        """
        self.logger.info("Transcribing full audio (no speaker separation)...")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
            try:
                # Save full audio
                sf.write(temp_path, audio, self.sample_rate)
                
                # Transcribe
                options = {
                    'language': self.language if self.language != 'auto' else None,
                    'task': 'transcribe',
                    'fp16': self.device == 'cuda',
                    'word_timestamps': True  # Get word-level timestamps
                }
                
                result = self.model.transcribe(temp_path, **options)
                
                return {
                    'text': result['text'],
                    'language': result.get('language', 'unknown'),
                    'segments': result.get('segments', []),
                    'duration': len(audio) / self.sample_rate
                }
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def generate_meeting_transcript(self, transcription_results: List[Dict], 
                                   audio_quality: Dict, 
                                   speaker_stats: Dict = None) -> str:
        """
        Generate formatted meeting transcript
        
        Args:
            transcription_results: List of transcribed segments
            audio_quality: Audio quality assessment results
            speaker_stats: Speaker statistics (optional)
            
        Returns:
            Formatted transcript string
        """
        self.logger.info("Generating meeting transcript...")
        
        transcript_lines = []
        
        # Header
        transcript_lines.extend([
            "=" * 60,
            "SMART MEETING ANALYZER - TRANSCRIPT",
            "=" * 60,
            f"Generated on: {self._get_current_timestamp()}",
            f"Audio Quality: {audio_quality.get('quality_label', 'Unknown')} "
            f"(Score: {audio_quality.get('overall_quality_score', 0):.2f}/1.0)",
            ""
        ])
        
        # Audio quality assessment
        transcript_lines.extend([
            "AUDIO QUALITY ASSESSMENT:",
            "-" * 30
        ])
        
        recommendations = audio_quality.get('recommendations', [])
        for rec in recommendations:
            transcript_lines.append(f"• {rec}")
        transcript_lines.append("")
        
        # Speaker statistics (if available)
        if speaker_stats:
            transcript_lines.extend([
                "SPEAKER STATISTICS:",
                "-" * 20
            ])
            
            for speaker, stats in speaker_stats.items():
                speaking_time = format_timestamp(stats['total_speaking_time'])
                transcript_lines.append(
                    f"{speaker}: {speaking_time} speaking time, "
                    f"{stats['number_of_segments']} segments"
                )
            transcript_lines.append("")
        
        # Main transcript
        transcript_lines.extend([
            "MEETING TRANSCRIPT:",
            "-" * 20
        ])
        
        for result in transcription_results:
            timestamp = result['timestamp']
            speaker = result['speaker']
            text = result['text']
            
            # Add confidence indicator for low-confidence transcriptions
            confidence_indicator = ""
            if result.get('confidence', 1.0) < 0.5:
                confidence_indicator = " [?]"
            
            transcript_lines.append(f"[{timestamp}] {speaker}: {text}{confidence_indicator}")
        
        # Footer with statistics
        total_segments = len(transcription_results)
        failed_segments = sum(1 for r in transcription_results if '[Transcription failed]' in r['text'])
        success_rate = ((total_segments - failed_segments) / total_segments * 100) if total_segments > 0 else 0
        
        transcript_lines.extend([
            "",
            "=" * 60,
            f"Transcript Statistics:",
            f"• Total segments: {total_segments}",
            f"• Successfully transcribed: {total_segments - failed_segments}",
            f"• Success rate: {success_rate:.1f}%",
            "=" * 60
        ])
        
        return "\n".join(transcript_lines)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for transcript header"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def export_transcript_json(self, transcription_results: List[Dict], 
                              audio_quality: Dict, 
                              speaker_stats: Dict = None) -> Dict:
        """
        Export transcript in JSON format
        
        Args:
            transcription_results: List of transcribed segments
            audio_quality: Audio quality assessment results
            speaker_stats: Speaker statistics (optional)
            
        Returns:
            Dictionary in JSON-serializable format
        """
        return {
            'metadata': {
                'generated_on': self._get_current_timestamp(),
                'model_used': f"whisper-{self.model_size}",
                'total_segments': len(transcription_results),
                'audio_quality': audio_quality
            },
            'speaker_statistics': speaker_stats or {},
            'transcript': transcription_results
        }