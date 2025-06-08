"""
Main meeting analyzer orchestrating all components
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.audio_processor import AudioProcessor
from src.speaker_detector import SpeakerDetector
from src.transcriber import MeetingTranscriber
from src.nlp_highlighter import IntelligentHighlightExtractor
from src.utils import (
    setup_logging, load_audio, validate_audio_file, 
    estimate_processing_time, create_output_filename,
    calculate_speaking_stats, create_visualization,
    json_serializer
)
from config import OUTPUT_CONFIG, OUTPUT_DIR

class SmartMeetingAnalyzer:
    """
    Main class orchestrating the complete meeting analysis pipeline
    """
    
    def __init__(self):
        self.logger = setup_logging("SmartMeetingAnalyzer")
        
        # Initialize components
        self.logger.info("Initializing Smart Meeting Analyzer components...")
        
        try:
            self.audio_processor = AudioProcessor()
            self.speaker_detector = SpeakerDetector()
            self.transcriber = MeetingTranscriber()
            self.highlight_extractor = IntelligentHighlightExtractor()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def analyze_meeting(self, audio_file_path: str, 
                       output_prefix: str = None) -> Dict:
        """
        Complete meeting analysis pipeline
        
        Args:
            audio_file_path: Path to audio file
            output_prefix: Optional prefix for output files
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info(f"Starting meeting analysis for: {audio_file_path}")
        
        # Validate input
        if not validate_audio_file(audio_file_path):
            raise ValueError(f"Invalid audio file: {audio_file_path}")
        
        try:
            # Step 1: Load and preprocess audio
            self.logger.info("Step 1/6: Loading and preprocessing audio...")
            audio, sr = load_audio(audio_file_path)
            duration = len(audio) / sr
            
            self.logger.info(f"Audio loaded: {duration:.2f} seconds, {sr}Hz")
            self.logger.info(f"Estimated processing time: {estimate_processing_time(duration)}")
            
            processed_audio = self.audio_processor.preprocess_audio(audio, sr)
            
            # Step 2: Assess audio quality
            self.logger.info("Step 2/6: Assessing audio quality...")
            quality_assessment = self.audio_processor.assess_audio_quality(processed_audio)
            
            # Step 3: Detect voice activity
            self.logger.info("Step 3/6: Detecting voice activity...")
            voice_segments, energy = self.audio_processor.detect_voice_activity(processed_audio)
            
            if not voice_segments:
                self.logger.warning("No voice activity detected in audio")
                return self._create_empty_result(audio_file_path, quality_assessment)
            
            # Step 4: Identify speakers
            self.logger.info("Step 4/6: Identifying speakers...")
            speaker_segments = self.speaker_detector.identify_speakers(processed_audio, voice_segments)
            
            if not speaker_segments:
                self.logger.warning("No speakers identified")
                return self._create_empty_result(audio_file_path, quality_assessment)
            
            # Refine speaker segments
            refined_segments = self.speaker_detector.refine_speaker_segments(
                processed_audio, speaker_segments
            )
            
            # Step 5: Analyze speaker characteristics
            self.logger.info("Step 5/6: Analyzing speaker characteristics...")
            speaker_stats = self.speaker_detector.analyze_speaker_characteristics(
                processed_audio, refined_segments
            )
            
            # Step 6: Transcribe speech
            self.logger.info("Step 6/6: Transcribing speech...")
            transcription_results = self.transcriber.transcribe_segments(
                processed_audio, refined_segments
            )
            
            # Generate outputs
            results = self._generate_outputs(
                audio_file_path=audio_file_path,
                audio=processed_audio,
                energy=energy,
                voice_segments=voice_segments,
                speaker_segments=refined_segments,
                transcription_results=transcription_results,
                quality_assessment=quality_assessment,
                speaker_stats=speaker_stats,
                output_prefix=output_prefix
            )
            
            self.logger.info("Meeting analysis completed successfully!")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Meeting analysis failed: {str(e)}")
            raise
    
    def _create_empty_result(self, audio_file_path: str, quality_assessment: Dict) -> Dict:
        """Create result structure when no speech/speakers are detected"""
        return {
            'input_file': audio_file_path,
            'status': 'no_speech_detected',
            'quality_assessment': quality_assessment,
            'speaker_segments': [],
            'transcription_results': [],
            'speaker_stats': {},
            'output_files': {}
        }
    
    def _generate_outputs(self, audio_file_path: str, audio, energy, 
                         voice_segments, speaker_segments, transcription_results,
                         quality_assessment, speaker_stats, output_prefix=None) -> Dict:
        """Generate all output files and return results"""
        
        self.logger.info("Generating output files...")
        
        output_files = {}
        base_name = output_prefix or Path(audio_file_path).stem
        
        # 1. Generate meeting transcript
        if OUTPUT_CONFIG['generate_transcript']:
            transcript_text = self.transcriber.generate_meeting_transcript(
                transcription_results, quality_assessment, speaker_stats
            )
            
            transcript_file = create_output_filename(
                audio_file_path, "transcript", "txt"
            )
            
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            
            output_files['transcript'] = transcript_file
            self.logger.info(f"Transcript saved: {transcript_file}")
        
        # 2. Generate JSON export
        json_data = self.transcriber.export_transcript_json(
            transcription_results, quality_assessment, speaker_stats
        )
        
        json_file = create_output_filename(audio_file_path, "data", "json")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        output_files['json'] = json_file
        self.logger.info(f"JSON data saved: {json_file}")
        
        # 3. Generate visualization
        viz_file = create_output_filename(audio_file_path, "visualization", "png")
        
        try:
            self.audio_processor.visualize_audio_analysis(
                audio, energy, voice_segments, viz_file
            )
            output_files['audio_visualization'] = viz_file
            
            # Meeting overview visualization
            meeting_viz_file = create_output_filename(audio_file_path, "meeting_overview", "png")
            create_visualization(speaker_segments, quality_assessment, meeting_viz_file)
            output_files['meeting_visualization'] = meeting_viz_file
            
        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {str(e)}")
        
        # 4. Save speaker segments (if enabled)
        if OUTPUT_CONFIG['save_audio_segments']:
            segments_dir = OUTPUT_DIR / f"{base_name}_segments"
            saved_segments = self.speaker_detector.save_speaker_segments(
                audio, speaker_segments, str(segments_dir)
            )
            output_files['speaker_segments'] = saved_segments
        
        # 5. Generate summary
        if OUTPUT_CONFIG['generate_summary']:
            summary = self._generate_meeting_summary(
                transcription_results, speaker_stats, quality_assessment
            )
            
            summary_file = create_output_filename(audio_file_path, "summary", "txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            output_files['summary'] = summary_file
            self.logger.info(f"Summary saved: {summary_file}")
        
        # Compile final results
        results = {
            'input_file': audio_file_path,
            'status': 'success',
            'processing_info': {
                'total_segments': len(speaker_segments),
                'unique_speakers': len(set(seg['speaker'] for seg in speaker_segments)),
                'total_duration': len(audio) / 16000,  # Assuming 16kHz
                'transcribed_segments': len([r for r in transcription_results if r['text'] != '[Transcription failed]'])
            },
            'quality_assessment': quality_assessment,
            'speaker_segments': speaker_segments,
            'transcription_results': transcription_results,
            'speaker_stats': speaker_stats,
            'output_files': output_files
        }
        
        return results
    
    def _generate_meeting_summary(self, transcription_results: List[Dict], 
                                 speaker_stats: Dict, quality_assessment: Dict) -> str:
        """Generate a concise meeting summary"""
        
        summary_lines = []
        
        # Header
        summary_lines.extend([
            "MEETING SUMMARY",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # Key statistics
        total_duration = sum(result['duration'] for result in transcription_results)
        unique_speakers = len(speaker_stats)
        
        summary_lines.extend([
            "KEY STATISTICS:",
            f"• Meeting Duration: {self._format_duration(total_duration)}",
            f"• Number of Speakers: {unique_speakers}",
            f"• Audio Quality: {quality_assessment.get('quality_label', 'Unknown')}",
            f"• Total Segments: {len(transcription_results)}",
            ""
        ])
        
        # Speaker participation
        if speaker_stats:
            summary_lines.extend([
                "SPEAKER PARTICIPATION:",
                "-" * 25
            ])
            
            # Sort speakers by speaking time
            sorted_speakers = sorted(
                speaker_stats.items(),
                key=lambda x: x[1]['total_speaking_time'],
                reverse=True
            )
            
            for speaker, stats in sorted_speakers:
                speaking_time = self._format_duration(stats['total_speaking_time'])
                percentage = (stats['total_speaking_time'] / total_duration * 100) if total_duration > 0 else 0
                
                summary_lines.append(
                    f"• {speaker}: {speaking_time} ({percentage:.1f}% of meeting)"
                )
            
            summary_lines.append("")
        
        # Audio quality insights
        recommendations = quality_assessment.get('recommendations', [])
        if recommendations and not any('good' in rec.lower() for rec in recommendations):
            summary_lines.extend([
                "AUDIO QUALITY RECOMMENDATIONS:",
                "-" * 35
            ])
            
            for rec in recommendations[:3]:  # Top 3 recommendations
                summary_lines.append(f"• {rec}")
            
            summary_lines.append("")
        
        # Meeting highlights (using advanced NLP)
        highlights = self._extract_meeting_highlights(transcription_results)
        if highlights:
            summary_lines.extend([
                "KEY HIGHLIGHTS & INSIGHTS:",
                "-" * 30
            ])
            
            for highlight in highlights[:8]:  # Top 8 highlights
                summary_lines.append(f"• {highlight}")
            
            summary_lines.append("")
        
        summary_lines.extend([
            "=" * 40,
            "End of Summary"
        ])
        
        return "\n".join(summary_lines)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _extract_meeting_highlights(self, transcription_results: List[Dict]) -> List[str]:
        """Extract intelligent meeting highlights using advanced NLP"""
        try:
            return self.highlight_extractor.extract_highlights(transcription_results)
        except Exception as e:
            self.logger.error(f"Advanced highlight extraction failed: {str(e)}")
            # Fallback to simple extraction
            return self._simple_fallback_highlights(transcription_results)
    
    def _simple_fallback_highlights(self, transcription_results: List[Dict]) -> List[str]:
        """Simple fallback when advanced NLP fails"""
        try:
            all_text = " ".join([result['text'] for result in transcription_results])
            
            # Basic sentence extraction for important statements
            import re
            sentences = re.split(r'[.!?]+', all_text)
            
            highlights = []
            for sentence in sentences:
                sentence = sentence.strip()
                # Look for sentences with important indicators
                if (len(sentence.split()) > 5 and 
                    any(word in sentence.lower() for word in [
                        'important', 'key', 'main', 'focus', 'result', 
                        'conclusion', 'finding', 'issue', 'problem', 'solution'
                    ])):
                    highlights.append(f"Key point: {sentence}")
                    if len(highlights) >= 5:
                        break
            
            return highlights
            
        except Exception as e:
            self.logger.debug(f"Error in fallback highlight extraction: {str(e)}")
            return []
    
    def quick_analysis(self, audio_file_path: str) -> Dict:
        """
        Quick analysis for demo purposes - faster but less detailed
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Basic analysis results
        """
        self.logger.info(f"Starting quick analysis for: {audio_file_path}")
        
        try:
            # Load audio (limit to first 2 minutes for demo)
            audio, sr = load_audio(audio_file_path)
            max_samples = sr * 120  # 2 minutes
            
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                self.logger.info("Limited analysis to first 2 minutes for demo")
            
            # Basic processing
            processed_audio = self.audio_processor.preprocess_audio(audio, sr)
            quality_assessment = self.audio_processor.assess_audio_quality(processed_audio)
            voice_segments, energy = self.audio_processor.detect_voice_activity(processed_audio)
            
            # Simplified speaker detection (faster clustering)
            if voice_segments:
                speaker_segments = self.speaker_detector.identify_speakers(processed_audio, voice_segments)
                
                # Quick transcription (fewer segments)
                if speaker_segments:
                    # Limit to first 5 segments for demo
                    limited_segments = speaker_segments[:5]
                    transcription_results = self.transcriber.transcribe_segments(
                        processed_audio, limited_segments
                    )
                else:
                    transcription_results = []
            else:
                speaker_segments = []
                transcription_results = []
            
            return {
                'input_file': audio_file_path,
                'status': 'quick_analysis_complete',
                'note': 'This is a quick demo analysis. Use analyze_meeting() for full processing.',
                'quality_assessment': quality_assessment,
                'speaker_segments': speaker_segments,
                'transcription_results': transcription_results,
                'processing_info': {
                    'demo_mode': True,
                    'analyzed_duration': len(audio) / sr,
                    'segments_processed': len(transcription_results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Quick analysis failed: {str(e)}")
            raise
    
    def analyze_audio_quality_only(self, audio_file_path: str) -> Dict:
        """
        Analyze only audio quality without transcription (very fast)
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Audio quality assessment results
        """
        self.logger.info(f"Analyzing audio quality for: {audio_file_path}")
        
        try:
            # Load and process audio
            audio, sr = load_audio(audio_file_path)
            processed_audio = self.audio_processor.preprocess_audio(audio, sr)
            
            # Quality assessment only
            quality_assessment = self.audio_processor.assess_audio_quality(processed_audio)
            voice_segments, energy = self.audio_processor.detect_voice_activity(processed_audio)
            
            # Generate basic visualization
            viz_file = create_output_filename(audio_file_path, "quality_analysis", "png")
            try:
                self.audio_processor.visualize_audio_analysis(
                    processed_audio, energy, voice_segments, viz_file
                )
                quality_assessment['visualization'] = viz_file
            except Exception as e:
                self.logger.warning(f"Failed to generate visualization: {str(e)}")
            
            return {
                'input_file': audio_file_path,
                'status': 'quality_analysis_complete',
                'quality_assessment': quality_assessment,
                'voice_segments_detected': len(voice_segments),
                'total_speech_duration': sum(end - start for start, end in voice_segments),
                'processing_info': {
                    'analysis_type': 'quality_only',
                    'total_duration': len(audio) / sr
                }
            }
            
        except Exception as e:
            self.logger.error(f"Quality analysis failed: {str(e)}")
            raise
    
    def batch_analyze(self, audio_files: List[str], output_dir: str = None) -> Dict:
        """
        Analyze multiple audio files in batch
        
        Args:
            audio_files: List of audio file paths
            output_dir: Optional output directory for results
            
        Returns:
            Dictionary with results for each file
        """
        self.logger.info(f"Starting batch analysis of {len(audio_files)} files...")
        
        results = {}
        failed_files = []
        
        for i, audio_file in enumerate(audio_files):
            self.logger.info(f"Processing file {i+1}/{len(audio_files)}: {Path(audio_file).name}")
            
            try:
                # Use quick analysis for batch processing
                file_result = self.quick_analysis(audio_file)
                results[audio_file] = file_result
                
            except Exception as e:
                self.logger.error(f"Failed to process {audio_file}: {str(e)}")
                failed_files.append({
                    'file': audio_file,
                    'error': str(e)
                })
                results[audio_file] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Generate batch summary
        batch_summary = {
            'total_files': len(audio_files),
            'successful': len(audio_files) - len(failed_files),
            'failed': len(failed_files),
            'failed_files': failed_files,
            'results': results
        }
        
        # Save batch summary
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            summary_file = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, indent=2, ensure_ascii=False, default=json_serializer)
            
            self.logger.info(f"Batch summary saved: {summary_file}")
        
        return batch_summary
    
    def analyze_with_custom_config(self, audio_file_path: str, custom_config: Dict) -> Dict:
        """
        Analyze meeting with custom configuration parameters
        
        Args:
            audio_file_path: Path to audio file
            custom_config: Custom configuration overrides
            
        Returns:
            Analysis results with custom settings
        """
        self.logger.info(f"Running custom analysis for: {audio_file_path}")
        
        # Store original config
        original_config = {}
        
        try:
            # Apply custom configuration temporarily
            from config import WHISPER_CONFIG, SPEAKER_CONFIG, OUTPUT_CONFIG
            
            if 'whisper' in custom_config:
                original_config['whisper'] = WHISPER_CONFIG.copy()
                WHISPER_CONFIG.update(custom_config['whisper'])
            
            if 'speaker' in custom_config:
                original_config['speaker'] = SPEAKER_CONFIG.copy()
                SPEAKER_CONFIG.update(custom_config['speaker'])
            
            if 'output' in custom_config:
                original_config['output'] = OUTPUT_CONFIG.copy()
                OUTPUT_CONFIG.update(custom_config['output'])
            
            # Run analysis with custom config
            results = self.analyze_meeting(audio_file_path)
            
            # Add config info to results
            results['custom_config_used'] = custom_config
            
            return results
            
        finally:
            # Restore original configuration
            if 'whisper' in original_config:
                WHISPER_CONFIG.update(original_config['whisper'])
            if 'speaker' in original_config:
                SPEAKER_CONFIG.update(original_config['speaker'])
            if 'output' in original_config:
                OUTPUT_CONFIG.update(original_config['output'])
    
    def compare_models(self, audio_file_path: str) -> Dict:
        """
        Compare performance with different model configurations
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Comparison results across different configurations
        """
        self.logger.info(f"Running model comparison for: {audio_file_path}")
        
        configs = [
            {
                'name': 'Fast (Tiny Whisper)',
                'whisper': {'model_size': 'tiny'},
                'speaker': {'similarity_threshold': 0.8}
            },
            {
                'name': 'Balanced (Base Whisper)',
                'whisper': {'model_size': 'base'},
                'speaker': {'similarity_threshold': 0.75}
            },
            {
                'name': 'Accurate (Small Whisper)',
                'whisper': {'model_size': 'small'},
                'speaker': {'similarity_threshold': 0.7}
            }
        ]
        
        comparison_results = {}
        
        for config in configs:
            try:
                start_time = datetime.now()
                
                result = self.analyze_with_custom_config(audio_file_path, config)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                comparison_results[config['name']] = {
                    'processing_time_seconds': processing_time,
                    'status': result['status'],
                    'unique_speakers': result['processing_info']['unique_speakers'],
                    'transcribed_segments': result['processing_info']['transcribed_segments'],
                    'quality_score': result['quality_assessment']['overall_quality_score']
                }
                
            except Exception as e:
                comparison_results[config['name']] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return {
            'input_file': audio_file_path,
            'comparison_results': comparison_results,
            'recommendation': self._get_model_recommendation(comparison_results)
        }
    
    def _get_model_recommendation(self, comparison_results: Dict) -> str:
        """Generate model recommendation based on comparison results"""
        
        successful_configs = {name: results for name, results in comparison_results.items() 
                            if results.get('status') == 'success'}
        
        if not successful_configs:
            return "All configurations failed. Check audio quality and try again."
        
        # Find fastest configuration
        fastest = min(successful_configs.items(), 
                     key=lambda x: x[1].get('processing_time_seconds', float('inf')))
        
        # Find most accurate (highest quality score)
        most_accurate = max(successful_configs.items(),
                           key=lambda x: x[1].get('quality_score', 0))
        
        recommendation = f"For speed: {fastest[0]} ({fastest[1]['processing_time_seconds']:.1f}s). "
        recommendation += f"For accuracy: {most_accurate[0]} (quality: {most_accurate[1]['quality_score']:.2f})."
        
        return recommendation