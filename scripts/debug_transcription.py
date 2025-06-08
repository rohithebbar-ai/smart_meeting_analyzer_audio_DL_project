#!/usr/bin/env python3
"""
Debug transcription quality and suggest improvements
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.meeting_analyzer import SmartMeetingAnalyzer
from src.audio_processor import AudioProcessor
from src.utils import load_audio

def analyze_transcription_quality(audio_file):
    """
    Analyze why transcription quality might be poor
    """
    print("🔍 Analyzing Transcription Quality Issues...")
    print("=" * 50)
    
    # 1. Check audio quality first
    print("\n1️⃣ AUDIO QUALITY ANALYSIS:")
    try:
        audio, sr = load_audio(audio_file)
        processor = AudioProcessor()
        
        quality = processor.assess_audio_quality(audio)
        
        print(f"   Overall Quality: {quality['quality_label']} ({quality['overall_quality_score']:.2f}/1.0)")
        print(f"   Energy Level: {quality['energy_level']:.3f}")
        print(f"   Clarity: {quality['clarity']:.0f} Hz")
        print(f"   Noise Level: {quality['noise_level']:.4f}")
        
        # Quality recommendations
        print(f"\n   🔧 Quality Issues:")
        for rec in quality['recommendations']:
            print(f"      • {rec}")
        
        # Voice activity
        voice_segments, energy = processor.detect_voice_activity(audio)
        total_speech = sum(end - start for start, end in voice_segments)
        total_duration = len(audio) / sr
        speech_ratio = total_speech / total_duration
        
        print(f"\n   📊 Speech Analysis:")
        print(f"      • Total Duration: {total_duration:.1f}s")
        print(f"      • Speech Duration: {total_speech:.1f}s")
        print(f"      • Speech Ratio: {speech_ratio:.1%}")
        print(f"      • Voice Segments: {len(voice_segments)}")
        
        if speech_ratio < 0.3:
            print(f"      ⚠️  Low speech ratio - lots of silence/noise")
        if quality['overall_quality_score'] < 0.6:
            print(f"      ⚠️  Poor audio quality will hurt transcription")
            
    except Exception as e:
        print(f"   ❌ Audio analysis failed: {e}")
    
    # 2. Test different Whisper models
    print("\n2️⃣ WHISPER MODEL COMPARISON:")
    
    models_to_test = ['tiny', 'base', 'small']
    best_model = None
    best_score = 0
    
    for model_size in models_to_test:
        print(f"\n   Testing {model_size} model...")
        
        try:
            custom_config = {
                'whisper': {
                    'model_size': model_size,
                    'language': None  # Auto-detect
                }
            }
            
            analyzer = SmartMeetingAnalyzer()
            results = analyzer.quick_analysis(audio_file)
            
            if results['status'] in ['quick_analysis_complete', 'success']:
                transcription = results.get('transcription_results', [])
                
                # Calculate quality metrics
                total_segments = len(transcription)
                failed_segments = sum(1 for r in transcription if '[Transcription failed]' in r.get('text', ''))
                low_conf_segments = sum(1 for r in transcription if r.get('confidence', 1.0) < 0.5)
                
                success_rate = (total_segments - failed_segments) / total_segments if total_segments > 0 else 0
                conf_score = 1 - (low_conf_segments / total_segments) if total_segments > 0 else 0
                overall_score = (success_rate + conf_score) / 2
                
                print(f"      ✅ Success Rate: {success_rate:.1%}")
                print(f"      ✅ Confidence Score: {conf_score:.1%}")
                print(f"      ✅ Overall Score: {overall_score:.1%}")
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_model = model_size
                
                # Show sample transcription
                if transcription:
                    print(f"      📝 Sample: \"{transcription[0].get('text', 'N/A')[:60]}...\"")
                    
            else:
                print(f"      ❌ Model {model_size} failed")
                
        except Exception as e:
            print(f"      ❌ Error testing {model_size}: {e}")
    
    if best_model:
        print(f"\n   🏆 Best Model: {best_model} (score: {best_score:.1%})")
    
    # 3. Analyze transcription patterns
    print("\n3️⃣ TRANSCRIPTION PATTERN ANALYSIS:")
    
    try:
        analyzer = SmartMeetingAnalyzer()
        results = analyzer.quick_analysis(audio_file)
        
        if results.get('transcription_results'):
            transcription = results['transcription_results']
            
            # Analyze patterns
            total_text = " ".join([r.get('text', '') for r in transcription])
            
            # Count issues
            uncertain_markers = total_text.count('[?]')
            failed_transcriptions = sum(1 for r in transcription if '[Transcription failed]' in r.get('text', ''))
            very_short_segments = sum(1 for r in transcription if len(r.get('text', '')) < 10)
            
            print(f"      • Total Segments: {len(transcription)}")
            print(f"      • Uncertain Markers [?]: {uncertain_markers}")
            print(f"      • Failed Transcriptions: {failed_transcriptions}")
            print(f"      • Very Short Segments: {very_short_segments}")
            
            # Look for patterns in failed transcriptions
            common_issues = []
            
            if uncertain_markers > len(transcription) * 0.3:
                common_issues.append("High uncertainty - audio quality or clarity issues")
            
            if failed_transcriptions > len(transcription) * 0.1:
                common_issues.append("Many failed transcriptions - severe audio issues")
            
            if very_short_segments > len(transcription) * 0.4:
                common_issues.append("Many short segments - possible speaker separation issues")
            
            # Check for non-English text
            import re
            non_english = re.findall(r'[^\x00-\x7F]+', total_text)
            if non_english:
                common_issues.append(f"Non-English text detected: {set(non_english)}")
            
            if common_issues:
                print(f"\n      🚨 Identified Issues:")
                for issue in common_issues:
                    print(f"         • {issue}")
            
    except Exception as e:
        print(f"   ❌ Pattern analysis failed: {e}")
    
    # 4. Recommendations
    print("\n4️⃣ IMPROVEMENT RECOMMENDATIONS:")
    print("   🎯 Immediate Actions:")
    
    # Audio quality improvements
    if quality.get('overall_quality_score', 1.0) < 0.7:
        print("      • Enhance audio quality first:")
        print("        python scripts/enhance_audio.py data/input/input.mp3")
    
    # Model improvements
    if best_model and best_model != 'small':
        print(f"      • Switch to {best_model} model for better results")
        print(f"        Edit config.py: WHISPER_CONFIG['model_size'] = '{best_model}'")
    elif best_model == 'small':
        print("      • Try 'medium' or 'large' model for better accuracy:")
        print("        Edit config.py: WHISPER_CONFIG['model_size'] = 'medium'")
    
    # Transcription settings
    print("      • Enable language auto-detection:")
    print("        Edit config.py: WHISPER_CONFIG['language'] = None")
    
    print("      • Use high-accuracy analysis:")
    print("        python scripts/high_accuracy_analysis.py data/input/input.mp3")
    
    # Audio source improvements
    print("\n   📋 Future Recording Tips:")
    print("      • Use WAV format instead of MP3")
    print("      • Record in quiet environment")
    print("      • Ensure speakers are close to microphone")
    print("      • Avoid overlapping speech")
    print("      • Speak clearly and at moderate pace")
    print("      • Use external microphone if possible")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_transcription.py <audio_file>")
        print("Example: python debug_transcription.py data/input/input.mp3")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)
    
    analyze_transcription_quality(audio_file)