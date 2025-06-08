#!/usr/bin/env python3
"""
Custom analysis with optimized settings for accuracy
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.meeting_analyzer import SmartMeetingAnalyzer

def analyze_with_high_accuracy(audio_file):
    """
    Run analysis with settings optimized for accuracy
    """
    print("🎯 Running high-accuracy analysis...")
    
    # Custom configuration for better accuracy
    custom_config = {
        'whisper': {
            'model_size': 'small',  # Better than base
            'language': None,       # Auto-detect
        },
        'speaker': {
            'similarity_threshold': 0.7,  # More sensitive speaker detection
            'min_segment_duration': 0.5   # Shorter segments for better accuracy
        }
    }
    
    try:
        analyzer = SmartMeetingAnalyzer()
        results = analyzer.analyze_with_custom_config(audio_file, custom_config)
        
        print("\n📊 High-Accuracy Analysis Results:")
        print(f"Status: {results['status']}")
        
        if results['status'] == 'success':
            proc_info = results['processing_info']
            print(f"Speakers: {proc_info['unique_speakers']}")
            print(f"Segments: {proc_info['transcribed_segments']}/{proc_info['total_segments']}")
            
            # Show output files
            output_files = results.get('output_files', {})
            if 'transcript' in output_files:
                print(f"\n📄 Transcript saved to: {output_files['transcript']}")
                
                # Show preview of transcript
                try:
                    with open(output_files['transcript'], 'r') as f:
                        lines = f.readlines()
                    
                    print(f"\n📄 Transcript Preview (first 10 lines):")
                    for line in lines[:10]:
                        if line.strip() and not line.startswith('='):
                            print(f"   {line.strip()}")
                except:
                    pass
            
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python high_accuracy_analysis.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    results = analyze_with_high_accuracy(audio_file)