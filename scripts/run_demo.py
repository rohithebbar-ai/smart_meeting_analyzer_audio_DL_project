#!/usr/bin/env python3
"""
Demo script for Smart Meeting Analyzer
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.meeting_analyzer import SmartMeetingAnalyzer
from src.utils import setup_logging
from config import INPUT_DIR

def run_demo():
    """Run demo with sample audio or user-provided file"""
    
    logger = setup_logging("Demo")
    
    print("🎤 Smart Meeting Analyzer Demo")
    print("=" * 40)
    
    # Check for demo audio files
    demo_files = list(INPUT_DIR.glob("*.wav")) + list(INPUT_DIR.glob("*.mp3"))
    
    if not demo_files:
        print("❌ No audio files found in data/input/")
        print("Please add a sample audio file (WAV or MP3) to data/input/")
        print("You can download sample meeting audio from:")
        print("- https://www.voiptroubleshooter.com/open_speech/american.html")
        print("- Or record a short conversation")
        return
    
    print(f"📁 Found {len(demo_files)} audio file(s):")
    for i, file in enumerate(demo_files):
        print(f"   {i+1}. {file.name}")
    
    # Let user choose or use first file
    if len(demo_files) == 1:
        selected_file = demo_files[0]
        print(f"\n🎯 Using: {selected_file.name}")
    else:
        try:
            choice = input(f"\nSelect file (1-{len(demo_files)}) or press Enter for first: ").strip()
            if choice:
                idx = int(choice) - 1
                selected_file = demo_files[idx]
            else:
                selected_file = demo_files[0]
            print(f"🎯 Using: {selected_file.name}")
        except (ValueError, IndexError):
            selected_file = demo_files[0]
            print(f"🎯 Using default: {selected_file.name}")
    
    # Ask for analysis type
    print("\nChoose analysis type:")
    print("1. Quick Demo (fast, limited analysis)")
    print("2. Full Analysis (comprehensive, slower)")
    
    analysis_choice = input("Enter choice (1 or 2) or press Enter for quick: ").strip()
    
    try:
        # Initialize analyzer
        print("\n🚀 Initializing Smart Meeting Analyzer...")
        analyzer = SmartMeetingAnalyzer()
        
        # Run analysis
        if analysis_choice == "2":
            print("🔄 Running full analysis...")
            print("⏳ This may take several minutes depending on audio length...")
            results = analyzer.analyze_meeting(str(selected_file))
        else:
            print("⚡ Running quick demo analysis...")
            results = analyzer.quick_analysis(str(selected_file))
        
        # Display results
        print_demo_results(results)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure audio file is valid (WAV/MP3)")
        print("2. Check that all dependencies are installed")
        print("3. Try with a different audio file")
        return

def print_demo_results(results):
    """Print demo results in a user-friendly format"""
    
    print("\n" + "🎉 " + "="*50)
    print("DEMO RESULTS")
    print("="*54)
    
    # Status
    status = results.get('status', 'unknown')
    if status == 'success':
        print("✅ Analysis completed successfully!")
    elif status == 'quick_analysis_complete':
        print("⚡ Quick analysis completed!")
    else:
        print(f"⚠️  Analysis status: {status}")
    
    # Basic info
    proc_info = results.get('processing_info', {})
    if 'total_duration' in proc_info:
        duration = proc_info['total_duration']
        print(f"⏱️  Audio Duration: {duration:.1f} seconds")
    
    speakers = proc_info.get('unique_speakers', 0)
    print(f"👥 Speakers Detected: {speakers}")
    
    # Audio quality
    quality = results.get('quality_assessment', {})
    if quality:
        quality_label = quality.get('quality_label', 'Unknown')
        quality_score = quality.get('overall_quality_score', 0)
        print(f"🎵 Audio Quality: {quality_label} ({quality_score:.2f}/1.0)")
        
        recommendations = quality.get('recommendations', [])
        if recommendations and not any('good' in rec.lower() for rec in recommendations):
            print("💡 Quality Recommendations:")
            for rec in recommendations[:2]:  # Show top 2
                print(f"   • {rec}")
    
    # Transcription preview
    transcription = results.get('transcription_results', [])
    if transcription:
        print(f"\n💬 Transcription Preview:")
        print("-" * 30)
        
        for i, segment in enumerate(transcription[:3]):  # Show first 3
            timestamp = segment.get('timestamp', '00:00')
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            
            if text and text != '[Transcription failed]':
                # Truncate long text
                if len(text) > 70:
                    text = text[:67] + "..."
                print(f"[{timestamp}] {speaker}: {text}")
        
        if len(transcription) > 3:
            print(f"... and {len(transcription) - 3} more segments")
    
    # Output files
    output_files = results.get('output_files', {})
    if output_files:
        print(f"\n📄 Generated Files:")
        for file_type, file_path in output_files.items():
            if isinstance(file_path, str) and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                size_kb = file_size / 1024
                print(f"   • {file_type.title()}: {file_path} ({size_kb:.1f} KB)")
            elif isinstance(file_path, dict):
                print(f"   • {file_type.title()}: {len(file_path)} files")
    
    # Speaker statistics
    speaker_stats = results.get('speaker_stats', {})
    if speaker_stats:
        print(f"\n👥 Speaker Statistics:")
        print("-" * 20)
        
        for speaker, stats in speaker_stats.items():
            speaking_time = stats.get('total_speaking_time', 0)
            segments = stats.get('number_of_segments', 0)
            
            minutes = int(speaking_time // 60)
            seconds = int(speaking_time % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            
            print(f"{speaker}: {time_str} ({segments} segments)")
    
    print("\n" + "="*54)
    
    # Next steps
    if status in ['success', 'quick_analysis_complete']:
        print("🎯 What you can do next:")
        
        if output_files.get('transcript'):
            print(f"   • View full transcript: {output_files['transcript']}")
        
        if output_files.get('json'):
            print(f"   • Examine JSON data: {output_files['json']}")
        
        if output_files.get('audio_visualization'):
            print(f"   • Check audio visualization: {output_files['audio_visualization']}")
        
        if status == 'quick_analysis_complete':
            print("   • Run full analysis: python main.py [audio_file]")
        
        print("   • Try with different audio files")
        print("   • Examine the code to understand the pipeline")
    
    print("\n💡 Learning Tips:")
    print("   • Check the generated visualizations to understand spectrograms")
    print("   • Compare transcript accuracy with original audio")
    print("   • Try different audio qualities to see quality assessment")
    print("   • Examine speaker segmentation for multi-speaker recordings")

def create_sample_audio():
    """Create a simple sample audio file for testing"""
    try:
        import numpy as np
        import soundfile as sf
        
        print("🎵 Creating sample audio for testing...")
        
        # Generate simple test audio: two tones representing different "speakers"
        duration = 10  # seconds
        sample_rate = 16000
        
        t = np.linspace(0, duration, duration * sample_rate, False)
        
        # Speaker 1: Lower frequency tone for first 3 seconds
        speaker1_freq = 440  # A4
        speaker1_audio = np.sin(2 * np.pi * speaker1_freq * t[:3 * sample_rate]) * 0.3
        
        # Silence for 1 second
        silence1 = np.zeros(1 * sample_rate)
        
        # Speaker 2: Higher frequency tone for 3 seconds
        speaker2_freq = 880  # A5
        speaker2_audio = np.sin(2 * np.pi * speaker2_freq * t[:3 * sample_rate]) * 0.3
        
        # More silence
        silence2 = np.zeros(1 * sample_rate)
        
        # Speaker 1 again for 2 seconds
        speaker1_again = np.sin(2 * np.pi * speaker1_freq * t[:2 * sample_rate]) * 0.3
        
        # Combine all
        full_audio = np.concatenate([
            speaker1_audio, silence1, speaker2_audio, silence2, speaker1_again
        ])
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.02, len(full_audio))
        full_audio += noise
        
        # Save to input directory
        sample_path = INPUT_DIR / "sample_meeting.wav"
        sf.write(sample_path, full_audio, sample_rate)
        
        print(f"✅ Sample audio created: {sample_path}")
        print("   This is a simple test audio with two 'speakers' (different tones)")
        return str(sample_path)
        
    except Exception as e:
        print(f"❌ Failed to create sample audio: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Meeting Analyzer Demo")
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create sample audio file for testing')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis only')
    parser.add_argument('--file', type=str,
                       help='Specific audio file to analyze')
    
    args = parser.parse_args()
    
    if args.create_sample:
        sample_file = create_sample_audio()
        if sample_file:
            print(f"\n🎯 Sample created! You can now run:")
            print(f"   python scripts/run_demo.py")
            print(f"   python main.py {sample_file}")
    else:
        run_demo()