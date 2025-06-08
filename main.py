#!/usr/bin/env python3
"""
Smart Meeting Analyzer - Main Entry Point

Usage:
    python main.py path/to/audio/file.wav
    python main.py --quick path/to/audio/file.wav
    python main.py --demo

Examples:
    python main.py data/input/meeting.wav
    python main.py --quick data/input/meeting.wav --output-prefix "team_standup"
    python main.py --demo
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.meeting_analyzer import SmartMeetingAnalyzer
from src.utils import setup_logging, validate_audio_file
from config import INPUT_DIR, OUTPUT_DIR

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Smart Meeting Analyzer - AI-powered meeting transcription and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s meeting.wav                    # Full analysis
    %(prog)s --quick meeting.wav            # Quick demo analysis  
    %(prog)s --demo                         # Run with sample audio
    %(prog)s meeting.wav --output-prefix team_standup
        """
    )
    
    parser.add_argument(
        'audio_file',
        nargs='?',
        help='Path to audio file for analysis'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick analysis (faster, for demo purposes)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with sample audio'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        help='Prefix for output files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("Main")
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.demo:
        audio_file = _get_demo_audio()
        if not audio_file:
            logger.error("Demo audio not found. Please add a sample audio file to data/input/")
            return 1
    elif args.audio_file:
        audio_file = args.audio_file
        
        # Check if file exists, if not try in input directory
        if not os.path.exists(audio_file):
            input_path = INPUT_DIR / audio_file
            if input_path.exists():
                audio_file = str(input_path)
            else:
                logger.error(f"Audio file not found: {audio_file}")
                return 1
    else:
        parser.print_help()
        return 1
    
    # Validate audio file
    if not validate_audio_file(audio_file):
        logger.error(f"Invalid audio file: {audio_file}")
        return 1
    
    try:
        # Initialize analyzer
        logger.info("Initializing Smart Meeting Analyzer...")
        analyzer = SmartMeetingAnalyzer()
        
        # Run analysis
        if args.quick:
            logger.info("Running quick analysis...")
            results = analyzer.quick_analysis(audio_file)
        else:
            logger.info("Running full meeting analysis...")
            results = analyzer.analyze_meeting(
                audio_file_path=audio_file,
                output_prefix=args.output_prefix
            )
        
        # Print results summary
        _print_results_summary(results, logger)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def _get_demo_audio() -> str:
    """Find demo audio file"""
    demo_files = [
        'sample_meeting.wav',
        'demo.wav',
        'meeting.wav'
    ]
    
    for filename in demo_files:
        demo_path = INPUT_DIR / filename
        if demo_path.exists():
            return str(demo_path)
    
    return None

def _print_results_summary(results: dict, logger):
    """Print analysis results summary"""
    
    print("\n" + "="*60)
    print("SMART MEETING ANALYZER - RESULTS SUMMARY")
    print("="*60)
    
    status = results.get('status', 'unknown')
    
    if status == 'success':
        proc_info = results.get('processing_info', {})
        
        print(f"✅ Analysis Status: SUCCESS")
        print(f"📁 Input File: {results.get('input_file', 'Unknown')}")
        print(f"⏱️  Total Duration: {proc_info.get('total_duration', 0):.1f} seconds")
        print(f"👥 Speakers Detected: {proc_info.get('unique_speakers', 0)}")
        print(f"📝 Segments Transcribed: {proc_info.get('transcribed_segments', 0)}/{proc_info.get('total_segments', 0)}")
        
        # Audio quality
        quality = results.get('quality_assessment', {})
        quality_score = quality.get('overall_quality_score', 0)
        quality_label = quality.get('quality_label', 'Unknown')
        
        print(f"🎵 Audio Quality: {quality_label} ({quality_score:.2f}/1.0)")
        
        # Output files
        output_files = results.get('output_files', {})
        if output_files:
            print(f"\n📄 Generated Files:")
            for file_type, file_path in output_files.items():
                if isinstance(file_path, str):
                    print(f"   • {file_type.title()}: {file_path}")
                elif isinstance(file_path, dict):
                    print(f"   • {file_type.title()}: {len(file_path)} files")
        
        # Quick transcript preview
        transcription = results.get('transcription_results', [])
        if transcription:
            print(f"\n💬 Transcript Preview (first 2 segments):")
            for i, segment in enumerate(transcription[:2]):
                timestamp = segment.get('timestamp', '00:00')
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '')
                
                # Truncate long text
                if len(text) > 80:
                    text = text[:77] + "..."
                
                print(f"   [{timestamp}] {speaker}: {text}")
            
            if len(transcription) > 2:
                print(f"   ... and {len(transcription) - 2} more segments")
    
    elif status == 'no_speech_detected':
        print(f"⚠️  Analysis Status: NO SPEECH DETECTED")
        print(f"📁 Input File: {results.get('input_file', 'Unknown')}")
        print("   No voice activity found in the audio file.")
        
        # Still show quality assessment
        quality = results.get('quality_assessment', {})
        if quality:
            quality_label = quality.get('quality_label', 'Unknown')
            print(f"🎵 Audio Quality: {quality_label}")
            
            recommendations = quality.get('recommendations', [])
            if recommendations:
                print("💡 Recommendations:")
                for rec in recommendations:
                    print(f"   • {rec}")
    
    elif status == 'quick_analysis_complete':
        print(f"⚡ Analysis Status: QUICK DEMO COMPLETE")
        print(f"📁 Input File: {results.get('input_file', 'Unknown')}")
        
        proc_info = results.get('processing_info', {})
        print(f"⏱️  Analyzed Duration: {proc_info.get('analyzed_duration', 0):.1f} seconds")
        print(f"📝 Segments Processed: {proc_info.get('segments_processed', 0)}")
        
        print("\n💡 Note: This was a quick demo analysis.")
        print("   For full processing, run without --quick flag.")
    
    else:
        print(f"❌ Analysis Status: {status.upper()}")
        print(f"📁 Input File: {results.get('input_file', 'Unknown')}")
    
    print("\n" + "="*60)
    
    # Additional tips
    output_files = results.get('output_files', {})
    if output_files.get('transcript'):
        print(f"💡 Tip: View the full transcript at:")
        print(f"   {output_files['transcript']}")
    
    if output_files.get('json'):
        print(f"💡 Tip: Programmatic access via JSON at:")
        print(f"   {output_files['json']}")
    
    print()

if __name__ == "__main__":
    sys.exit(main())