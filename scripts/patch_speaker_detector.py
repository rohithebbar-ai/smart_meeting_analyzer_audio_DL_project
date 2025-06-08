#!/usr/bin/env python3
"""
Emergency patch for speaker_detector.py
"""

from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent
speaker_detector_path = project_root / "src" / "speaker_detector.py"

def patch_speaker_detector():
    print("🔧 Patching speaker_detector.py...")
    
    try:
        # Read current content
        with open(speaker_detector_path, 'r') as f:
            content = f.read()
        
        # Fix the __init__ method
        old_init = '''    def __init__(self):
        self.logger = setup_logging("SpeakerDetector")
        self.encoder = VoiceEncoder()
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        self.min_segment_duration = SPEAKER_CONFIG['min_segment_duration']
        self.similarity_threshold = SPEAKER_CONFIG['similarity_threshold']
        self.max_speakers = SPEAKER_CONFIG['max_speakers']
        
        self.logger.info("Resemblyzer VoiceEncoder initialized")'''
        
        new_init = '''    def __init__(self):
        self.logger = setup_logging("SpeakerDetector")
        self.encoder = VoiceEncoder()
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        
        # Initialize speaker detection parameters with safe defaults
        self.min_segment_duration = SPEAKER_CONFIG.get('min_segment_duration', 1.0)
        self.similarity_threshold = SPEAKER_CONFIG.get('similarity_threshold', 0.75)
        self.max_speakers = SPEAKER_CONFIG.get('max_speakers', 10)
        
        self.logger.info("Resemblyzer VoiceEncoder initialized")'''
        
        # Apply the patch
        if old_init in content:
            content = content.replace(old_init, new_init)
            print("✅ Patched __init__ method")
        elif "self.similarity_threshold = SPEAKER_CONFIG['similarity_threshold']" in content:
            # Alternative fix if exact match doesn't work
            content = content.replace(
                "self.similarity_threshold = SPEAKER_CONFIG['similarity_threshold']",
                "self.similarity_threshold = SPEAKER_CONFIG.get('similarity_threshold', 0.75)"
            )
            content = content.replace(
                "self.min_segment_duration = SPEAKER_CONFIG['min_segment_duration']",
                "self.min_segment_duration = SPEAKER_CONFIG.get('min_segment_duration', 1.0)"
            )
            content = content.replace(
                "self.max_speakers = SPEAKER_CONFIG['max_speakers']",
                "self.max_speakers = SPEAKER_CONFIG.get('max_speakers', 10)"
            )
            print("✅ Applied alternative patch")
        else:
            print("⚠️  Could not find exact pattern, applying safe defaults...")
            # If we can't find the exact pattern, add safe initialization
            if "def __init__(self):" in content:
                content = content.replace(
                    "def __init__(self):",
                    '''def __init__(self):
        self.logger = setup_logging("SpeakerDetector")
        self.encoder = VoiceEncoder()
        self.sample_rate = AUDIO_CONFIG.get('sample_rate', 16000)
        
        # Initialize speaker detection parameters with safe defaults
        self.min_segment_duration = SPEAKER_CONFIG.get('min_segment_duration', 1.0)
        self.similarity_threshold = SPEAKER_CONFIG.get('similarity_threshold', 0.75)
        self.max_speakers = SPEAKER_CONFIG.get('max_speakers', 10)
        
        self.logger.info("Resemblyzer VoiceEncoder initialized")
        return  # Skip the rest of original __init__
    
    def __init___original(self):'''
                )
        
        # Write the patched content
        with open(speaker_detector_path, 'w') as f:
            f.write(content)
        
        print("✅ speaker_detector.py patched successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Patch failed: {e}")
        return False

def test_import():
    """Test if the patched module works"""
    try:
        import sys
        sys.path.append(str(project_root))
        
        # Remove any cached imports
        if 'src.speaker_detector' in sys.modules:
            del sys.modules['src.speaker_detector']
        
        from src.speaker_detector import SpeakerDetector
        detector = SpeakerDetector()
        
        print(f"✅ SpeakerDetector initialized successfully!")
        print(f"   similarity_threshold: {detector.similarity_threshold}")
        print(f"   min_segment_duration: {detector.min_segment_duration}")
        print(f"   max_speakers: {detector.max_speakers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🔧 Emergency SpeakerDetector Patch")
    print("=" * 40)
    
    if patch_speaker_detector():
        if test_import():
            print("\n🎉 Patch successful! You can now run:")
            print("  python main.py --quick data/input/input.mp3")
            return True
    
    print("\n❌ Patch failed. Manual fix needed.")
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)