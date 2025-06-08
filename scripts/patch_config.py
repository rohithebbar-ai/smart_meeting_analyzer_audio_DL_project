#!/usr/bin/env python3
"""
Emergency patch for config.py
"""

import os
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent
config_path = project_root / "config.py"

# Complete working config content
CONFIG_CONTENT = '''"""
Configuration file for Smart Meeting Analyzer
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Audio processing settings
AUDIO_CONFIG = {
    'sample_rate': 16000,  # Standard for speech
    'frame_length': 1024,
    'hop_length': 512,
    'n_mels': 128,
    'vad_aggressiveness': 2  # WebRTC VAD aggressiveness (0-3)
}

# Voice Activity Detection settings
VAD_CONFIG = {
    'frame_duration_ms': 30,  # Frame duration in milliseconds
    'energy_threshold_factor': 0.6,  # Multiplier for dynamic energy threshold
    'min_speech_duration': 0.5,  # Minimum speech segment duration in seconds
    'min_silence_duration': 0.3   # Minimum silence gap between segments
}

# Speaker detection settings
SPEAKER_CONFIG = {
    'min_segment_duration': 1.0,  # Minimum segment duration for speaker analysis
    'similarity_threshold': 0.75,  # Threshold for speaker similarity (0-1)
    'clustering_method': 'agglomerative',  # 'kmeans' or 'agglomerative'
    'max_speakers': 10  # Maximum number of speakers to detect
}

# Whisper transcription settings
WHISPER_CONFIG = {
    'model_size': 'small',  # 'tiny', 'base', 'small', 'medium', 'large'
    'language': 'en',  # Language code or None for auto-detection
    'device': 'auto'   # 'cpu', 'cuda', or 'auto'
}

# Output settings
OUTPUT_CONFIG = {
    'generate_transcript': True,
    'generate_summary': True,
    'include_timestamps': True,
    'include_speaker_stats': True,
    'save_audio_segments': False,  # Save individual speaker segments
    'output_format': 'txt'  # 'txt', 'json', 'csv'
}

# Quality assessment thresholds
QUALITY_CONFIG = {
    'min_energy_threshold': 0.01,
    'max_noise_threshold': 0.005,
    'min_clarity_threshold': 1000,
    'quality_score_weights': {
        'energy': 0.25,
        'clarity': 0.30,
        'noise': 0.25,
        'consistency': 0.20
    }
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True
}
'''

def main():
    print("🔧 Patching config.py with complete working configuration...")
    
    try:
        # Backup existing config
        if config_path.exists():
            backup_path = config_path.with_suffix('.py.backup')
            with open(config_path, 'r') as f:
                backup_content = f.read()
            with open(backup_path, 'w') as f:
                f.write(backup_content)
            print(f"✅ Backed up original config to {backup_path}")
        
        # Write new config
        with open(config_path, 'w') as f:
            f.write(CONFIG_CONTENT)
        
        print("✅ Config patched successfully!")
        
        # Test import
        import sys
        sys.path.append(str(project_root))
        import config
        
        print("✅ Config imports successfully!")
        print(f"✅ AUDIO_CONFIG keys: {list(config.AUDIO_CONFIG.keys())}")
        print(f"✅ SPEAKER_CONFIG keys: {list(config.SPEAKER_CONFIG.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Patch failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 You can now run: python main.py --quick data/input/input.mp3")
    exit(0 if success else 1)
