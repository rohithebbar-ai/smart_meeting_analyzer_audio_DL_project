#!/usr/bin/env python3
"""
Audio preprocessing for better transcription
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import sys

def enhance_audio_for_transcription(input_path, output_path=None):
    """
    Enhance audio quality for better transcription
    """
    if output_path is None:
        output_path = Path(input_path).with_suffix('.enhanced.wav')
    
    print(f"🎵 Enhancing audio: {input_path}")
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=16000)
    
    # 1. Normalize volume
    audio = librosa.util.normalize(audio)
    
    # 2. Reduce noise (simple spectral gating)
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    
    # Estimate noise floor
    noise_floor = np.percentile(magnitude, 20)
    
    # Apply spectral gating
    mask = magnitude > (noise_floor * 2)
    stft_cleaned = stft * mask
    
    # Reconstruct audio
    audio_cleaned = librosa.istft(stft_cleaned)
    
    # 3. Apply mild compression to even out volume levels
    audio_compressed = np.tanh(audio_cleaned * 3) / 3
    
    # Save enhanced audio
    sf.write(output_path, audio_compressed, sr)
    
    print(f"✅ Enhanced audio saved to: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python enhance_audio.py <input_audio_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    enhanced_file = enhance_audio_for_transcription(input_file)
    
    print(f"\n🎯 Now run transcription on enhanced audio:")
    print(f"python main.py --quick {enhanced_file}")