"""
Utility functions for smart meeting analyzer
"""
import os
import logging
import numpy as np
import librosa
import soundfile as sf 
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt



from config import LOGGING_CONFIG, OUTPUT_DIR

def setup_logging(name: str = 'MeetingAnalyzer') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG['format'])
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if enabled
    if LOGGING_CONFIG['file_logging']:
        log_file = OUTPUT_DIR / f"meeting_analyzer_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_audio(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0 :
            audio = audio / np.max(np.abs(audio))
            
        return audio, sr
    
    except Exception as e:
        raise Exception(f"Error loading audio file {file_path}: {str(e)}")
    
def save_audio_segment(audio: np.ndarray, sr: int, start_time: float, 
                      end_time: float, output_path: str) -> None:
    """Save audio segment to file"""
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = audio[start_sample:end_sample]
    
    sf.write(output_path, segment, sr)
    
def format_timestamp(seconds: float) -> str:
    """Convert seconds to readable timestamp format (MM:SS)"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def calculate_speaking_stats(segments: List[Dict]) -> Dict:
    """Calculate speaking statistics from segments"""
    if not segments:
        return {}
    
    stats = {}
    total_duration = max(seg['end_time'] for seg in segments)
    
    # Group by speaker
    speaker_times = {}
    for segment in segments:
        speaker = segment ['speaker']
        duration = segment['end_time'] - segment['start_time']
        
        if speaker not in speaker_times:
            speaker_times[speaker] = []
        speaker_times[speaker].append(duration)
        
    # Calculate stats for each speaker
    for speaker, times in speaker_times.items():
        total_time = sum(times)
        avg_segment_length = np.mean(times)
        num_segments = len(times)
        
        stats[speaker] = {
            'total_speaking_time': total_time,
            'percentage_of_meeting': (total_time / total_duration) * 100,
            'number_of_segments': num_segments,
            'average_segment_length': avg_segment_length,
            'speaking_time_formatted': format_timestamp(total_time)
        }
    
    return stats

def create_visualization(segments: List[Dict], quality_metrics: Dict, 
                        output_path: str) -> None:
    """Create visualization of meeting analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Speaker timeline
    speakers = list(set(seg['speaker'] for seg in segments))
    colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
    speaker_colors = dict(zip(speakers, colors))
    
    for segment in segments:
        speaker = segment['speaker']
        start = segment['start_time']
        duration = segment['end_time'] - segment['start_time']
        ax1.barh(speaker, duration, left=start, 
                color=speaker_colors[speaker], alpha=0.7)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title('Speaker Timeline')
    ax1.grid(True, alpha=0.3)
    
    # 2. Speaking time distribution
    stats = calculate_speaking_stats(segments)
    if stats:
        speakers = list(stats.keys())
        times = [stats[speaker]['total_speaking_time'] for speaker in speakers]
        
        ax2.pie(times, labels=speakers, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Speaking Time Distribution')
    
    # 3. Quality metrics
    if quality_metrics:
        metrics = ['energy_level', 'clarity', 'noise_level']
        values = [quality_metrics.get(metric, 0) for metric in metrics]
        
        ax3.bar(metrics, values, color=['green', 'blue', 'red'], alpha=0.7)
        ax3.set_title('Audio Quality Metrics')
        ax3.set_ylabel('Score')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Segment lengths histogram
    segment_lengths = [seg['end_time'] - seg['start_time'] for seg in segments]
    ax4.hist(segment_lengths, bins=20, alpha=0.7, color='skyblue')
    ax4.set_xlabel('Segment Length (seconds)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Segment Lengths')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def validate_audio_file(file_path: str) -> bool:
    """Validate if file is a supported audio format"""
    supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension not in supported_formats:
        return False
    
    if not os.path.exists(file_path):
        return False
    
    # Try to load the file
    try:
        librosa.load(file_path, sr=None, duration=1.0)  # Load just 1 second to test
        return True
    except:
        return False

def estimate_processing_time(audio_duration: float) -> str:
    """Estimate processing time based on audio duration"""
    # Rough estimates based on typical processing speeds
    base_time = 5  # Base processing time in seconds
    time_per_minute = 15  # Additional seconds per minute of audio
    
    estimated_seconds = base_time + (audio_duration / 60) * time_per_minute
    
    if estimated_seconds < 60:
        return f"~{int(estimated_seconds)} seconds"
    else:
        minutes = int(estimated_seconds // 60)
        seconds = int(estimated_seconds % 60)
        return f"~{minutes}m {seconds}s"

def create_output_filename(input_path: str, suffix: str = "", 
                          extension: str = "txt") -> str:
    """Create output filename based on input file"""
    input_name = Path(input_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if suffix:
        filename = f"{input_name}_{suffix}_{timestamp}.{extension}"
    else:
        filename = f"{input_name}_analysis_{timestamp}.{extension}"
    
    return str(OUTPUT_DIR / filename)

def json_serializer(obj):
    """Custom JSON serializer for numpy and other non-serializable types"""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # For numpy scalar types
        return obj.item()
    else:
        # For any other non-serializable type, convert to string
        return str(obj)