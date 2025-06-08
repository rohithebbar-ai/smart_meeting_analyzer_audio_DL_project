"""
Audio preprocessing and quality assessment module
"""

import numpy as np
import librosa
import librosa.display
import webrtcvad
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from config import AUDIO_CONFIG, VAD_CONFIG, QUALITY_CONFIG
from src.utils import setup_logging

class AudioProcessor:
    """Handles audio preprocessing and quality assessment"""
    
    def __init__(self):
        self.logger = setup_logging("AudioProcessor")
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        self.frame_length = AUDIO_CONFIG['frame_length']
        self.hop_length = AUDIO_CONFIG['hop_length']
        
        # Initialize WebRTC VAD - fix the config access
        try:
            vad_aggressiveness = AUDIO_CONFIG.get('vad_aggressiveness', 2)
            self.vad = webrtcvad.Vad(vad_aggressiveness)
        except Exception as e:
            self.logger.warning(f"Failed to initialize WebRTC VAD: {e}")
            self.vad = None
        
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio for analysis
        
        Args:
            audio: Raw audio array
            sr: Sample rate
            
        Returns:
            Preprocessed audio array
        """
        self.logger.info("Preprocessing audio...")
        
        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            self.logger.info(f"Resampled audio from {sr}Hz to {self.sample_rate}Hz")
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Basic noise reduction (simple high-pass filter)
        audio = self._apply_high_pass_filter(audio)
        
        return audio
    
    def _apply_high_pass_filter(self, audio: np.ndarray, cutoff_freq: int = 80) -> np.ndarray:
        """Apply simple high-pass filter to remove low-frequency noise"""
        from scipy.signal import butter, filtfilt
        
        # Design filter
        nyquist = self.sample_rate // 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(1, normal_cutoff, btype='high', analog=False)
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio)
        return filtered_audio
    
    def detect_voice_activity(self, audio: np.ndarray) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Detect voice activity in audio using energy-based and VAD approaches
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            List of (start_time, end_time) tuples for speech segments
            Energy array for visualization
        """
        self.logger.info("Detecting voice activity...")
        
        # Method 1: Energy-based detection
        speech_segments_energy = self._energy_based_vad(audio)
        
        # Method 2: WebRTC VAD (more robust) - with fallback
        try:
            speech_segments_vad = self._webrtc_vad(audio)
        except Exception as e:
            self.logger.warning(f"WebRTC VAD failed: {e}, using energy-based detection")
            speech_segments_vad = []
        
        # Combine results (use WebRTC as primary, energy as backup)
        if speech_segments_vad and len(speech_segments_vad) > 0:
            final_segments = speech_segments_vad
            self.logger.info(f"Using WebRTC VAD: found {len(final_segments)} segments")
        else:
            final_segments = speech_segments_energy
            self.logger.info(f"Using energy-based VAD: found {len(final_segments)} segments")
        
        # Clean up segments (merge close ones, remove short ones)
        cleaned_segments = self._clean_segments(final_segments)
        
        # Calculate energy for visualization
        energy = librosa.feature.rms(y=audio, frame_length=self.frame_length, 
                                   hop_length=self.hop_length)[0]
        
        self.logger.info(f"Final voice activity: {len(cleaned_segments)} segments")
        return cleaned_segments, energy
    
    def _energy_based_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Energy-based voice activity detection"""
        # Calculate frame energy
        energy = librosa.feature.rms(y=audio, frame_length=self.frame_length,
                                   hop_length=self.hop_length)[0]
        
        # Dynamic threshold
        threshold = np.mean(energy) + VAD_CONFIG['energy_threshold_factor'] * np.std(energy)
        
        # Find speech frames
        speech_frames = energy > threshold
        
        # Convert to time segments
        segments = self._frames_to_segments(speech_frames)
        return segments
    
    def _webrtc_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """WebRTC VAD for more robust voice detection"""
        if self.vad is None:
            self.logger.warning("WebRTC VAD not available, falling back to energy-based detection")
            return self._energy_based_vad(audio)
            
        frame_duration_ms = VAD_CONFIG['frame_duration_ms']
        frame_length = int(self.sample_rate * frame_duration_ms / 1000)
        
        # Ensure audio is in correct format for WebRTC VAD
        audio_int16 = (audio * 32767).astype(np.int16)
        
        speech_frames = []
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length]
            
            # WebRTC VAD requires specific frame sizes
            if len(frame) == frame_length:
                try:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                    speech_frames.append(is_speech)
                except Exception as e:
                    # Fallback for frames that VAD can't process
                    self.logger.debug(f"WebRTC VAD frame processing failed: {e}")
                    speech_frames.append(False)
            else:
                speech_frames.append(False)
        
        # Convert to time segments
        segments = self._frames_to_segments(speech_frames, frame_duration_ms / 1000)
        return segments
    
    def _frames_to_segments(self, speech_frames: List[bool], 
                           frame_duration: float = None) -> List[Tuple[float, float]]:
        """Convert frame-level speech detection to time segments"""
        if frame_duration is None:
            frame_duration = self.hop_length / self.sample_rate
        
        segments = []
        start_time = None
        
        for i, is_speech in enumerate(speech_frames):
            current_time = i * frame_duration
            
            if is_speech and start_time is None:
                # Start of speech segment
                start_time = current_time
            elif not is_speech and start_time is not None:
                # End of speech segment
                segments.append((start_time, current_time))
                start_time = None
        
        # Handle case where speech continues to end of audio
        if start_time is not None:
            final_time = len(speech_frames) * frame_duration
            segments.append((start_time, final_time))
        
        return segments
    
    def _clean_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Clean up detected segments"""
        if not segments:
            return segments
        
        cleaned = []
        min_duration = VAD_CONFIG['min_speech_duration']
        min_silence = VAD_CONFIG['min_silence_duration']
        
        for start, end in segments:
            # Remove segments that are too short
            if end - start >= min_duration:
                cleaned.append((start, end))
        
        # Merge segments that are close together
        if len(cleaned) <= 1:
            return cleaned
        
        merged = [cleaned[0]]
        
        for start, end in cleaned[1:]:
            last_end = merged[-1][1]
            
            # If gap is small, merge with previous segment
            if start - last_end <= min_silence:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        
        return merged
    
    def assess_audio_quality(self, audio: np.ndarray) -> Dict:
        """
        Comprehensive audio quality assessment
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Dictionary with quality metrics and recommendations
        """
        self.logger.info("Assessing audio quality...")
        
        # Calculate individual metrics
        energy_score = self._assess_energy_level(audio)
        clarity_score = self._assess_clarity(audio)
        noise_score = self._assess_noise_level(audio)
        consistency_score = self._assess_consistency(audio)
        
        # Calculate overall quality score
        weights = QUALITY_CONFIG['quality_score_weights']
        overall_score = (
            weights['energy'] * energy_score +
            weights['clarity'] * clarity_score +
            weights['noise'] * noise_score +
            weights['consistency'] * consistency_score
        )
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(
            energy_score, clarity_score, noise_score, consistency_score
        )
        
        quality_metrics = {
            'overall_quality_score': overall_score,
            'energy_level': energy_score,
            'clarity': clarity_score,
            'noise_level': noise_score,
            'consistency': consistency_score,
            'quality_label': self._score_to_label(overall_score),
            'recommendations': recommendations
        }
        
        self.logger.info(f"Audio quality assessment: {quality_metrics['quality_label']} "
                        f"(score: {overall_score:.2f})")
        
        return quality_metrics
    
    def _assess_energy_level(self, audio: np.ndarray) -> float:
        """Assess overall energy/volume level"""
        rms_energy = np.mean(librosa.feature.rms(y=audio))
        
        # Score based on optimal energy range
        if rms_energy < 0.01:
            return 0.3  # Too quiet
        elif rms_energy > 0.3:
            return 0.7  # Too loud
        else:
            return 0.9  # Good level
    
    def _assess_clarity(self, audio: np.ndarray) -> float:
        """Assess speech clarity using spectral centroid"""
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        
        # Score based on typical speech clarity range
        if spectral_centroid < 1000:
            return 0.4  # Low clarity
        elif spectral_centroid > 4000:
            return 0.6  # Too bright
        else:
            return 0.9  # Good clarity
    
    def _assess_noise_level(self, audio: np.ndarray) -> float:
        """Assess background noise level"""
        # Estimate noise during quiet moments
        rms = librosa.feature.rms(y=audio)[0]
        noise_threshold = np.percentile(rms, 20)  # Bottom 20% assumed to be noise
        
        if noise_threshold > 0.01:
            return 0.3  # High noise
        elif noise_threshold > 0.005:
            return 0.6  # Moderate noise
        else:
            return 0.9  # Low noise
    
    def _assess_consistency(self, audio: np.ndarray) -> float:
        """Assess audio consistency (stable levels)"""
        rms = librosa.feature.rms(y=audio)[0]
        consistency = 1.0 - (np.std(rms) / np.mean(rms))
        return max(0.0, min(1.0, consistency))
    
    def _score_to_label(self, score: float) -> str:
        """Convert numeric score to quality label"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_quality_recommendations(self, energy: float, clarity: float, 
                                        noise: float, consistency: float) -> List[str]:
        """Generate actionable quality recommendations"""
        recommendations = []
        
        if energy < 0.5:
            recommendations.append("Audio level is low - participants should speak louder or move closer to microphone")
        
        if clarity < 0.5:
            recommendations.append("Audio clarity could be improved - check microphone quality and positioning")
        
        if noise < 0.5:
            recommendations.append("Background noise detected - consider using a quieter environment or noise-canceling microphones")
        
        if consistency < 0.5:
            recommendations.append("Audio levels are inconsistent - ensure stable microphone positioning and consistent speaking volume")
        
        if not recommendations:
            recommendations.append("Audio quality is good! No specific improvements needed.")
        
        return recommendations
    
    def visualize_audio_analysis(self, audio: np.ndarray, energy: np.ndarray, 
                                segments: List[Tuple[float, float]], 
                                output_path: str) -> None:
        """Create visualization of audio analysis"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
        
        # 1. Waveform
        time_axis = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        ax1.plot(time_axis, audio, alpha=0.7)
        ax1.set_title('Audio Waveform')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy with speech segments highlighted
        energy_time = np.linspace(0, len(audio) / self.sample_rate, len(energy))
        ax2.plot(energy_time, energy, color='blue', alpha=0.7)
        
        # Highlight speech segments
        for start, end in segments:
            ax2.axvspan(start, end, alpha=0.3, color='green', label='Speech' if start == segments[0][0] else "")
        
        ax2.set_title('Energy Level with Detected Speech Segments')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('RMS Energy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        img = librosa.display.specshow(mel_spec_db, sr=self.sample_rate, 
                                     hop_length=self.hop_length, x_axis='time', 
                                     y_axis='mel', ax=ax3)
        ax3.set_title('Mel-Spectrogram')
        plt.colorbar(img, ax=ax3, format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Audio analysis visualization saved to {output_path}")