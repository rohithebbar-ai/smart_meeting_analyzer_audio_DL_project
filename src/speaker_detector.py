"""
Speaker identification and segmentation using Resemblyzer
"""

import numpy as np
import librosa
import soundfile as sf 
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, List, Any, Optional
import tempfile
import os
from pathlib import Path

from config import SPEAKER_CONFIG, AUDIO_CONFIG
from src.utils import setup_logging, save_audio_segment

class SpeakerDetector:
    """Handles speaker identification and segmentation using Resemblyzer"""
    
    def __init__(self):
        self.logger = setup_logging("SpeakerDetector")
        self.encoder = VoiceEncoder()
        self.sample_rate = AUDIO_CONFIG.get('sample_rate', 16000)
        
        # Initialize speaker detection parameters with safe defaults
        self.min_segment_duration = SPEAKER_CONFIG.get('min_segment_duration', 1.0)
        self.similarity_threshold = SPEAKER_CONFIG.get('similarity_threshold', 0.75)
        self.max_speakers = SPEAKER_CONFIG.get('max_speakers', 10)
        
        self.logger.info("Resemblyzer VoiceEncoder initialized")
        return  # Skip the rest of original __init__
    
    def __init___original(self):
        self.logger = setup_logging("SpeakerDetector")
        self.encoder = VoiceEncoder()
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        self.min_segment_duration = SPEAKER_CONFIG['min_segment_duration']
        self.similarlity_threshold = SPEAKER_CONFIG['similarity_threshold']
        self.max_speakers = SPEAKER_CONFIG['max_speakers']
        
        self.logger.info("Resemblyzer VoiceEncoder Initialised")
        
    def identify_speakers(self, audio:np.ndarray, voice_segments: List[Tuple[float, float]]) -> List[Dict]:
        """
        Identify speakers in audio segments using Resemblyzer
        
        Args:
            audio: Full audio array
            voice_segments: List of (start_time, end_time) for detected speech
            
        Returns:
            List of segments with speaker identification
        
        """
        self.logger.info(f"Identifying speakers in {len(voice_segments)} segments.....")
        
        # Filter segments by minimum duration
        valid_segments = [
            seg for seg in voice_segments
            if seg[1] - seg[0] >= self.min_segment_duration
        ]
        
        self.logger.info(f"Processing {len(valid_segments)} segments (min duration: {self.min_segment_duration}s)")
        
        if not valid_segments:
            self.logger.warning("No segments meet minimum duration requirement")
            return []
        
        # Extract embeddings for each segment
        embeddings = []
        segment_data = []
        
        for i, (start_time, end_time) in enumerate(valid_segments):
            try:
                #Extract audio segment
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                segment_audio = audio[start_sample:end_sample]
                
                #Preprocess for resemblyzer
                processed_audio = preprocess_wav(segment_audio, self.sample_rate)
                
                # Generate embedding
                embedding = self.encoder.embed_utterance(processed_audio)
                embeddings.append(embedding)
                
                segment_data.append({
                    'segment_id': i,
                    "start_time": start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
            except Exception as e:
                self.logger.warning(f"Failed to process segment {i} ({start_time:.2f}s-{end_time:.2f}s): {str(e)}")
                continue
            
        if not embeddings:
            self.logger.error("No valid embeddings extracted")
            return []
        
        # Cluster embeddings to identify speakers
        speaker_labels = self._cluster_speakers(embeddings)
        
        # Assign speaker labels to segments
        speaker_segments = []
        for segment_info, speaker_id in zip(segment_data, speaker_labels):
            speaker_segments.append({
                'start_time': segment_info['start_time'],
                'end_time': segment_info['end_time'],
                'duration': segment_info['duration'],
                'speaker': f'Speaker_{speaker_id + 1}',
                'speaker_id': speaker_id,
                'segment_id': segment_info['segment_id']
            })
        
        # Sort by start time
        speaker_segments.sort(key=lambda x: x['start_time'])
        
        num_speakers = len(set(speaker_labels))
        self.logger.info(f"Identified {num_speakers} unique speakers")
        
        return speaker_segments
    
    def _cluster_speakers(self, embeddings: List[np.ndarray]) -> List[int]:
        """
        Cluster speaker embeddings to identify unique speakers
        
        Args:
            embeddings: List of voice embeddings
            
        Returns:
            List of speaker cluster labels
        """
        embeddings_array = np.array(embeddings)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # Determine optimal number of clusters
        n_clusters = self._estimate_num_speakers(similarity_matrix)
        
        self.logger.info(f"Clustering into {n_clusters} speakers using {SPEAKER_CONFIG['clustering_method']}")
        
        # Perform clustering
        if SPEAKER_CONFIG['clustering_method'] == 'agglomerative':
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)
        else:  # kmeans
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(embeddings_array)
        
        return labels.tolist()
    
    def _estimate_num_speakers(self, similarity_matrix: np.ndarray) -> int:
        """
        Estimate number of speakers based on similarity matrix
        
        Args:
            similarity_matrix: Cosine similarity matrix of embeddings
            
        Returns:
            Estimated number of speakers
        """
        # Method 1: Count segments with low similarity to others
        mean_similarities = np.mean(similarity_matrix, axis=1)
        unique_speakers = np.sum(mean_similarities < self.similarity_threshold)
        
        # Method 2: Use threshold-based approach
        # Find pairs with similarity below threshold
        n_segments = similarity_matrix.shape[0]
        low_similarity_pairs = 0
        
        for i in range(n_segments):
            for j in range(i + 1, n_segments):
                if similarity_matrix[i, j] < self.similarity_threshold:
                    low_similarity_pairs += 1
        
        # Estimate speakers based on connectivity
        estimated_speakers = max(1, min(
            unique_speakers,
            int(np.sqrt(2 * low_similarity_pairs)) + 1,
            self.max_speakers
        ))
        
        # Ensure reasonable bounds
        estimated_speakers = max(1, min(estimated_speakers, n_segments, self.max_speakers))
        
        return estimated_speakers
    
    def refine_speaker_segments(self, audio: np.ndarray, 
                               speaker_segments: List[Dict]) -> List[Dict]:
        """
        Refine speaker segments by merging adjacent segments from same speaker
        
        Args:
            audio: Full audio array
            speaker_segments: Initial speaker segments
            
        Returns:
            Refined speaker segments
        """
        if not speaker_segments:
            return speaker_segments
        
        self.logger.info("Refining speaker segments...")
        
        # Sort by start time
        segments = sorted(speaker_segments, key=lambda x: x['start_time'])
        
        refined_segments = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # If same speaker and segments are close, merge them
            time_gap = next_segment['start_time'] - current_segment['end_time']
            same_speaker = current_segment['speaker'] == next_segment['speaker']
            
            if same_speaker and time_gap < 2.0:  # Max 2 second gap for merging
                # Merge segments
                current_segment['end_time'] = next_segment['end_time']
                current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
            else:
                # Save current segment and start new one
                refined_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        # Don't forget the last segment
        refined_segments.append(current_segment)
        
        self.logger.info(f"Refined from {len(segments)} to {len(refined_segments)} segments")
        
        return refined_segments
    
    def analyze_speaker_characteristics(self, audio: np.ndarray, 
                                      speaker_segments: List[Dict]) -> Dict:
        """
        Analyze characteristics of each identified speaker
        
        Args:
            audio: Full audio array
            speaker_segments: Speaker-labeled segments
            
        Returns:
            Dictionary with speaker characteristics
        """
        self.logger.info("Analyzing speaker characteristics...")
        
        speaker_stats = {}
        
        # Group segments by speaker
        speakers = {}
        for segment in speaker_segments:
            speaker_id = segment['speaker']
            if speaker_id not in speakers:
                speakers[speaker_id] = []
            speakers[speaker_id].append(segment)
        
        # Analyze each speaker
        for speaker_id, segments in speakers.items():
            # Calculate total speaking time
            total_time = sum(seg['duration'] for seg in segments)
            
            # Calculate average segment length
            avg_segment_length = np.mean([seg['duration'] for seg in segments])
            
            # Extract all audio for this speaker
            speaker_audio_chunks = []
            for segment in segments:
                start_sample = int(segment['start_time'] * self.sample_rate)
                end_sample = int(segment['end_time'] * self.sample_rate)
                chunk = audio[start_sample:end_sample]
                speaker_audio_chunks.append(chunk)
            
            # Combine all audio from this speaker
            if speaker_audio_chunks:
                combined_audio = np.concatenate(speaker_audio_chunks)
                
                # Analyze voice characteristics
                voice_stats = self._analyze_voice_features(combined_audio)
            else:
                voice_stats = {}
            
            speaker_stats[speaker_id] = {
                'total_speaking_time': total_time,
                'number_of_segments': len(segments),
                'average_segment_length': avg_segment_length,
                'voice_characteristics': voice_stats
            }
        
        return speaker_stats
    
    def _analyze_voice_features(self, audio: np.ndarray) -> Dict:
        """
        Analyze voice characteristics for a speaker
        
        Args:
            audio: Combined audio from one speaker
            
        Returns:
            Dictionary with voice characteristics
        """
        try:
            # Calculate basic voice features
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                avg_pitch = np.mean(pitch_values)
                pitch_range = np.max(pitch_values) - np.min(pitch_values)
            else:
                avg_pitch = 0
                pitch_range = 0
            
            # Energy characteristics
            rms_energy = np.mean(librosa.feature.rms(y=audio))
            
            # Spectral characteristics
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
            
            # Speaking rate (rough estimate)
            zero_crossings = librosa.feature.zero_crossing_rate(audio)
            avg_zcr = np.mean(zero_crossings)
            
            return {
                'average_pitch': float(avg_pitch),
                'pitch_range': float(pitch_range),
                'energy_level': float(rms_energy),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'zero_crossing_rate': float(avg_zcr)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing voice features: {str(e)}")
            return {}
    
    def save_speaker_segments(self, audio: np.ndarray, speaker_segments: List[Dict], 
                             output_dir: str) -> Dict[str, List[str]]:
        """
        Save individual speaker segments as separate audio files
        
        Args:
            audio: Full audio array
            speaker_segments: Speaker-labeled segments
            output_dir: Directory to save segments
            
        Returns:
            Dictionary mapping speaker IDs to list of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        for segment in speaker_segments:
            speaker_id = segment['speaker']
            start_time = segment['start_time']
            end_time = segment['end_time']
            
            # Create filename
            filename = f"{speaker_id}_segment_{start_time:.2f}s-{end_time:.2f}s.wav"
            filepath = os.path.join(output_dir, filename)
            
            # Extract and save segment
            save_audio_segment(audio, self.sample_rate, start_time, end_time, filepath)
            
            # Track saved files
            if speaker_id not in saved_files:
                saved_files[speaker_id] = []
            saved_files[speaker_id].append(filepath)
        
        self.logger.info(f"Saved {len(speaker_segments)} speaker segments to {output_dir}")
        
        return saved_files