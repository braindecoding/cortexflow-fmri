#!/usr/bin/env python3
"""
🧠⚡ EEG Alignment and Preprocessing

Advanced preprocessing techniques for better EEG reconstruction.
"""

import numpy as np
import torch
import scipy.signal
from scipy import stats
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class EEGAligner:
    """Advanced EEG alignment and preprocessing."""
    
    def __init__(self, sampling_rate=500, target_length=500):
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.pca = None
        self.ica = None
        self.scaler = StandardScaler()
        
    def temporal_alignment(self, eeg_epochs, method='peak_detection'):
        """
        Align EEG epochs temporally for better consistency.
        
        Args:
            eeg_epochs: [n_epochs, n_channels, n_timepoints]
            method: 'peak_detection', 'cross_correlation', 'phase_locking'
        """
        print(f"🔄 Temporal alignment using {method}...")
        
        aligned_epochs = []
        
        if method == 'peak_detection':
            # Find peak activity in each epoch and align
            for epoch in eeg_epochs:
                # Calculate power across channels
                power = np.mean(epoch ** 2, axis=0)
                
                # Find peak activity
                peak_idx = np.argmax(power)
                
                # Align to center (target_length // 2)
                center = self.target_length // 2
                shift = center - peak_idx
                
                # Apply shift
                if shift > 0:
                    # Pad beginning
                    aligned = np.pad(epoch, ((0, 0), (shift, 0)), mode='constant')[:, :self.target_length]
                elif shift < 0:
                    # Trim beginning
                    aligned = epoch[:, -shift:(-shift + self.target_length)]
                else:
                    aligned = epoch[:, :self.target_length]
                
                aligned_epochs.append(aligned)
                
        elif method == 'cross_correlation':
            # Use first epoch as template
            template = eeg_epochs[0]
            template_power = np.mean(template ** 2, axis=0)
            
            for epoch in eeg_epochs:
                epoch_power = np.mean(epoch ** 2, axis=0)
                
                # Cross-correlation
                correlation = np.correlate(epoch_power, template_power, mode='full')
                shift = np.argmax(correlation) - len(template_power) + 1
                
                # Apply shift and truncate/pad
                if shift > 0:
                    aligned = np.pad(epoch, ((0, 0), (shift, 0)), mode='constant')[:, :self.target_length]
                elif shift < 0:
                    aligned = epoch[:, -shift:(-shift + self.target_length)]
                else:
                    aligned = epoch[:, :self.target_length]
                
                aligned_epochs.append(aligned)
        
        else:  # No alignment
            aligned_epochs = [epoch[:, :self.target_length] for epoch in eeg_epochs]
        
        return np.array(aligned_epochs)
    
    def artifact_removal(self, eeg_epochs, method='ica'):
        """
        Remove artifacts from EEG data.
        
        Args:
            eeg_epochs: [n_epochs, n_channels, n_timepoints]
            method: 'ica', 'pca', 'bandpass'
        """
        print(f"🧹 Artifact removal using {method}...")
        
        n_epochs, n_channels, n_timepoints = eeg_epochs.shape
        
        if method == 'ica':
            # Reshape for ICA: [n_samples, n_features]
            data_reshaped = eeg_epochs.reshape(-1, n_channels)
            
            # Apply ICA
            if self.ica is None:
                self.ica = FastICA(n_components=min(n_channels, 20), random_state=42)
                ica_components = self.ica.fit_transform(data_reshaped)
            else:
                ica_components = self.ica.transform(data_reshaped)
            
            # Remove artifact components (heuristic: remove components with high kurtosis)
            kurtosis_values = [stats.kurtosis(comp) for comp in ica_components.T]
            artifact_threshold = np.percentile(np.abs(kurtosis_values), 80)
            
            clean_components = ica_components.copy()
            for i, kurt in enumerate(kurtosis_values):
                if np.abs(kurt) > artifact_threshold:
                    clean_components[:, i] = 0
            
            # Reconstruct clean data
            clean_data = self.ica.inverse_transform(clean_components)
            clean_epochs = clean_data.reshape(n_epochs, n_channels, n_timepoints)
            
        elif method == 'pca':
            # Reshape for PCA
            data_reshaped = eeg_epochs.reshape(-1, n_channels)
            
            # Apply PCA
            if self.pca is None:
                self.pca = PCA(n_components=min(n_channels, 32))
                pca_components = self.pca.fit_transform(data_reshaped)
            else:
                pca_components = self.pca.transform(data_reshaped)
            
            # Reconstruct with reduced components
            clean_data = self.pca.inverse_transform(pca_components)
            clean_epochs = clean_data.reshape(n_epochs, n_channels, n_timepoints)
            
        elif method == 'bandpass':
            # Bandpass filter (1-40 Hz for cognitive tasks)
            nyquist = self.sampling_rate / 2
            low_freq = 1.0 / nyquist
            high_freq = 40.0 / nyquist
            
            b, a = scipy.signal.butter(4, [low_freq, high_freq], btype='band')
            
            clean_epochs = np.zeros_like(eeg_epochs)
            for i in range(n_epochs):
                for ch in range(n_channels):
                    clean_epochs[i, ch, :] = scipy.signal.filtfilt(b, a, eeg_epochs[i, ch, :])
        
        else:
            clean_epochs = eeg_epochs
        
        return clean_epochs
    
    def spatial_filtering(self, eeg_epochs, method='car'):
        """
        Apply spatial filtering to EEG data.
        
        Args:
            eeg_epochs: [n_epochs, n_channels, n_timepoints]
            method: 'car' (Common Average Reference), 'laplacian', 'none'
        """
        print(f"🌐 Spatial filtering using {method}...")
        
        if method == 'car':
            # Common Average Reference
            filtered_epochs = eeg_epochs - np.mean(eeg_epochs, axis=1, keepdims=True)
            
        elif method == 'laplacian':
            # Simple Laplacian (approximate)
            filtered_epochs = np.zeros_like(eeg_epochs)
            for i in range(eeg_epochs.shape[0]):
                for ch in range(eeg_epochs.shape[1]):
                    # Simple approximation: subtract average of neighboring channels
                    neighbors = []
                    for neighbor in range(max(0, ch-2), min(eeg_epochs.shape[1], ch+3)):
                        if neighbor != ch:
                            neighbors.append(eeg_epochs[i, neighbor, :])
                    
                    if neighbors:
                        neighbor_avg = np.mean(neighbors, axis=0)
                        filtered_epochs[i, ch, :] = eeg_epochs[i, ch, :] - neighbor_avg
                    else:
                        filtered_epochs[i, ch, :] = eeg_epochs[i, ch, :]
        
        else:  # No filtering
            filtered_epochs = eeg_epochs
        
        return filtered_epochs
    
    def feature_enhancement(self, eeg_epochs, method='spectral'):
        """
        Enhance EEG features for better reconstruction.
        
        Args:
            eeg_epochs: [n_epochs, n_channels, n_timepoints]
            method: 'spectral', 'wavelet', 'envelope'
        """
        print(f"✨ Feature enhancement using {method}...")
        
        if method == 'spectral':
            # Add spectral power features
            enhanced_epochs = []
            
            for epoch in eeg_epochs:
                # Original signal
                enhanced_epoch = epoch.copy()
                
                # Add frequency band powers
                freqs = np.fft.fftfreq(epoch.shape[1], 1/self.sampling_rate)
                fft_data = np.fft.fft(epoch, axis=1)
                power_spectrum = np.abs(fft_data) ** 2
                
                # Alpha band (8-12 Hz)
                alpha_mask = (freqs >= 8) & (freqs <= 12)
                alpha_power = np.mean(power_spectrum[:, alpha_mask], axis=1, keepdims=True)
                alpha_feature = np.tile(alpha_power, (1, epoch.shape[1]))
                
                # Beta band (13-30 Hz)
                beta_mask = (freqs >= 13) & (freqs <= 30)
                beta_power = np.mean(power_spectrum[:, beta_mask], axis=1, keepdims=True)
                beta_feature = np.tile(beta_power, (1, epoch.shape[1]))
                
                # Combine features (normalize to same scale)
                alpha_norm = (alpha_feature - alpha_feature.mean()) / (alpha_feature.std() + 1e-8)
                beta_norm = (beta_feature - beta_feature.mean()) / (beta_feature.std() + 1e-8)
                
                # Weight original signal more heavily
                enhanced_epoch = 0.7 * enhanced_epoch + 0.2 * alpha_norm + 0.1 * beta_norm
                enhanced_epochs.append(enhanced_epoch)
            
            return np.array(enhanced_epochs)
            
        elif method == 'envelope':
            # Add signal envelope
            enhanced_epochs = []
            
            for epoch in eeg_epochs:
                # Calculate envelope using Hilbert transform
                analytic_signal = scipy.signal.hilbert(epoch, axis=1)
                envelope = np.abs(analytic_signal)
                
                # Normalize envelope
                envelope_norm = (envelope - envelope.mean()) / (envelope.std() + 1e-8)
                
                # Combine with original
                enhanced_epoch = 0.8 * epoch + 0.2 * envelope_norm
                enhanced_epochs.append(enhanced_epoch)
            
            return np.array(enhanced_epochs)
        
        else:
            return eeg_epochs
    
    def normalize_advanced(self, eeg_epochs, method='robust'):
        """
        Advanced normalization for EEG data.
        
        Args:
            eeg_epochs: [n_epochs, n_channels, n_timepoints]
            method: 'robust', 'quantile', 'channel_wise'
        """
        print(f"📊 Advanced normalization using {method}...")
        
        if method == 'robust':
            # Robust normalization using median and MAD
            normalized_epochs = np.zeros_like(eeg_epochs)
            
            for ch in range(eeg_epochs.shape[1]):
                channel_data = eeg_epochs[:, ch, :].flatten()
                median = np.median(channel_data)
                mad = np.median(np.abs(channel_data - median))
                
                if mad > 0:
                    normalized_epochs[:, ch, :] = (eeg_epochs[:, ch, :] - median) / (1.4826 * mad)
                else:
                    normalized_epochs[:, ch, :] = eeg_epochs[:, ch, :] - median
            
        elif method == 'quantile':
            # Quantile normalization
            normalized_epochs = np.zeros_like(eeg_epochs)
            
            for ch in range(eeg_epochs.shape[1]):
                channel_data = eeg_epochs[:, ch, :].flatten()
                q25, q75 = np.percentile(channel_data, [25, 75])
                iqr = q75 - q25
                
                if iqr > 0:
                    normalized_epochs[:, ch, :] = (eeg_epochs[:, ch, :] - q25) / iqr
                else:
                    normalized_epochs[:, ch, :] = eeg_epochs[:, ch, :] - q25
        
        elif method == 'channel_wise':
            # Channel-wise z-score normalization
            normalized_epochs = np.zeros_like(eeg_epochs)
            
            for ch in range(eeg_epochs.shape[1]):
                channel_data = eeg_epochs[:, ch, :]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                
                if std > 0:
                    normalized_epochs[:, ch, :] = (channel_data - mean) / std
                else:
                    normalized_epochs[:, ch, :] = channel_data - mean
        
        else:
            normalized_epochs = eeg_epochs
        
        return normalized_epochs
    
    def process_pipeline(self, eeg_epochs, config=None):
        """
        Complete preprocessing pipeline.
        
        Args:
            eeg_epochs: [n_epochs, n_channels, n_timepoints]
            config: dict with processing options
        """
        if config is None:
            config = {
                'temporal_alignment': 'peak_detection',
                'artifact_removal': 'bandpass',
                'spatial_filtering': 'car',
                'feature_enhancement': 'spectral',
                'normalization': 'robust'
            }
        
        print("🔄 Starting EEG preprocessing pipeline...")
        
        # Step 1: Temporal alignment
        if config.get('temporal_alignment', 'none') != 'none':
            eeg_epochs = self.temporal_alignment(eeg_epochs, config['temporal_alignment'])
        
        # Step 2: Artifact removal
        if config.get('artifact_removal', 'none') != 'none':
            eeg_epochs = self.artifact_removal(eeg_epochs, config['artifact_removal'])
        
        # Step 3: Spatial filtering
        if config.get('spatial_filtering', 'none') != 'none':
            eeg_epochs = self.spatial_filtering(eeg_epochs, config['spatial_filtering'])
        
        # Step 4: Feature enhancement
        if config.get('feature_enhancement', 'none') != 'none':
            eeg_epochs = self.feature_enhancement(eeg_epochs, config['feature_enhancement'])
        
        # Step 5: Normalization
        if config.get('normalization', 'none') != 'none':
            eeg_epochs = self.normalize_advanced(eeg_epochs, config['normalization'])
        
        print("✅ EEG preprocessing pipeline completed!")
        return eeg_epochs
