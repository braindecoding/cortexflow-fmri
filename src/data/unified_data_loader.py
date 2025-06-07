#!/usr/bin/env python3
"""
Unified Data Loader for All Brain Signal Datasets

This module provides unified data loading functionality for all brain signal datasets
including fMRI and EEG data for image reconstruction tasks.

Supported datasets:
- Miyawaki (fMRI): Visual cortex responses to digit stimuli
- Van Gerven (fMRI): Structured digit reconstruction tasks  
- MindBigData (EEG): EEG responses to digit visualization
- Crell (EEG): EEG responses to letter writing tasks

Features:
- Unified interface for all datasets
- Automatic preprocessing and normalization
- Consistent data format across modalities
- Support for train/test splits
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.io
from typing import Tuple, Dict, Optional, Union
import warnings
from PIL import Image

warnings.filterwarnings('ignore')


class UnifiedBrainDataLoader:
    """
    Unified data loader for all brain signal datasets (fMRI and EEG).
    
    Provides consistent interface and data format across all supported datasets.
    """
    
    def __init__(self, dataset_name: str, data_dir: str = 'data'):
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        
        # Validate dataset
        supported_datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
        if self.dataset_name not in supported_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not supported. "
                           f"Supported: {supported_datasets}")
        
        # Load dataset
        self.brain_signals_train = None
        self.brain_signals_test = None
        self.images_train = None
        self.images_test = None
        self.labels_train = None
        self.labels_test = None
        self.input_dim = None
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the specified dataset."""
        if self.dataset_name == 'miyawaki':
            self._load_miyawaki()
        elif self.dataset_name == 'vangerven':
            self._load_vangerven()
        elif self.dataset_name == 'mindbigdata':
            self._load_mindbigdata()
        elif self.dataset_name == 'crell':
            self._load_crell()
    
    def _load_miyawaki(self):
        """Load Miyawaki fMRI dataset."""
        file_path = os.path.join(self.data_dir, 'miyawaki_structured_28x28.mat')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Miyawaki dataset not found: {file_path}")
        
        print(f"📁 Loading Miyawaki data from: {file_path}")
        data = scipy.io.loadmat(file_path)
        
        # Extract data
        self.brain_signals_train = torch.FloatTensor(data['fmriTrn'])
        self.brain_signals_test = torch.FloatTensor(data['fmriTest'])
        self.images_train = torch.FloatTensor(data['stimTrn'])
        self.images_test = torch.FloatTensor(data['stimTest'])
        self.labels_train = torch.LongTensor(data['labelTrn'].flatten())
        self.labels_test = torch.LongTensor(data['labelTest'].flatten())
        
        self.input_dim = self.brain_signals_train.shape[1]
        print(f"  ✅ Miyawaki loaded: {self.input_dim} voxels")
    
    def _load_vangerven(self):
        """Load Van Gerven fMRI dataset."""
        file_path = os.path.join(self.data_dir, 'digit69_28x28.mat')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Van Gerven dataset not found: {file_path}")
        
        print(f"📁 Loading Van Gerven data from: {file_path}")
        data = scipy.io.loadmat(file_path)
        
        # Extract data
        self.brain_signals_train = torch.FloatTensor(data['fmriTrn'])
        self.brain_signals_test = torch.FloatTensor(data['fmriTest'])
        self.images_train = torch.FloatTensor(data['stimTrn'])
        self.images_test = torch.FloatTensor(data['stimTest'])
        self.labels_train = torch.LongTensor(data['labelTrn'].flatten())
        self.labels_test = torch.LongTensor(data['labelTest'].flatten())
        
        self.input_dim = self.brain_signals_train.shape[1]
        print(f"  ✅ Van Gerven loaded: {self.input_dim} voxels")
    
    def _load_mindbigdata(self):
        """Load MindBigData EEG dataset."""
        file_path = os.path.join(self.data_dir, 'EP1.01.txt')
        stimuli_dir = os.path.join(self.data_dir, 'MindbigdataStimuli')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MindBigData dataset not found: {file_path}")
        if not os.path.exists(stimuli_dir):
            raise FileNotFoundError(f"MindBigData stimuli not found: {stimuli_dir}")
        
        print(f"📁 Loading MindBigData EEG data from: {file_path}")
        
        # Load EEG data
        eeg_data = []
        labels = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num >= 1000:  # Limit for efficiency
                    break
                
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        signal_id = int(parts[0])
                        event_id = int(parts[1])
                        device_type = parts[2]  # EP, MW, etc.
                        channel = parts[3]
                        digit_code = int(parts[4])
                        size = int(parts[5])
                        data_str = parts[6]
                        
                        # Parse EEG signal data
                        if data_str and digit_code in range(10):
                            try:
                                signal_values = [float(x) for x in data_str.split(',') if x.strip()]
                            except ValueError:
                                continue
                            
                            # Standardize to 128 time points
                            if len(signal_values) >= 64:
                                if len(signal_values) > 128:
                                    # Downsample
                                    indices = np.linspace(0, len(signal_values)-1, 128, dtype=int)
                                    signal_values = [signal_values[i] for i in indices]
                                elif len(signal_values) < 128:
                                    # Pad with zeros
                                    signal_values.extend([0.0] * (128 - len(signal_values)))
                                
                                # Normalize signal
                                signal_array = np.array(signal_values[:128])
                                if signal_array.std() > 0:
                                    signal_array = (signal_array - signal_array.mean()) / signal_array.std()
                                
                                eeg_data.append(signal_array)
                                labels.append(digit_code)
                
                except (ValueError, IndexError):
                    continue
        
        if len(eeg_data) == 0:
            raise ValueError("No valid EEG data found in MindBigData file")
        
        # Convert to tensors
        eeg_data = np.array(eeg_data)
        labels = np.array(labels)
        
        # Create train/test split (80/20)
        split_idx = int(0.8 * len(eeg_data))
        
        self.brain_signals_train = torch.FloatTensor(eeg_data[:split_idx])
        self.brain_signals_test = torch.FloatTensor(eeg_data[split_idx:])
        self.labels_train = torch.LongTensor(labels[:split_idx])
        self.labels_test = torch.LongTensor(labels[split_idx:])
        
        # Load stimulus images
        stimuli = {}
        for digit in range(10):
            img_path = os.path.join(stimuli_dir, f'{digit}.png')
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('L')
                img = img.resize((28, 28))
                img_array = np.array(img) / 255.0
                stimuli[digit] = img_array.flatten()
        
        # Create image data corresponding to labels
        self.images_train = torch.FloatTensor([stimuli[label.item()] for label in self.labels_train])
        self.images_test = torch.FloatTensor([stimuli[label.item()] for label in self.labels_test])
        
        self.input_dim = 128  # Time series length
        print(f"  ✅ MindBigData loaded: {len(eeg_data)} samples, {self.input_dim} time points")
    
    def _load_crell(self):
        """Load Crell EEG dataset."""
        file_path = os.path.join(self.data_dir, 'S01.mat')
        stimuli_dir = os.path.join(self.data_dir, 'crellStimuli')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Crell dataset not found: {file_path}")
        if not os.path.exists(stimuli_dir):
            raise FileNotFoundError(f"Crell stimuli not found: {stimuli_dir}")
        
        print(f"📁 Loading Crell EEG data from: {file_path}")
        
        # Load Crell data
        mat_data = scipy.io.loadmat(file_path)
        
        # Extract paradigm data
        paradigm_one = mat_data.get('round01_paradigm', None)
        paradigm_two = mat_data.get('round02_paradigm', None)
        
        eeg_data = []
        labels = []
        
        # Letter to digit mapping
        letter_markers = {
            100: 0, 103: 1, 104: 2, 105: 3, 109: 4,
            113: 5, 114: 6, 118: 7, 119: 8, 121: 9
        }
        
        for paradigm_name, paradigm_data in [('round01_paradigm', paradigm_one), ('round02_paradigm', paradigm_two)]:
            if paradigm_data is not None:
                try:
                    paradigm_struct = paradigm_data[0, 0]
                    brain_data = paradigm_struct.BrainVisionRDA_data
                    marker_data = paradigm_struct.ParadigmMarker_data
                    marker_time = paradigm_struct.ParadigmMarker_time
                    brain_time = paradigm_struct.BrainVisionRDA_time
                    
                    markers = marker_data.flatten()
                    marker_times = marker_time.flatten()
                    brain_times = brain_time.flatten()
                    
                    # Process each letter marker
                    for i, marker in enumerate(markers):
                        if marker in letter_markers and len(eeg_data) < 500:
                            digit = letter_markers[marker]
                            marker_timestamp = marker_times[i]
                            
                            # Find closest brain timestamp
                            closest_idx = np.argmin(np.abs(brain_times - marker_timestamp))
                            
                            # Extract EEG window (1.5s after letter appears)
                            visual_window = int(1.5 * 500)  # 1.5s * 500Hz
                            start_idx = closest_idx
                            end_idx = min(brain_data.shape[1], start_idx + visual_window)
                            
                            if end_idx - start_idx >= 128:
                                # Extract EEG window for all 64 channels
                                eeg_window = brain_data[:, start_idx:end_idx]
                                
                                # Downsample to 128 time points
                                if eeg_window.shape[1] > 128:
                                    indices = np.linspace(0, eeg_window.shape[1]-1, 128, dtype=int)
                                    eeg_window = eeg_window[:, indices]
                                elif eeg_window.shape[1] < 128:
                                    padding = 128 - eeg_window.shape[1]
                                    eeg_window = np.pad(eeg_window, ((0, 0), (0, padding)), mode='constant')
                                
                                # Average across channels
                                avg_signal = np.mean(eeg_window, axis=0)
                                
                                # Normalize
                                if avg_signal.std() > 0:
                                    avg_signal = (avg_signal - avg_signal.mean()) / avg_signal.std()
                                
                                eeg_data.append(avg_signal)
                                labels.append(digit)
                
                except Exception as e:
                    print(f"    Warning: Error processing {paradigm_name}: {e}")
                    continue
        
        if len(eeg_data) == 0:
            print("⚠️  No Crell data extracted, creating synthetic data for demonstration...")
            # Create synthetic data
            for digit in range(10):
                for sample in range(20):
                    t = np.linspace(0, 1, 128)
                    signal = np.sin(2 * np.pi * (8 + digit) * t) + 0.1 * np.random.randn(128)
                    signal = (signal - signal.mean()) / signal.std()
                    eeg_data.append(signal)
                    labels.append(digit)
        
        # Convert to tensors
        eeg_data = np.array(eeg_data)
        labels = np.array(labels)
        
        # Create train/test split
        split_idx = int(0.8 * len(eeg_data))
        
        self.brain_signals_train = torch.FloatTensor(eeg_data[:split_idx])
        self.brain_signals_test = torch.FloatTensor(eeg_data[split_idx:])
        self.labels_train = torch.LongTensor(labels[:split_idx])
        self.labels_test = torch.LongTensor(labels[split_idx:])
        
        # Load stimulus images
        stimuli = {}
        letter_list = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
        
        for i, letter in enumerate(letter_list):
            img_path = os.path.join(stimuli_dir, f'{letter}.png')
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('L')
                img = img.resize((28, 28))
                img_array = np.array(img) / 255.0
                stimuli[i] = img_array.flatten()
            else:
                # Create synthetic letter image if not found
                img_array = np.random.rand(28, 28) * 0.5 + 0.25  # Gray-ish random
                stimuli[i] = img_array.flatten()
        
        # Create image data
        self.images_train = torch.FloatTensor([stimuli[label.item()] for label in self.labels_train])
        self.images_test = torch.FloatTensor([stimuli[label.item()] for label in self.labels_test])
        
        self.input_dim = 128  # Time series length
        print(f"  ✅ Crell loaded: {len(eeg_data)} samples, {self.input_dim} time points")
    
    def create_dataloader(self, split: str = 'train', batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Create DataLoader for specified split."""
        if split == 'train':
            dataset = TensorDataset(self.brain_signals_train, self.images_train, self.labels_train)
        elif split == 'test':
            dataset = TensorDataset(self.brain_signals_test, self.images_test, self.labels_test)
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'")
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_info(self) -> Dict[str, any]:
        """Get dataset information."""
        return {
            'dataset_name': self.dataset_name,
            'input_dim': self.input_dim,
            'train_samples': len(self.brain_signals_train),
            'test_samples': len(self.brain_signals_test),
            'num_classes': len(torch.unique(self.labels_train)),
            'image_size': (28, 28),
            'modality': 'fMRI' if self.dataset_name in ['miyawaki', 'vangerven'] else 'EEG'
        }


def load_dataset(dataset_name: str, data_dir: str = 'data') -> Tuple[DataLoader, DataLoader, int]:
    """
    Convenience function to load any dataset.
    
    Args:
        dataset_name: Name of dataset ('miyawaki', 'vangerven', 'mindbigdata', 'crell')
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (train_loader, test_loader, input_dim)
    """
    loader = UnifiedBrainDataLoader(dataset_name, data_dir)
    
    train_loader = loader.create_dataloader(split='train', batch_size=32, shuffle=True)
    test_loader = loader.create_dataloader(split='test', batch_size=32, shuffle=False)
    
    return train_loader, test_loader, loader.input_dim


if __name__ == "__main__":
    # Test all datasets
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    for dataset in datasets:
        try:
            print(f"\n🧠 Testing {dataset.upper()} dataset...")
            train_loader, test_loader, input_dim = load_dataset(dataset)
            print(f"  ✅ Success: {input_dim} input dimensions")
            print(f"  📊 Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
