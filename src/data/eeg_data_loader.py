"""
🧠⚡ EEG Data Loader for Brain-to-Image Reconstruction

Supports:
1. MindBigData Dataset (EEG → Digit Images)
2. Crell Dataset (EEG → Letter Images)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from PIL import Image


class EEGDataLoader:
    """EEG data loader for brain-to-image reconstruction."""
    
    def __init__(self,
                 dataset_type: str = 'mindbigdata',  # 'mindbigdata' or 'crell'
                 data_path: str = None,
                 stimuli_path: str = None,
                 device: str = 'cuda',
                 normalize_eeg: bool = True,
                 normalize_stimuli: bool = True):
        """
        Initialize EEG data loader.
        
        Args:
            dataset_type: 'mindbigdata' or 'crell'
            data_path: Path to EEG data file
            stimuli_path: Path to stimuli folder
            device: Device for tensors ('cpu' or 'cuda')
            normalize_eeg: Whether to normalize EEG data
            normalize_stimuli: Whether to normalize stimulus images
        """
        self.dataset_type = dataset_type.lower()
        self.device = device
        self.normalize_eeg = normalize_eeg
        self.normalize_stimuli = normalize_stimuli
        
        # Set default paths if not provided
        if data_path is None:
            if self.dataset_type == 'mindbigdata':
                data_path = "data/EP1.01.txt"
            elif self.dataset_type == 'crell':
                data_path = "data/S01.mat"
        
        if stimuli_path is None:
            if self.dataset_type == 'mindbigdata':
                stimuli_path = "data/MindbigdataStimuli"
            elif self.dataset_type == 'crell':
                stimuli_path = "data/crellStimuli"
        
        self.data_path = Path(data_path)
        self.stimuli_path = Path(stimuli_path)
        
        # Validate files exist
        if not self.data_path.exists():
            raise FileNotFoundError(f"EEG data file not found: {data_path}")
        if not self.stimuli_path.exists():
            raise FileNotFoundError(f"Stimuli folder not found: {stimuli_path}")
        
        # Load and process data
        print(f"🧠⚡ Loading {self.dataset_type.upper()} EEG dataset...")
        print(f"📁 EEG data: {self.data_path.name}")
        print(f"🖼️ Stimuli: {self.stimuli_path.name}/")
        
        self.stimuli_images = self._load_stimuli()
        self.raw_eeg_data = self._load_eeg_data()
        self.processed_data = self._process_data()
        
        # Display info
        self._print_info()
    
    def _load_stimuli(self) -> Dict[str, np.ndarray]:
        """Load stimulus images."""
        stimuli = {}
        
        if self.dataset_type == 'mindbigdata':
            # Load digit images (0.jpg - 9.jpg)
            for digit in range(10):
                img_path = self.stimuli_path / f"{digit}.jpg"
                if img_path.exists():
                    img = self._load_and_preprocess_image(img_path)
                    stimuli[str(digit)] = img
                    
        elif self.dataset_type == 'crell':
            # Load letter images (a.png, d.png, etc.)
            letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
            for letter in letters:
                img_path = self.stimuli_path / f"{letter}.png"
                if img_path.exists():
                    img = self._load_and_preprocess_image(img_path)
                    stimuli[letter] = img
        
        print(f"✅ Loaded {len(stimuli)} stimulus images")
        return stimuli
    
    def _load_and_preprocess_image(self, img_path: Path) -> np.ndarray:
        """Load and preprocess image to 28x28 grayscale."""
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        return img_array
    
    def _load_eeg_data(self) -> Dict:
        """Load EEG data based on dataset type."""
        if self.dataset_type == 'mindbigdata':
            return self._load_mindbigdata_eeg()
        elif self.dataset_type == 'crell':
            return self._load_crell_eeg()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_mindbigdata_eeg(self) -> Dict:
        """Load MindBigData EEG from text file."""
        print("📊 Loading MindBigData EEG data...")
        
        signals_by_event = defaultdict(lambda: defaultdict(list))
        
        # For production, limit to first 100000 lines for reasonable training time
        max_lines = 100000

        with open(self.data_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num >= max_lines:
                    print(f"  Limiting to first {max_lines} lines for testing...")
                    break

                if line_num % 10000 == 0:
                    print(f"  Processing line {line_num}...")

                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    try:
                        event_id = int(parts[1])
                        channel = parts[3]
                        digit_code = int(parts[4])
                        data_str = parts[6]

                        # Skip random signals (-1)
                        if digit_code == -1:
                            continue

                        # Parse data values
                        data_values = [float(x) for x in data_str.split(',') if x.strip()]
                        if len(data_values) > 0:
                            signals_by_event[event_id][channel].append({
                                'code': digit_code,
                                'data': np.array(data_values)
                            })
                    except (ValueError, IndexError):
                        continue
        
        print(f"✅ Loaded {len(signals_by_event)} events")
        return signals_by_event
    
    def _load_crell_eeg(self) -> Dict:
        """Load Crell EEG from .mat file with real data extraction."""
        print("📊 Loading Crell EEG data...")

        data = scipy.io.loadmat(self.data_path)

        # Check available keys in the .mat file
        available_keys = [key for key in data.keys() if not key.startswith('__')]
        print(f"  Available keys: {available_keys}")

        # Extract paradigm data with visual-motor separation
        signals_by_trial = defaultdict(lambda: defaultdict(list))

        # Letter mapping according to the description
        letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
        letter_codes = [100, 103, 104, 105, 109, 113, 114, 118, 119, 121]  # 100 + ascii index
        code_to_letter = {code: letter for code, letter in zip(letter_codes, letters)}

        print(f"  Letter codes mapping: {code_to_letter}")

        # Process round01_paradigm (focus on visual phase only)
        if 'round01_paradigm' in data:
            paradigm_data = data['round01_paradigm']
            print(f"  Processing round01_paradigm...")

            # Extract EEG data, markers, and timestamps
            eeg_data = paradigm_data['BrainVisionRDA_data'][0, 0]  # [64 channels, time_points]
            eeg_time = paradigm_data['BrainVisionRDA_time'][0, 0]  # [1, time_points]
            markers = paradigm_data['ParadigmMarker_data'][0, 0]   # [1, n_markers]
            marker_time = paradigm_data['ParadigmMarker_time'][0, 0]  # [1, n_markers]

            print(f"    EEG data shape: {eeg_data.shape}")
            print(f"    EEG time shape: {eeg_time.shape}")
            print(f"    Markers shape: {markers.shape}")
            print(f"    Marker time shape: {marker_time.shape}")

            # Flatten marker arrays if needed
            if markers.ndim > 1:
                markers = markers.flatten()
            if marker_time.ndim > 1:
                marker_time = marker_time.flatten()
            if eeg_time.ndim > 1:
                eeg_time = eeg_time.flatten()

            # Find letter presentation events
            letter_events = []
            for i, marker in enumerate(markers):
                if marker in code_to_letter:
                    letter_events.append({
                        'letter': code_to_letter[marker],
                        'marker_time': marker_time[i],
                        'marker_code': marker
                    })

            print(f"    Found {len(letter_events)} letter events")

            # Extract EEG epochs for visual phase (before writing)
            # Focus on the period from letter fade-in to fade-out (visual processing)
            sampling_rate = 500  # Hz
            epoch_duration = 2.5  # seconds (2s fade-in + 0.5s full opacity)
            epoch_samples = int(epoch_duration * sampling_rate)  # 1250 samples

            print(f"    Extracting epochs with {epoch_samples} samples each...")

            # Note: EEG data is [time_points, channels], need to transpose
            eeg_data = eeg_data.T  # Now [channels, time_points]
            print(f"    EEG data transposed to: {eeg_data.shape}")

            trial_idx = 0
            for event in letter_events[:100]:  # Limit for testing
                # Find the closest EEG time point to marker time
                time_diff = np.abs(eeg_time - event['marker_time'])
                marker_sample = np.argmin(time_diff)

                # Extract epoch from marker to marker + epoch_duration
                start_sample = marker_sample
                end_sample = start_sample + epoch_samples

                if end_sample <= eeg_data.shape[1]:
                    # Extract EEG epoch [64 channels, epoch_samples]
                    eeg_epoch = eeg_data[:, start_sample:end_sample]

                    # Downsample to 500 time points for consistency
                    if eeg_epoch.shape[1] > 500:
                        # Downsample by taking every nth sample
                        step = eeg_epoch.shape[1] // 500
                        eeg_epoch = eeg_epoch[:, ::step][:, :500]
                    elif eeg_epoch.shape[1] < 500:
                        # Pad with zeros
                        padding = np.zeros((64, 500 - eeg_epoch.shape[1]))
                        eeg_epoch = np.concatenate([eeg_epoch, padding], axis=1)

                    signals_by_trial[trial_idx]['letter'] = event['letter']
                    signals_by_trial[trial_idx]['eeg'] = eeg_epoch
                    signals_by_trial[trial_idx]['marker_time'] = event['marker_time']
                    trial_idx += 1

                    if trial_idx % 20 == 0:
                        print(f"      Processed {trial_idx} epochs...")

            print(f"    Successfully extracted {trial_idx} epochs")

        else:
            print("  Warning: round01_paradigm not found!")
            return {}

        print(f"✅ Loaded {len(signals_by_trial)} Crell trials with real data")
        return signals_by_trial
    
    def _process_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Process and organize EEG data."""
        if self.dataset_type == 'mindbigdata':
            return self._process_mindbigdata()
        elif self.dataset_type == 'crell':
            return self._process_crell()
    
    def _process_mindbigdata(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Process MindBigData into train/test splits."""
        print("🔄 Processing MindBigData...")
        
        # Organize data by digit
        eeg_epochs = []
        stimulus_images = []
        labels = []
        
        channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
        for event_id, event_data in self.raw_eeg_data.items():
            # Get digit code from first channel
            first_channel = list(event_data.keys())[0]
            if len(event_data[first_channel]) > 0:
                digit_code = event_data[first_channel][0]['code']
                
                # Create EEG epoch (14 channels)
                epoch = np.zeros((14, 256))  # 14 channels, 256 time points (2 seconds at 128Hz)
                
                for ch_idx, channel in enumerate(channels):
                    if channel in event_data and len(event_data[channel]) > 0:
                        data = event_data[channel][0]['data']
                        # Pad or truncate to 256 points
                        if len(data) >= 256:
                            epoch[ch_idx, :] = data[:256]
                        else:
                            epoch[ch_idx, :len(data)] = data
                
                eeg_epochs.append(epoch)
                
                # Get corresponding stimulus image
                if str(digit_code) in self.stimuli_images:
                    stimulus_images.append(self.stimuli_images[str(digit_code)].flatten())
                    labels.append(digit_code)
        
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(np.array(eeg_epochs))
        stimuli_tensor = torch.FloatTensor(np.array(stimulus_images))
        labels_tensor = torch.LongTensor(labels)
        
        # Normalize if requested
        if self.normalize_eeg:
            eeg_tensor = self._normalize_eeg(eeg_tensor)
        
        # Split into train/test (80/20)
        n_samples = len(eeg_tensor)
        n_train = int(0.8 * n_samples)
        
        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        # Keep data on CPU initially to avoid pin_memory conflicts
        processed = {
            'train': {
                'eeg': eeg_tensor[train_idx],
                'stimuli': stimuli_tensor[train_idx],
                'labels': labels_tensor[train_idx]
            },
            'test': {
                'eeg': eeg_tensor[test_idx],
                'stimuli': stimuli_tensor[test_idx],
                'labels': labels_tensor[test_idx]
            }
        }
        
        return processed
    
    def _process_crell(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Process Crell data into train/test splits."""
        print("🔄 Processing Crell data...")

        # Organize data by letter
        eeg_epochs = []
        stimulus_images = []
        labels = []

        letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
        letter_to_idx = {letter: idx for idx, letter in enumerate(letters)}

        for trial_idx, trial_data in self.raw_eeg_data.items():
            letter = trial_data['letter']
            eeg_data = trial_data['eeg']  # [64, 500]

            # Ensure EEG data has correct shape
            if eeg_data.shape[0] == 64 and eeg_data.shape[1] >= 500:
                # Truncate or pad to exactly 500 time points
                if eeg_data.shape[1] > 500:
                    eeg_data = eeg_data[:, :500]
                elif eeg_data.shape[1] < 500:
                    padding = np.zeros((64, 500 - eeg_data.shape[1]))
                    eeg_data = np.concatenate([eeg_data, padding], axis=1)

                eeg_epochs.append(eeg_data)

                # Get corresponding stimulus image
                if letter in self.stimuli_images:
                    stimulus_images.append(self.stimuli_images[letter].flatten())
                    labels.append(letter_to_idx[letter])

        print(f"✅ Processed {len(eeg_epochs)} Crell epochs")

        if len(eeg_epochs) == 0:
            print("❌ No valid Crell epochs found!")
            return {'train': {}, 'test': {}}

        # Convert to tensors
        eeg_tensor = torch.FloatTensor(np.array(eeg_epochs))
        stimuli_tensor = torch.FloatTensor(np.array(stimulus_images))
        labels_tensor = torch.LongTensor(labels)

        # Apply advanced preprocessing if enabled
        if hasattr(self, 'use_alignment') and self.use_alignment:
            from data.eeg_alignment import EEGAligner
            aligner = EEGAligner(sampling_rate=128 if self.dataset_type == 'mindbigdata' else 500)

            # Convert to numpy for processing
            eeg_numpy = eeg_tensor.numpy()

            # Apply preprocessing pipeline
            config = {
                'temporal_alignment': 'peak_detection',
                'artifact_removal': 'bandpass',
                'spatial_filtering': 'car',
                'feature_enhancement': 'spectral',
                'normalization': 'robust'
            }

            eeg_numpy = aligner.process_pipeline(eeg_numpy, config)
            eeg_tensor = torch.FloatTensor(eeg_numpy)

        # Standard normalization
        elif self.normalize_eeg:
            eeg_tensor = self._normalize_eeg(eeg_tensor)

        # Split into train/test (80/20)
        n_samples = len(eeg_tensor)
        n_train = int(0.8 * n_samples)

        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        processed = {
            'train': {
                'eeg': eeg_tensor[train_idx],
                'stimuli': stimuli_tensor[train_idx],
                'labels': labels_tensor[train_idx]
            },
            'test': {
                'eeg': eeg_tensor[test_idx],
                'stimuli': stimuli_tensor[test_idx],
                'labels': labels_tensor[test_idx]
            }
        }

        return processed
    
    def _normalize_eeg(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize EEG data (z-score per channel)."""
        # Normalize each channel independently
        for ch in range(eeg_tensor.shape[1]):
            channel_data = eeg_tensor[:, ch, :]
            mean = channel_data.mean()
            std = channel_data.std()
            if std > 0:
                eeg_tensor[:, ch, :] = (channel_data - mean) / std
        return eeg_tensor
    
    def _print_info(self):
        """Print dataset information."""
        print(f"\n📊 {self.dataset_type.upper()} Dataset Information:")
        
        for split in ['train', 'test']:
            if split in self.processed_data and self.processed_data[split]:
                data = self.processed_data[split]
                print(f"\n🎯 {split.capitalize()} Data:")
                
                for key, tensor in data.items():
                    print(f"  {key}: {tensor.shape}")
                    if key == 'eeg':
                        print(f"    Channels: {tensor.shape[1]}, Time points: {tensor.shape[2]}")
                        print(f"    Mean: {tensor.mean():.3f}, Std: {tensor.std():.3f}")
                    elif key == 'stimuli':
                        print(f"    Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Public API methods
    def get_train_data(self) -> Dict[str, torch.Tensor]:
        """Get training data."""
        return self.processed_data['train'].copy()
    
    def get_test_data(self) -> Dict[str, torch.Tensor]:
        """Get test data."""
        return self.processed_data['test'].copy()
    
    def get_eeg(self, split: str = 'train') -> torch.Tensor:
        """Get EEG data."""
        return self.processed_data[split]['eeg']
    
    def get_stimuli(self, split: str = 'train') -> torch.Tensor:
        """Get stimulus data."""
        return self.processed_data[split]['stimuli']
    
    def get_labels(self, split: str = 'train') -> torch.Tensor:
        """Get labels."""
        return self.processed_data[split]['labels']


# Convenience function
def load_eeg_data(dataset_type: str = 'mindbigdata',
                  batch_size: int = 32,
                  device: str = 'cuda',
                  **kwargs):
    """
    Load EEG data and return train/test loaders.
    
    Args:
        dataset_type: 'mindbigdata' or 'crell'
        batch_size: Batch size for DataLoader
        device: Device for tensors
        **kwargs: Additional arguments for EEGDataLoader
    
    Returns:
        tuple: (train_loader, test_loader, input_dim)
    """
    # Load data
    loader = EEGDataLoader(dataset_type=dataset_type, device=device, **kwargs)
    
    # Get input dimension (EEG dimension)
    sample_eeg = loader.get_eeg('train')
    input_dim = sample_eeg.shape[1] * sample_eeg.shape[2]  # channels * time_points
    
    return loader, input_dim


# Demo
if __name__ == "__main__":
    print("🧠⚡ Demo: EEG Data Loader")
    print("=" * 40)
    
    # Test MindBigData
    try:
        loader = EEGDataLoader(dataset_type='mindbigdata')
        print("✅ MindBigData loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading MindBigData: {e}")
