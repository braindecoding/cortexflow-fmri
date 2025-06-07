# Panduan Dataset EEG
## ✅ **CORRECTED UNDERSTANDING OF DATASETS**

### 🎯 **MindBigData Dataset (EEG)**
- **File**: `EP1.01.txt`
- **Focus**: **Pure Visual Phase** (Digit Image Reconstruction)
- **Task**: EEG → **Digit Image Reconstruction**
- **Stimulus Type**: **28×28 Digit Images** (matching Van Gerven format) in `MindbigdataStimuli/` folder
- **Content**: Digit images
- **Data**: EPOC 14 channels, digits 0-9

### 🎯 **Crell Dataset (EEG)**
- **File**: `S01.mat`
- **Focus**: **Visual Phase Only** (Letter Image Reconstruction)
- **Task**: EEG → **Letter Image Reconstruction**
- **Stimulus Type**: **28×28 Letter Images**  in `crellStimuli/` folder
- **Content**: Letter images (a,d,e,f,j,n,o,s,t,v)
- **Data**: 64 channels, 500Hz, visual paradigm
- **Focus**: Pure visual processing (motor tasks need to be removed)

## 🔍 **KEY DIFFERENCES CLARIFIED**

1. **MindBigData**: EEG → **Digit Image Reconstruction** (28×28 images)
2. **Crell**: EEG → **Letter Image Reconstruction** (28×28 images)

## 🚀 **2-Task Pure EEG Reconstruction Framework**

### **Corrected Task Definitions:**

1. **Task 1**: EEG → **Digit Image Reconstruction** (MindBigData)
   - Input: EEG signals (14 channels)
   - Output: 28×28 digit images (matching Van Gerven)

2. **Task 2**: EEG → **Letter Image Reconstruction** (Crell)
   - Input: EEG signals (64 channels)
   - Output: 28×28 letter images

## Ringkasan
Proyek ini menggunakan 2 dataset utama untuk penelitian **pure reconstruction** multi-task brain decoding dengan Bayesian uncertainty quantification:
1. **2 dataset fMRI** untuk rekonstruksi citra (Van Gerven + Miyawaki)
2. **2 dataset EEG** untuk rekonstruksi citra (MindBigData + Crell)

### ⭐ **BREAKTHROUGH: Pure Reconstruction Framework**
- **EEG → Digit image reconstruction** (MindBigData)
- **EEG → Letter image reconstruction** (Crell Visual)

## Dataset yang Digunakan

### 1. Dataset MindBigData - EEG ⭐ **ENHANCED FOR RECONSTRUCTION**
**File:** `EP1.01.txt`
**Stimulus Images:** `MindbigdataStimuli/` folder (0.jpg - 9.jpg)
**Jenis:** EEG

#### Struktur Dataset:
- **Format:** Tab-separated text file
- **Device:** Emotiv EPOC (14 channels)
- **Sampling Rate:** ~128Hz
- **Task:** EEG → Digit reconstruction ⭐ UPGRADED from classification
- **Paradigm:** Visual digit thinking + Image reconstruction
- **Cross-Modal:** Direct comparison with fMRI digit reconstruction (Van Gerven)

#### Format File:
```
[id]	[event]	[device]	[channel]	[code]	[size]	[data]
```

#### Detail Atribut:
1. **Channels (14 total)**
   - AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
   - **Sampling:** ~128Hz, 2-second epochs
   - **Range:** Raw EEG values in microvolts

2. **Digit Codes**
   - **Values:** 0-9 for digits, -1 for random signals
   - **Task:** Subject thinks about the digit
   - **Paradigm:** Visual digit perception

#### Cara Load Dataset:
```python
def load_mindbigdata_eeg(filepath):
    """Load MindBigData EEG from text file."""
    signals_by_event = defaultdict(lambda: defaultdict(list))

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                event_id = int(parts[1])
                channel = parts[3]
                digit_code = int(parts[4])
                data_str = parts[6]

                data_values = [float(x) for x in data_str.split(',')]
                signals_by_event[event_id][channel].append({
                    'code': digit_code,
                    'data': np.array(data_values)
                })

    return signals_by_event
```

### 2. Dataset Crell - EEG ⭐ **ENHANCED WITH VISUAL-MOTOR SEPARATION**
**File:** `S01.mat`
**Jenis:** EEG
**Stimulus Images:** `crellStimuli/` folder (a.png, d.png, etc.)

#### Struktur Dataset:
- **Format:** MATLAB .mat file
- **Device:** 64-channel EEG system
- **Sampling Rate:** 500Hz
- **Tasks:**
  - **Visual Phase**: Letter image reconstruction (EEG → Images)
  - **Motor Phase**: Letter trajectory reconstruction + Classification (EEG → Trajectories)
- **Paradigm:** Visual-Motor separation

#### ⭐ **PARADIGM TEMPORAL STRUCTURE:**
```
Marker 1: Letter starts to fade in     [Visual Phase Start]
Marker 2: Letter completely faded in   [Pure Visual Phase] ← EXTRACTED
Marker 3: Letter starts to fade out    [Visual-Motor Transition]
Marker 4: Writing starts               [Motor Phase Start]
Marker 5: Writing complete             [Motor Phase End]
```

#### Detail Atribut:
1. **EEG Data (BrainVisionRDA_data)**
   - **Channels:** 64 (full 10-20 system)
   - **Sampling:** 500Hz
   - **Range:** Microvolts (µV)
   - **Reference:** Right mastoid
   - **Visual Epochs:** 1.5s (Marker 2→3, pure visual processing)
   - **Motor Epochs:** 2.0s (Marker 4→5, motor execution)

2. **Motion Capture (MoCap_data)**
   - **Channels:** 3 (x, y, detection_flag)
   - **Sampling:** 30Hz
   - **Task:** Finger trajectory during letter writing

3. **Letters & Stimuli**
   - **Set:** a, d, e, f, j, n, o, s, t, v (10 letters)
   - **Repetitions:** 60 per letter
   - **Stimulus Images:** 28×28 PNG files for each letter
   - **ASCII Encoding:** 100+x (a=100, d=103, e=104, etc.)
   - **Paradigm:** Visual cue → Pure visual processing → Motor execution

#### ⭐ **NEW: Enhanced Data Loading with Visual-Motor Separation:**
```python
def load_crell_eeg_enhanced(filepath):
    """Load Crell EEG with visual-motor paradigm separation."""
    data = scipy.io.loadmat(filepath)

    # Extract paradigm data
    paradigm_one = data['paradigm_one']
    paradigm_two = data['paradigm_two']

    for round_data in [paradigm_one, paradigm_two]:
        eeg_data = round_data['BrainVisionRDA_data'][0, 0]  # (64, timepoints)
        eeg_times = round_data['BrainVisionRDA_time'][0, 0]
        mocap_data = round_data['MoCap_data'][0, 0]  # (3, timepoints)
        marker_data = round_data['ParadigmMarker_data'][0, 0]
        marker_times = round_data['ParadigmMarker_time'][0, 0]

        # Parse trials with temporal markers
        trials = parse_markers(marker_data, marker_times)

        # Extract visual epochs (pure visual processing)
        visual_epochs, visual_labels = extract_visual_epochs(
            eeg_data, eeg_times, trials, epoch_duration=1.5
        )

        # Extract motor epochs (motor execution)
        motor_epochs, motor_labels = extract_motor_epochs(
            eeg_data, eeg_times, trials, epoch_duration=2.0
        )

        # Extract trajectories
        trajectories = extract_trajectories(mocap_data, mocap_times, trials)

    return {
        'visual_eeg': visual_epochs,      # (trials, 64, 750) - for image reconstruction
        'motor_eeg': motor_epochs,        # (trials, 64, 1000) - for trajectory reconstruction
        'visual_labels': visual_labels,   # Letter labels for visual phase
        'motor_labels': motor_labels,     # Letter labels for motor phase
        'trajectories': trajectories      # (trials, 200) - x,y coordinates
    }

def load_crell_stimuli(stimuli_dir):
    """Load Crell stimulus images for visual reconstruction."""
    letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
    stimuli = {}

    for letter in letters:
        img_path = f"{stimuli_dir}/{letter}.png"
        img = load_and_preprocess_image(img_path, target_size=(28, 28))
        stimuli[letter] = img

    return stimuli
```



