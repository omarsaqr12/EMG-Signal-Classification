# EMG Signal Classification using KNN

A Digital Signal Processing project for classifying hand movements from surface Electromyography (sEMG) signals using K-Nearest Neighbors (KNN) classification with feature extraction in both time and frequency domains.

## 📋 Overview

This project implements a complete pipeline for processing and classifying EMG signals:

1. **Data Loading** - Load multi-channel EMG data from MATLAB `.mat` files
2. **Preprocessing** - Apply high-pass Butterworth filtering to remove low-frequency noise
3. **Trial Segmentation** - Segment continuous data into individual movement trials based on stimulus changes
4. **Feature Extraction** - Extract features in both time-domain and frequency-domain
5. **Classification** - Use KNN classifier with Leave-One-Out Cross-Validation
6. **Optimization** - Find optimal K value and best-performing channel/method

## 🔬 Feature Extraction Methods

### Time-Domain Features
- Mean Absolute Value (MAV)
- Root Mean Square (RMS)
- Zero Crossings (ZC)
- Slope Sign Changes (SSC)
- Raw signal flattening

### Frequency-Domain Features
- Mean Frequency (MNF)
- Median Frequency (MDF)
- Power Spectral Density analysis via FFT

### Combined Features
- Concatenation of time and frequency domain features for enhanced classification

## 📊 Results

The project achieves high classification accuracy across three test subjects:

| Subject | Best Method | Best Accuracy |
|---------|-------------|---------------|
| 1 | All Channels (Time) | 95% |
| 2 | All Channels (Frequency) | 100% |
| 3 | All Channels (Time/Freq) | 100% |

**Key Findings:**
- Frequency-domain features consistently outperform time-domain features for single-channel classification
- Combining data from all channels significantly improves classification performance
- Lower K values (K=1-3) generally yield better results with this dataset

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/EMG-Signal-Classification.git
cd EMG-Signal-Classification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Usage

```bash
# Run the main classification script
python pythonProject/main.py
```

### Using Alternative Scripts

```bash
# Run with different analysis approach
python g.py
```

### Analyzing Different Subjects

Modify the file path in the main script to analyze different subjects:

```python
emg, stimulus = load_data_from_mat('subject1.mat')  # Change to subject2.mat or subject3.mat
```

## 📁 Project Structure

```
EMG-Signal-Classification/
├── pythonProject/
│   └── main.py              # Main classification script
├── Project Dataset/
│   ├── subject_1.mat        # EMG data for subject 1
│   ├── subject_2.mat        # EMG data for subject 2
│   └── subject_3.mat        # EMG data for subject 3
├── subject1.mat             # EMG data (alternative location)
├── subject2.mat
├── subject3.mat
├── g.py                     # Alternative analysis script
├── results.txt              # Detailed classification results
├── classification_results.csv  # Summary results in CSV format
├── docs/
│   ├── DSP_Project.pdf      # Project documentation
│   ├── EMG.pdf              # EMG background information
│   └── Project Description.pdf
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## 📈 Data Format

The project expects MATLAB `.mat` files containing:
- `emg`: 2D array of EMG signals (samples × channels)
- `stimulus`: 1D array of movement labels
- `subject`: Subject identifier

**Signal Parameters:**
- Sampling Rate: 100 Hz
- Number of Channels: 10 electrodes
- Trial Duration: ~5 seconds per movement

## 🔧 Configuration

Key parameters that can be modified:

```python
# High-pass filter settings
cutoff_frequency = 10  # Hz
filter_order = 4
sampling_rate = 100  # Hz

# KNN settings
k_range = range(1, 20)  # K values to evaluate

# Trial segmentation
trial_duration = 5  # seconds
```

## 📚 Methods Explained

### Method 1: Single Channel Time-Domain
Evaluates each electrode independently using raw EMG data as features.

### Method 2: Single Channel Frequency-Domain
Transforms EMG signals to frequency domain using FFT and extracts spectral features.

### Method 3: Combined All-Channels
Concatenates features from all 10 electrodes for comprehensive classification.

## 🎓 Academic Context

This project was developed for **CSCE3611 - Digital Signal Processing** at The American University in Cairo.

**Contributors:**
- Adham Ali
- Omar Saqr
- Saif Abdelfattah

**Supervised by:** Dr. Seif Eldawlatly

## 📄 License

This project is for educational purposes. Please contact the authors for usage rights.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration opportunities, please open an issue on this repository.

---

*Last updated: December 2024*
