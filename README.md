# FastICA Blind Source Separation (BSS) Project

## 📋 Project Overview

This project implements and evaluates **FastICA algorithm** for blind source separation of audio signals, with a focus on understanding the gap between theoretical assumptions and real-world acoustic conditions.

## 🏛️ Academic Context

- **Institution**: Politecnico di Milano (PoliMi)
- **Course**: Information Theory
- **Academic Year**: 2024/2025
- **Project Sponsor**: BdSound Company

## 👥 Team Members

- **F. Afshinrad**
- **A. Edareh Heidarabadi**

## 🎯 Objectives

1. Implement FastICA algorithm with multiple non-linearity functions
2. Compare performance across different mixing scenarios:
   - Instantaneous mixing (ICA-friendly)
   - Room acoustics simulation (realistic conditions)
   - Hybrid approaches
3. Evaluate separation quality using standard BSS metrics (SIR, SDR, Correlation)
4. Analyze the theory-practice gap in blind source separation

## 🚀 Key Features

### Algorithm Implementation
- **FastICA Core**: Newton-based optimization with multiple non-linearity options
- **Non-linearity Functions**: logcosh, exp, pow4 with different robustness profiles
- **Orthogonalization**: Both deflation and symmetric approaches
- **Weight Initialization**: Smart initialization for faster convergence

### Realistic Audio Mixing
- **Instantaneous Mixing**: Matrix-based signal combination
- **Room Simulation**: Physics-based acoustic modeling using pyroomacoustics
- **Configurable Scenarios**: Various microphone spacings and room configurations

### Comprehensive Evaluation
- **BSS Metrics**: SIR, SDR, correlation coefficient calculation
- **Performance Analysis**: Convergence speed and quality assessment
- **Systematic Testing**: Controlled experiments across mixing scenarios

## 📁 Project Structure

```
├── fast_ica.py           # Core FastICA algorithm implementation
├── combine_voices.py     # Audio mixing and room simulation
├── bss_metrics.py        # Evaluation metrics (SIR, SDR, correlation)
├── main.py              # Main experiment runner
├── voices/              # Audio files and results
└── README.md           # This file
```

## 🛠️ Dependencies

```python
numpy
scipy
pyroomacoustics
```

## 🔬 Key Findings

### What Works Well
- **Instantaneous mixing** with well-conditioned matrices (SIR > 12 dB)
- **Optimal microphone spacing** (3cm spacing vs 1.5cm)
- **Clean acoustic environments** with minimal reverberation

### Limitations Discovered
- **Room acoustics** create convolutive mixing that violates ICA assumptions
- **Matrix conditioning** predicts separation performance
- **Standard time-domain ICA** inadequate for realistic acoustic scenarios

### Performance Results
| Mixing Method | Mean SIR (dB) | Condition # | Quality |
|---------------|---------------|-------------|---------|
| Instantaneous (3.0cm) | 12.96 | 6.88 | Good |
| Instantaneous (1.5cm) | 0.63 | 24.24 | Fair |
| Traditional Room | -23.67 | 68.73 | Poor |

## 🎵 Usage Example

```python
from combine_voices import CombineVoices
from fast_ica import fast_ica_newton
from bss_metrics import BSSMetrics

# Create mixer and load audio sources
mixer = CombineVoices('./voices/music.wav', './voices/talk.wav')

# Test different mixing scenarios
X = mixer.mix_sources_instantaneous(mic_spacing=0.03)  # 3cm spacing

# Preprocess signals
Z, V, mean_vec = preprocess_mixed(X, mixer.sr)

# Apply FastICA
W, S = fast_ica_newton(Z, n_components=2, fun='logcosh')

# Evaluate separation quality
evaluator = BSSMetrics()
results = evaluator.evaluate_separation(original_sources, S)
evaluator.print_results(results)
```

## 📊 Research Contributions

1. **Matrix Conditioning as Performance Predictor**: Discovered that condition number reliably predicts ICA separation quality
2. **Systematic Reality Testing**: Created framework for evaluating BSS algorithms across theory-practice spectrum
3. **Implementation Best Practices**: Modular, extensible codebase with comprehensive evaluation

## 📚 References

- Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications
- Blind Source Separation of Underwater Acoustic Signal by Use of Negentropy-based Fast ICA Algorithm
- Blind Audio Source Separation Using Weight Initialized Independent Component Analysis

## 📄 License

This project is developed for academic purposes as part of the Information Theory course at Politecnico di Milano in collaboration with BdSound Company.

---

**Note**: This implementation demonstrates the theoretical foundations and practical limitations of ICA-based blind source separation, providing insights into the gap between algorithmic assumptions and real-world acoustic conditions.
