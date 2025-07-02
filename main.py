import numpy as np
from scipy.io import wavfile
from combine_voices import CombineVoices
from fast_ica import fast_ica_newton
from bss_metrics import BSSMetrics

def center(X: np.ndarray) -> tuple:
    """Center data by removing mean"""
    mean = np.mean(X, axis=1, keepdims=True)
    Xc = X - mean
    return Xc, mean

def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Compute correlation matrix"""
    n_samples = X.shape[1]
    Rx = (X @ X.T) / (n_samples - 1)
    return Rx

def whiten(Xc: np.ndarray) -> tuple:
    """Whiten the centered data"""
    Rx = compute_correlation_matrix(Xc)
    eigenvalues, Q = np.linalg.eigh(Rx)
    
    # Regularize small eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    V = D_inv_sqrt @ Q.T
    Z = V @ Xc
    
    return Z, V

def preprocess_mixed(X, sr):
    """Preprocess mixed signals"""
    print(f"\nPreprocessing signals...")
    
    # Center
    Xc, mean_vec = center(X)
    print(f"After centering: mean = {np.mean(Xc, axis=1)}")
    
    # Check condition before whitening
    cond = np.linalg.cond(np.cov(Xc))
    print(f"Condition number: {cond:.2f}")
    
    # Whiten
    Z, V = whiten(Xc)
    
    # Verify whitening
    Rz = compute_correlation_matrix(Z)
    print(f"Whitening check - diagonal: {np.diag(Rz)}")
    
    return Z, V, mean_vec


import numpy as np
import matplotlib.pyplot as plt


def plot_sources_and_groundtruth(S_true, S_est, source_names=None, fs=1.0):
    """
    Plot each true source next to its estimated counterpart.

    Parameters
    ----------
    S_true : np.ndarray, shape (n_sources, n_samples)
        The ground-truth sources.
    S_est : np.ndarray, shape (n_sources, n_samples)
        The estimated sources (already permuted/scaled to match S_true).
    source_names : list of str, optional
        Names for each source (length n_sources). Defaults to ['1','2',…].
    fs : float, optional
        Sampling frequency for the time axis (defaults to 1.0, i.e. samples).
    """
    n_sources, n_samples = S_true.shape
    t = np.arange(n_samples) / fs

    # default names if none given
    if source_names is None:
        source_names = [str(i + 1) for i in range(n_sources)]

    fig, axes = plt.subplots(n_sources, 2, figsize=(10, 2.5 * n_sources), sharex=True)

    # if there's only one source, axes will be 1D
    if n_sources == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_sources):
        ax_true, ax_est = axes[i]

        ax_true.plot(t, S_true[i])
        ax_true.set_title(f"True Source {i + 1}: {source_names[i]}")
        ax_true.set_ylabel("Amplitude")

        ax_est.plot(t, S_est[i])
        ax_est.set_title(f"Estimated Source {i + 1}: {source_names[i]}")
        ax_est.set_ylabel("Amplitude")

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def main():
    print("FastICA with Improved Mixing Approaches")
    print("="*60)
    
    # Test different mixing approaches
    approaches = [
        ("Instantaneous (phone-like)", "instantaneous", 0.015),
        ("Instantaneous (tablet)", "instantaneous", 0.03),
        ("Hybrid (light reverb)", "hybrid", 0.015),
        ("Nearfield room", "nearfield", 0.02),
        ("Traditional room", "traditional", 0.02)
    ]
    
    results_summary = []
    
    for desc, method, spacing in approaches:
        print(f"\n{'='*60}")
        print(f"Testing: {desc}")
        print('='*60)
        
        # Create mixer
        mixer = CombineVoices('./voices/music.wav', './voices/talk.wav')
        
        # Apply mixing method
        if method == "instantaneous":
            X = mixer.mix_sources_instantaneous(mic_spacing=spacing)
        elif method == "hybrid":
            X = mixer.create_hybrid_mixing(mic_spacing=spacing)
        elif method == "nearfield":
            X = mixer.mix_sources_nearfield(mic_spacing=spacing)
        else:  # traditional
            mixer.mix_sources(mic_spacing=spacing, absorption=0.2, max_order=3)
            X = mixer.room.mic_array.signals
        
        # Save mixed signals
        mixer.save_mics()
        
        # Preprocess
        Z, V, mean_vec = preprocess_mixed(X, mixer.sr)
        
        # Apply FastICA
        print("\nApplying FastICA...")
        W, S = fast_ica_newton(Z, n_components=2, fun='logcosh', 
                              max_iter=200,apply_weight_init=True)
        # Evaluate
        original_sources = np.array([mixer.s1, mixer.s2])
        min_len = min(original_sources.shape[1], S.shape[1])
        
        evaluator = BSSMetrics()
        plot_sources_and_groundtruth(original_sources[:, :min_len],
            S[:, :min_len],source_names=['Music', 'Talk'], fs=mixer.sr)
        results = evaluator.evaluate_separation(
            original_sources[:, :min_len], 
            S[:, :min_len]
        )
        evaluator.print_results(results)
        
        results_summary.append((desc, results['Mean_SIR']))
        
        # Save best result
        if method == "instantaneous" and spacing == 0.015:
            print("\nSaving separated sources...")
            for i in range(S.shape[0]):
                signal = S[i] / (np.max(np.abs(S[i])) + 1e-10) * 0.9
                filename = f'./voices/separated_{i+1}_instantaneous.wav'
                wavfile.write(filename, mixer.sr, signal.astype(np.float32))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: ICA Performance by Mixing Method")
    print('='*60)
    print(f"{'Method':<30} {'Mean SIR (dB)':<15} {'Quality':<20}")
    print('-'*60)
    
    for desc, sir in results_summary:
        quality = "Excellent" if sir > 20 else "Good" if sir > 10 else "Fair" if sir > 0 else "Poor"
        print(f"{desc:<30} {sir:<15.1f} {quality:<20}")
    
    print("\nConclusions:")
    print("1. Instantaneous mixing works well with ICA (as expected)")
    print("2. Room acoustics create convolutive mixtures that standard ICA cannot handle")
    print("3. For real phone BSS, you need:")
    print("   - Frequency-domain ICA, or")
    print("   - Time-delayed ICA algorithms, or")
    print("   - Beamforming + post-filtering approaches")
    print("\nUsing of pyroomacoustics is correct for realistic simulation,")
    print("but standard time-domain FastICA is not the right algorithm for this scenario!")

if __name__ == "__main__":
    main()