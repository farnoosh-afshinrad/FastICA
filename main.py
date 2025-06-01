import os
import numpy as np
from scipy.io import wavfile
import soundfile as sf
from combine_voices import CombineVoices
from fast_ica import fast_ica_newton
from bss_metrics import BSSMetrics  # Add this import

def center(X: np.ndarray) -> tuple:
    """Center data by removing mean"""
    mean = np.mean(X, axis=1, keepdims=True)
    Xc = X - mean
    return Xc, mean


def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix using time average as specified in paper:
    Rx = (1/(n-1)) Σ X(i)X(i)ᵀ
    """
    n_samples = X.shape[1]
    # Time-averaged correlation matrix
    Rx = (X @ X.T) / (n_samples - 1)
    return Rx


def whiten(Xc: np.ndarray) -> tuple:
    """
    Proper whitening matrix derivation following paper:
    V = D^(-1/2)Qᵀ with explicit eigendecomposition Rx = QDQᵀ
    """
    # Compute correlation matrix
    Rx = compute_correlation_matrix(Xc)
    
    # Eigendecomposition: Rx = QDQᵀ
    eigenvalues, Q = np.linalg.eigh(Rx)
    
    # D^(-1/2) - inverse square root of eigenvalue matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))  # small epsilon for stability
    
    # Whitening matrix: V = D^(-1/2)Qᵀ
    V = D_inv_sqrt @ Q.T
    
    # Whitened data
    Z = V @ Xc
    
    # Verify whitening: E[ZZᵀ] should be I
    Rz = compute_correlation_matrix(Z)
    print(f"Whitening verification - should be ~I:")
    print(f"Diagonal elements: {np.diag(Rz)}")
    print(f"Off-diagonal norm: {np.linalg.norm(Rz - np.diag(np.diag(Rz))):.6f}")
    
    return Z, V

def preprocess_mixed(mixer):
    """Load and preprocess mixed signals from multiple microphones"""
    # Get the mixed signals from the room simulation
    # mixer.room.mic_array.signals has shape (n_mics, n_samples)
    X = mixer.room.mic_array.signals.astype(np.float32)
    sr = mixer.sr
    
    print(f"\nLoaded mixed signals: shape={X.shape}, sr={sr}")
    
    # Center the data
    Xc, mean_vec = center(X)
    print(f"After centering: mean per channel = {np.mean(Xc, axis=1)}")
    
    # Whiten the data
    Z, V = whiten(Xc)
    
    return Z, V, mean_vec, sr


def main():
    print("FastICA Implementation with Complete Mathematical Framework")
    print("="*60)
    
    # Step 1: Create mixed signals with better mixing conditions
    print("\n1. Creating mixed signals...")
    mixer = CombineVoices('./voices/music.wav', './voices/talk.wav')
    # Use parameters that create different mixtures at each microphone:
    # - Lower absorption (0.3) for some reflections
    # - Higher order (2) for multiple reflection paths  
    # - Larger mic spacing (0.2m) for more spatial diversity
    mixer.mix_sources()
    mixer.save_mics()
    
    # Step 2: Preprocess - use the mixer object to get proper mixed signals
    print("\n2. Preprocessing (centering + whitening)...")
    Z, V, mean_vec, sr = preprocess_mixed(mixer)
    
    # Step 3: Apply FastICA with different non-linearities
    print("\n3. Applying FastICA with Newton iteration...")
    
    functions = ['exp', 'logcosh', 'pow4']
    all_results = {}
    best_function = None
    best_sir = -np.inf
    
    for fun in functions:
        print(f"\n{'='*50}")
        print(f"Testing with '{fun}' non-linearity")
        print('='*50)
        
        try:
            W, S = fast_ica_newton(Z, n_components=2, fun=fun, tol=1e-6, max_iter=200)
            
            # Get original sources
            original_sources = np.array([mixer.s1, mixer.s2])
            
            # Ensure same length
            min_len = min(original_sources.shape[1], S.shape[1])
            original_trimmed = original_sources[:, :min_len]
            separated_trimmed = S[:, :min_len]
            
            # Evaluate
            print(f"\nEvaluating separation quality for '{fun}':")
            evaluator = BSSMetrics()
            results = evaluator.evaluate_separation(original_trimmed, separated_trimmed)
            evaluator.print_results(results)
            
            all_results[fun] = results
            
            if results['Mean_SIR'] > best_sir:
                best_sir = results['Mean_SIR']
                best_function = fun
                best_W, best_S = W, S
                
        except Exception as e:
            print(f"Error with '{fun}': {e}")
            continue
    
    # Step 4: Save best results
    if best_function:
        print(f"\n4. Best separation achieved with '{best_function}' function")
        print(f"   Mean SIR: {best_sir:.2f} dB")
        
        # Save separated sources
        print("\n5. Saving separated sources...")
        for i in range(best_S.shape[0]):
            # Normalize
            signal = best_S[i]
            signal = signal / np.max(np.abs(signal)) * 0.9
            
            filename = f'./voices/separated_{i+1}_{best_function}.wav'
            wavfile.write(filename, sr, signal.astype(np.float32))
            print(f"   Saved: {filename}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Function':<10} {'Mean SIR':<12} {'Mean SDR':<12} {'Mean Corr':<12}")
    print('-'*60)
    
    for fun, results in all_results.items():
        print(f"{fun:<10} {results['Mean_SIR']:<12.2f} {results['Mean_SDR']:<12.2f} "
              f"{results['Mean_Correlation']:<12.4f}")
    
    print(f"\nBest: {best_function} (SIR: {best_sir:.2f} dB)")
    print("="*60)


if __name__ == "__main__":
    main()