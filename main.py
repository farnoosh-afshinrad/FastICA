# import os
# import numpy as np
# from scipy.io import wavfile
# import soundfile as sf
# from combine_voices import CombineVoices
# from fast_ica import fast_ica_newton
# from bss_metrics import BSSMetrics

# def center(X: np.ndarray) -> tuple:
#     """Center data by removing mean"""
#     mean = np.mean(X, axis=1, keepdims=True)
#     Xc = X - mean
#     return Xc, mean


# def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
#     """
#     Compute correlation matrix using time average as specified in paper:
#     Rx = (1/(n-1)) Σ X(i)X(i)ᵀ
#     """
#     n_samples = X.shape[1]
#     # Time-averaged correlation matrix
#     Rx = (X @ X.T) / (n_samples - 1)
#     return Rx


# def whiten(Xc: np.ndarray) -> tuple:
#     """
#     Proper whitening matrix derivation following paper:
#     V = D^(-1/2)Qᵀ with explicit eigendecomposition Rx = QDQᵀ
#     """
#     # Compute correlation matrix
#     Rx = compute_correlation_matrix(Xc)
    
#     # Eigendecomposition: Rx = QDQᵀ
#     eigenvalues, Q = np.linalg.eigh(Rx)
    
#     # D^(-1/2) - inverse square root of eigenvalue matrix
#     D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
    
#     # Whitening matrix: V = D^(-1/2)Qᵀ
#     V = D_inv_sqrt @ Q.T
    
#     # Whitened data
#     Z = V @ Xc
    
#     # Verify whitening: E[ZZᵀ] should be I
#     Rz = compute_correlation_matrix(Z)
#     print(f"Whitening verification - should be ~I:")
#     print(f"Diagonal elements: {np.diag(Rz)}")
#     print(f"Off-diagonal norm: {np.linalg.norm(Rz - np.diag(np.diag(Rz))):.6f}")
    
#     return Z, V


# def preprocess_mixed(mixer):
#     """Load and preprocess mixed signals from multiple microphones"""
#     # Get the mixed signals from the room simulation
#     X = mixer.room.mic_array.signals.astype(np.float32)
#     sr = mixer.sr
    
#     print(f"\nLoaded mixed signals: shape={X.shape}, sr={sr}")
    
#     # Center the data
#     Xc, mean_vec = center(X)
#     print(f"After centering: mean per channel = {np.mean(Xc, axis=1)}")
    
#     # Whiten the data
#     Z, V = whiten(Xc)
    
#     return Z, V, mean_vec, sr


# def main():
#     print("FastICA with Realistic Pyroomacoustics Simulation")
#     print("="*60)
    
#     # Step 1: Demonstrate mic distance effect
#     print("\n1. Analyzing mic distance effect on separation difficulty...")
#     mixer = CombineVoices('./voices/music.wav', './voices/talk.wav')
#     distances, correlations = mixer.test_different_mic_distances()
    
#     # Step 2: Run full separation test with phone-like configuration
#     print("\n\n2. Running FastICA with phone-like mic configuration (1.5cm spacing)...")
#     print("="*60)
    
#     # Create realistic phone-like mixing
#     mixer.mix_sources(
#         mic_spacing=0.015,    # 1.5cm - typical phone distance
#         absorption=0.2,       # Some absorption for realism
#         max_order=3,          # Include reflections
#         room_dim=[6.0, 5.0, 3.0],
#         apply_noise=True      # Add realistic noise
#     )
#     mixer.save_mics()
    
#     # Preprocess
#     print("\n3. Preprocessing (centering + whitening)...")
#     Z, V, mean_vec, sr = preprocess_mixed(mixer)
    
#     # Apply FastICA with different configurations
#     print("\n4. Applying FastICA...")
    
#     # Test both orthogonalization methods with best non-linearity
#     methods = ['deflation', 'symmetric']
#     results_by_method = {}
    
#     for method in methods:
#         print(f"\n{'-'*50}")
#         print(f"Testing {method} orthogonalization")
#         print('-'*50)
        
#         W, S = fast_ica_newton(Z, n_components=2, fun='exp', 
#                               tol=1e-6, max_iter=200, 
#                               orthogonalization=method)
        
#         # Evaluate
#         original_sources = np.array([mixer.s1, mixer.s2])
#         min_len = min(original_sources.shape[1], S.shape[1])
#         original_trimmed = original_sources[:, :min_len]
#         separated_trimmed = S[:, :min_len]
        
#         evaluator = BSSMetrics()
#         results = evaluator.evaluate_separation(original_trimmed, separated_trimmed)
#         evaluator.print_results(results)
        
#         results_by_method[method] = results
        
#         # Save best result
#         if method == 'deflation':  # Usually performs better
#             for i in range(S.shape[0]):
#                 signal = S[i]
#                 signal = signal / np.max(np.abs(signal)) * 0.9
#                 filename = f'./voices/separated_{i+1}_phone_config.wav'
#                 wavfile.write(filename, sr, signal.astype(np.float32))
    
#     # Step 3: Compare with easier configuration (larger mic spacing)
#     print("\n\n5. Comparing with larger mic spacing (10cm)...")
#     print("="*60)
    
#     mixer2 = CombineVoices('./voices/music.wav', './voices/talk.wav', seed=1)
#     mixer2.mix_sources(
#         mic_spacing=0.10,     # 10cm - much easier
#         absorption=0.2,
#         max_order=3,
#         room_dim=[6.0, 5.0, 3.0],
#         apply_noise=True
#     )
    
#     Z2, V2, mean_vec2, sr2 = preprocess_mixed(mixer2)
#     W2, S2 = fast_ica_newton(Z2, n_components=2, fun='exp', 
#                             tol=1e-6, max_iter=200)
    
#     # Evaluate
#     original_sources2 = np.array([mixer2.s1, mixer2.s2])
#     min_len2 = min(original_sources2.shape[1], S2.shape[1])
#     results_10cm = evaluator.evaluate_separation(
#         original_sources2[:, :min_len2], 
#         S2[:, :min_len2]
#     )
    
#     # Final comparison
#     print(f"\n{'='*60}")
#     print("FINAL COMPARISON: Effect of Microphone Distance")
#     print('='*60)
#     print(f"{'Configuration':<25} {'Mean SIR (dB)':<15} {'Difficulty':<20}")
#     print('-'*60)
    
#     phone_sir = results_by_method['deflation']['Mean_SIR']
#     print(f"{'Phone-like (1.5cm)':<25} {phone_sir:<15.2f} {'Hard (realistic)':<20}")
#     print(f"{'Larger spacing (10cm)':<25} {results_10cm['Mean_SIR']:<15.2f} {'Easier':<20}")
    
#     print("\nConclusion:")
#     print("- Pyroomacoustics correctly simulates that closer mics receive more similar signals")
#     print("="*60)


# if __name__ == "__main__":
#     main()

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
                              max_iter=200, orthogonalization='deflation')
        
        # Evaluate
        original_sources = np.array([mixer.s1, mixer.s2])
        min_len = min(original_sources.shape[1], S.shape[1])
        
        evaluator = BSSMetrics()
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
    print("\nYour TA's suggestion of pyroomacoustics is correct for realistic simulation,")
    print("but standard time-domain FastICA is not the right algorithm for this scenario!")

if __name__ == "__main__":
    main()