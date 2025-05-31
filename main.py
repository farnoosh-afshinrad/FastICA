import os
import numpy as np
from scipy.io import wavfile
import soundfile as sf
from combine_voices import CombineVoices
from fast_ica import fast_ica
from bss_metrics import BSSMetrics  # Add this import

def center(X: np.ndarray) -> tuple:
    mean = np.mean(X, axis=1, keepdims=True)
    Xc = X - mean
    return Xc, mean

def whiten(Xc: np.ndarray) -> tuple:
    n_channels, n_samples = Xc.shape
    R = (Xc @ Xc.T) / (n_samples - 1)
    d, Q = np.linalg.eigh(R)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    V = D_inv_sqrt @ Q.T
    Z = V @ Xc
    return Z, V

def combine_sources():
    mixer = CombineVoices('./voices/music.wav', './voices/talk.wav')
    mixer.mix_sources(absorption=1.0, max_order=0)
    mixer.save_mics()
    return mixer  # Return mixer object for evaluation

def preprocess_mixed(path: str = './voices/test_voice.wav'):
    """
    Loads any file libsndfile supports (wav, flac, m4a, etc.),
    centers and whitens its channels for ICA preprocessing.
    """
    # Load with soundfile for broader format support
    data, sr = sf.read(path)
    print(f"Loaded {path}: shape={data.shape}, sr={sr}")
    
    # Convert to numpy float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Transpose so channels are rows: (n_channels, n_samples)
    if data.ndim == 1:
        X = data.reshape(1, -1)
    else:
        X = data.T  # Now shape is (n_channels, n_samples)
    
    print(f"X shape after transpose: {X.shape}")
    
    # Center
    Xc, mean_vec = center(X)
    print(f"After centering: mean per channel = {np.mean(Xc, axis=1)}")
    
    # Whiten
    Z, V = whiten(Xc)
    print(f"After whitening: Z shape = {Z.shape}")
    print(f"Whitening check - covariance diagonal: {np.diag((Z @ Z.T) / (Z.shape[1] - 1))}")
    
    return Z, V, mean_vec, sr

def evaluate_separation_quality(original_sources, separated_sources):
    """
    Evaluate the quality of source separation using multiple metrics.
    """
    evaluator = BSSMetrics()
    
    # Ensure sources are in the right format (n_sources, n_samples)
    if original_sources.shape[0] > original_sources.shape[1]:
        original_sources = original_sources.T
    if separated_sources.shape[0] > separated_sources.shape[1]:
        separated_sources = separated_sources.T
    
    # Comprehensive evaluation
    results = evaluator.evaluate_separation(original_sources, separated_sources)
    
    # Print formatted results
    evaluator.print_results(results)
    
    return results

def main():
    print("Starting Blind Source Separation with FastICA")
    print("=" * 50)
    
    # Step 1: Create realistic mixed signals
    print("\n1. Creating mixed signals with room simulation...")
    mixer = combine_sources()
    
    # Step 2: Load and preprocess the mixed signals
    print("\n2. Preprocessing mixed signals...")
    Z, V, mean_vec, sr = preprocess_mixed('./voices/mic1.wav')
    
    # Step 3: Apply FastICA
    print("\n3. Applying FastICA algorithm...")
    n_components = 2  # We have 2 source signals
    
    # Try different non-linearities and compare results
    functions = ['pow3', 'tanh', 'gauss']
    best_results = None
    best_function = None
    best_sir = -np.inf
    
    all_evaluations = {}
    
    for fun in functions:
        print(f"\n   Testing with '{fun}' non-linearity...")
        try:
            W, S = fast_ica(Z, n_components=n_components, fun=fun, tol=1e-6, max_iter=2000)
            print(f"   FastICA converged successfully with '{fun}'")
            
            # Step 4: Get original sources for comparison
            original_sources = np.array([mixer.s1, mixer.s2])  # Shape: (2, n_samples)
            
            # Ensure same length for comparison
            min_len = min(original_sources.shape[1], S.shape[1])
            original_sources_trimmed = original_sources[:, :min_len]
            separated_sources_trimmed = S[:, :min_len]
            
            # Step 5: Evaluate separation quality
            print(f"\n   Evaluating separation quality for '{fun}':")
            results = evaluate_separation_quality(original_sources_trimmed, separated_sources_trimmed)
            all_evaluations[fun] = results
            
            # Keep track of best performing function
            mean_sir = results['Mean_SIR']
            if mean_sir > best_sir:
                best_sir = mean_sir
                best_function = fun
                best_results = (W, S, results)
                
        except Exception as e:
            print(f"   Error with '{fun}': {e}")
            continue
    
    # Step 6: Report best results and save separated signals
    if best_results is not None:
        W_best, S_best, eval_results = best_results
        
        print(f"\n4. Best separation achieved with '{best_function}' function")
        print(f"   Mean SIR: {best_sir:.2f} dB")
        
        # Save separated sources
        print("\n5. Saving separated sources...")
        
        # Ensure we don't exceed the original length
        min_len = min(len(mixer.s1), len(mixer.s2), S_best.shape[1])
        
        # Save each separated source
        for i in range(S_best.shape[0]):
            separated_signal = S_best[i, :min_len]
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(separated_signal))
            if max_val > 0:
                separated_signal = separated_signal / max_val * 0.9
            
            output_path = f'./voices/separated_source_{i+1}_{best_function}.wav'
            wavfile.write(output_path, sr, separated_signal.astype(np.float32))
            print(f"   Saved: {output_path}")
        
        # Step 7: Summary comparison across all functions
        print(f"\n6. Performance Comparison Summary:")
        print("-" * 50)
        print(f"{'Function':<10} {'Mean SIR':<10} {'Mean SDR':<10} {'Mean Corr':<12}")
        print("-" * 50)
        
        for fun in functions:
            if fun in all_evaluations:
                results = all_evaluations[fun]
                print(f"{fun:<10} {results['Mean_SIR']:<10.2f} {results['Mean_SDR']:<10.2f} {results['Mean_Correlation']:<12.4f}")
        
        print("-" * 50)
        print(f"Best function: {best_function} (SIR: {best_sir:.2f} dB)")
        
    else:
        print("\nError: No successful separations achieved!")
        return
    
    print(f"\nBlind Source Separation completed successfully!")
    print(f"Check the './voices/' directory for separated audio files.")

if __name__ == "__main__":
    main()