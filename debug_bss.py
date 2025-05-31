import os
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import matplotlib.pyplot as plt
from combine_voices import CombineVoices
from fast_ica import fast_ica
from bss_metrics import BSSMetrics

def analyze_sources_and_mixing():
    """
    Comprehensive diagnostic analysis of sources and mixing process.
    """
    print("=== DIAGNOSTIC ANALYSIS ===")
    
    # Load original sources
    mixer = CombineVoices('./voices/music.wav', './voices/talk.wav')
    
    print(f"Original source shapes: s1={mixer.s1.shape}, s2={mixer.s2.shape}")
    print(f"Source statistics:")
    print(f"  s1: mean={np.mean(mixer.s1):.6f}, std={np.std(mixer.s1):.6f}")
    print(f"  s2: mean={np.mean(mixer.s2):.6f}, std={np.std(mixer.s2):.6f}")
    
    # Check if sources are sufficiently different
    cross_corr = np.corrcoef(mixer.s1, mixer.s2)[0, 1]
    print(f"Cross-correlation between sources: {cross_corr:.4f}")
    if abs(cross_corr) > 0.5:
        print("WARNING: Sources are highly correlated - may be difficult to separate!")
    
    # Test different room configurations
    configs = [
        {"absorption": 1.0, "max_order": 0, "name": "Anechoic"},
        {"absorption": 0.3, "max_order": 1, "name": "Slight_Reverb"},
        {"absorption": 0.1, "max_order": 2, "name": "High_Reverb"}
    ]
    
    best_config = None
    best_separation_potential = 0
    
    for config in configs:
        print(f"\n--- Testing {config['name']} room ---")
        
        # Create mixing
        mixer_test = CombineVoices('./voices/music.wav', './voices/talk.wav')
        mixer_test.mix_sources(
            absorption=config["absorption"], 
            max_order=config["max_order"]
        )
        
        # Get mixed signals
        mixed_signals = mixer_test.room.mic_array.signals  # Shape: (2, n_samples)
        print(f"Mixed signals shape: {mixed_signals.shape}")
        
        # Analyze mixing matrix effectiveness
        min_len = min(len(mixer_test.s1), len(mixer_test.s2), mixed_signals.shape[1])
        
        # Approximate the mixing matrix by comparing sources to mixed signals
        sources_matrix = np.array([mixer_test.s1[:min_len], mixer_test.s2[:min_len]])
        mixed_trimmed = mixed_signals[:, :min_len]
        
        # Check if mixing is actually happening
        source1_in_mic1 = np.corrcoef(sources_matrix[0], mixed_trimmed[0])[0, 1]
        source1_in_mic2 = np.corrcoef(sources_matrix[0], mixed_trimmed[1])[0, 1]
        source2_in_mic1 = np.corrcoef(sources_matrix[1], mixed_trimmed[0])[0, 1]
        source2_in_mic2 = np.corrcoef(sources_matrix[1], mixed_trimmed[1])[0, 1]
        
        print(f"Source 1 correlation with mic1: {source1_in_mic1:.4f}, mic2: {source1_in_mic2:.4f}")
        print(f"Source 2 correlation with mic1: {source2_in_mic1:.4f}, mic2: {source2_in_mic2:.4f}")
        
        # Separation potential: we want both sources in both mics, but differently
        separation_potential = abs(source1_in_mic1 - source1_in_mic2) + abs(source2_in_mic1 - source2_in_mic2)
        print(f"Separation potential: {separation_potential:.4f}")
        
        if separation_potential > best_separation_potential:
            best_separation_potential = separation_potential
            best_config = config
    
    print(f"\nBest configuration: {best_config['name']} (potential: {best_separation_potential:.4f})")
    return best_config

def test_simple_mixing():
    """
    Test with a simple, controlled mixing scenario.
    """
    print("\n=== TESTING SIMPLE MIXING ===")
    
    # Load sources
    mixer = CombineVoices('./voices/music.wav', './voices/talk.wav')
    
    # Create simple linear mixing (without room acoustics)
    min_len = min(len(mixer.s1), len(mixer.s2))
    sources = np.array([mixer.s1[:min_len], mixer.s2[:min_len]])
    
    # Simple mixing matrix
    A = np.array([[0.8, 0.3], 
                  [0.4, 0.7]])
    
    mixed_simple = A @ sources
    print(f"Simple mixed signals shape: {mixed_simple.shape}")
    
    # Save for testing
    stereo_simple = mixed_simple.T  # Transpose for saving
    wavfile.write('./voices/simple_mixed.wav', mixer.sr, stereo_simple.astype(np.float32))
    
    return sources, mixed_simple, A

def test_fastiva_with_simple_mixing():
    """
    Test FastICA with simple, controlled mixing.
    """
    print("\n=== TESTING FASTICA WITH SIMPLE MIXING ===")
    
    sources, mixed_simple, true_A = test_simple_mixing()
    
    # Center and whiten
    mixed_centered = mixed_simple - np.mean(mixed_simple, axis=1, keepdims=True)
    n_samples = mixed_centered.shape[1]
    R = (mixed_centered @ mixed_centered.T) / (n_samples - 1)
    d, Q = np.linalg.eigh(R)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-12))  # Add small epsilon
    V = D_inv_sqrt @ Q.T
    Z = V @ mixed_centered
    
    print(f"Whitened signal shape: {Z.shape}")
    print(f"Whitening check: {np.diag((Z @ Z.T) / (Z.shape[1] - 1))}")
    
    # Apply FastICA
    W, S = fast_ica(Z, n_components=2, fun='tanh', tol=1e-6, max_iter=2000)
    
    print(f"Separated sources shape: {S.shape}")
    
    # Evaluate
    evaluator = BSSMetrics()
    results = evaluator.evaluate_separation(sources, S)
    evaluator.print_results(results)
    
    # Check if we recovered the mixing matrix
    estimated_A_inv = W @ V
    print(f"\nTrue mixing matrix A:\n{true_A}")
    print(f"Estimated unmixing matrix W@V:\n{estimated_A_inv}")
    
    return results

def diagnose_current_implementation():
    """
    Diagnose issues with current room-based implementation.
    """
    print("\n=== DIAGNOSING ROOM-BASED IMPLEMENTATION ===")
    
    # Use best room configuration
    best_config = analyze_sources_and_mixing()
    
    # Create mixing with best config
    mixer = CombineVoices('./voices/music.wav', './voices/talk.wav')
    mixer.mix_sources(
        absorption=best_config["absorption"], 
        max_order=best_config["max_order"]
    )
    mixer.save_mics()
    
    # Load and analyze the mixed signals
    data, sr = sf.read('./voices/mic1.wav')
    print(f"Loaded mixed data: {data.shape}")
    
    # Check if the mixed data actually contains both sources
    min_len = min(len(mixer.s1), len(mixer.s2), data.shape[0])
    
    mic1_signal = data[:min_len, 0]
    mic2_signal = data[:min_len, 1]
    source1_trimmed = mixer.s1[:min_len]
    source2_trimmed = mixer.s2[:min_len]
    
    print(f"Signal lengths - mic1: {len(mic1_signal)}, mic2: {len(mic2_signal)}")
    print(f"Source lengths - s1: {len(source1_trimmed)}, s2: {len(source2_trimmed)}")
    
    # Correlation analysis
    corr_s1_m1 = np.corrcoef(source1_trimmed, mic1_signal)[0, 1]
    corr_s1_m2 = np.corrcoef(source1_trimmed, mic2_signal)[0, 1]
    corr_s2_m1 = np.corrcoef(source2_trimmed, mic1_signal)[0, 1]
    corr_s2_m2 = np.corrcoef(source2_trimmed, mic2_signal)[0, 1]
    
    print(f"\nCorrelation analysis:")
    print(f"Source1 vs Mic1: {corr_s1_m1:.4f}")
    print(f"Source1 vs Mic2: {corr_s1_m2:.4f}")
    print(f"Source2 vs Mic1: {corr_s2_m1:.4f}")
    print(f"Source2 vs Mic2: {corr_s2_m2:.4f}")
    
    # If correlations are too similar, there's insufficient mixing diversity
    mixing_diversity = abs(corr_s1_m1 - corr_s1_m2) + abs(corr_s2_m1 - corr_s2_m2)
    print(f"Mixing diversity: {mixing_diversity:.4f}")
    
    if mixing_diversity < 0.1:
        print("WARNING: Insufficient mixing diversity - sources may not be separable!")
        return False
    
    return True

def main():
    """
    Comprehensive debugging of the BSS implementation.
    """
    print("BSS DEBUGGING SESSION")
    print("=" * 50)
    
    # Step 1: Test with simple mixing
    simple_results = test_fastiva_with_simple_mixing()
    
    if simple_results['Mean_SIR'] < 5:
        print("\n❌ FastICA failing even with simple mixing - algorithm issue!")
        print("Possible issues:")
        print("- FastICA implementation bug")
        print("- Sources not sufficiently non-Gaussian")
        print("- Convergence issues")
        return
    else:
        print(f"\n✅ FastICA works with simple mixing (SIR: {simple_results['Mean_SIR']:.2f} dB)")
    
    # Step 2: Diagnose room-based implementation
    room_ok = diagnose_current_implementation()
    
    if not room_ok:
        print("\n❌ Room simulation not creating sufficient mixing diversity")
        print("Recommendations:")
        print("- Increase microphone spacing")
        print("- Change source positions")
        print("- Adjust room acoustics")
        return
    else:
        print("\n✅ Room simulation creating adequate mixing")
    
    # Step 3: If both work, the issue might be in preprocessing or source matching
    print("\n🔍 Issue likely in:")
    print("- Signal length alignment")
    print("- Preprocessing steps")
    print("- Source matching in evaluation")

if __name__ == "__main__":
    main()