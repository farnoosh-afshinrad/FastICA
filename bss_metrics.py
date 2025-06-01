import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate

class BSSMetrics:
    """
    Blind Source Separation evaluation metrics
    Including SIR, SDR, and correlation metrics
    """
    
    @staticmethod
    def compute_projection(s_est, s_true):
        """Compute projection of estimated source onto true source"""
        return np.dot(s_est, s_true) / np.dot(s_true, s_true) * s_true
    
    @staticmethod
    def sir(s_true, s_est):
        """Signal-to-Interference Ratio"""
        # Project estimated source onto true source
        alpha = np.dot(s_est, s_true) / (np.dot(s_true, s_true) + 1e-10)
        s_target = alpha * s_true
        e_interf = s_est - s_target
        
        sir = 10 * np.log10((np.sum(s_target**2) + 1e-10) / (np.sum(e_interf**2) + 1e-10))
        return sir
    
    @staticmethod
    def sdr(s_true, s_est):
        """Signal-to-Distortion Ratio"""
        # Scale estimated source to match true source
        alpha = np.dot(s_est, s_true) / (np.dot(s_true, s_true) + 1e-10)
        s_scaled = alpha * s_true
        distortion = s_est - s_scaled
        
        sdr = 10 * np.log10((np.sum(s_scaled**2) + 1e-10) / (np.sum(distortion**2) + 1e-10))
        return sdr
    
    @staticmethod
    def find_best_permutation(S_true, S_est):
        """Find best permutation of estimated sources to match true sources"""
        n_sources = S_true.shape[0]
        
        # Compute correlation matrix
        corr_matrix = np.abs(np.corrcoef(S_true, S_est)[:n_sources, n_sources:])
        
        # Find best permutation (greedy approach)
        permutation = []
        used = set()
        
        for i in range(n_sources):
            best_j = -1
            best_corr = -1
            
            for j in range(n_sources):
                if j not in used and corr_matrix[i, j] > best_corr:
                    best_corr = corr_matrix[i, j]
                    best_j = j
            
            permutation.append(best_j)
            used.add(best_j)
        
        return permutation
    
    @staticmethod
    def evaluate_separation(S_true, S_est):
        """Comprehensive evaluation of source separation quality"""
        n_sources = S_true.shape[0]
        
        # Find best permutation
        perm = BSSMetrics.find_best_permutation(S_true, S_est)
        
        # Reorder estimated sources
        S_est_perm = S_est[perm]
        
        results = {
            'permutation': perm,
            'SIR': [],
            'SDR': [],
            'correlation': []
        }
        
        for i in range(n_sources):
            # Normalize both signals before comparison to handle scale ambiguity
            s_true_norm = S_true[i] / (np.std(S_true[i]) + 1e-10)
            s_est_norm = S_est_perm[i] / (np.std(S_est_perm[i]) + 1e-10)
            
            # Find optimal scaling factor
            scale = np.dot(s_est_norm, s_true_norm) / (np.dot(s_est_norm, s_est_norm) + 1e-10)
            
            # Apply optimal scaling
            s_est_scaled = s_est_norm * scale
            
            # Compute metrics
            sir = BSSMetrics.sir(s_true_norm, s_est_scaled)
            sdr = BSSMetrics.sdr(s_true_norm, s_est_scaled)
            corr = np.abs(np.corrcoef(S_true[i], S_est_perm[i])[0, 1])
            
            results['SIR'].append(sir)
            results['SDR'].append(sdr)
            results['correlation'].append(corr)
        
        # Add mean values
        results['Mean_SIR'] = np.mean(results['SIR'])
        results['Mean_SDR'] = np.mean(results['SDR'])
        results['Mean_Correlation'] = np.mean(results['correlation'])
        
        return results
    
    @staticmethod
    def print_results(results):
        """Pretty print evaluation results"""
        print("\n" + "="*50)
        print("BSS EVALUATION RESULTS")
        print("="*50)
        print(f"Best permutation: {results['permutation']}")
        print("\nPer-source metrics:")
        
        n_sources = len(results['SIR'])
        for i in range(n_sources):
            print(f"\nSource {i+1}:")
            print(f"  SIR: {results['SIR'][i]:.2f} dB")
            print(f"  SDR: {results['SDR'][i]:.2f} dB")
            print(f"  Correlation: {results['correlation'][i]:.4f}")
        
        print("\nOverall performance:")
        print(f"  Mean SIR: {results['Mean_SIR']:.2f} dB")
        print(f"  Mean SDR: {results['Mean_SDR']:.2f} dB")
        print(f"  Mean Correlation: {results['Mean_Correlation']:.4f}")
        print("="*50)