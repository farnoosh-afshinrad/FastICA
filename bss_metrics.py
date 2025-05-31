import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate

class BSSMetrics:
    """
    Comprehensive evaluation metrics for Blind Source Separation (BSS)
    following standard BSS evaluation methodologies.
    """
    
    def __init__(self):
        pass
    
    def _find_best_permutation(self, reference_sources, estimated_sources):
        """
        Find the best permutation and scaling to match estimated sources 
        to reference sources using Hungarian algorithm.
        
        Args:
            reference_sources: Original sources (n_sources, n_samples)
            estimated_sources: Separated sources (n_sources, n_samples)
            
        Returns:
            perm_matrix: Permutation matrix
            scale_factors: Scaling factors
        """
        n_sources = reference_sources.shape[0]
        
        # Compute correlation matrix between all pairs
        corr_matrix = np.zeros((n_sources, n_sources))
        
        for i in range(n_sources):
            for j in range(n_sources):
                # Normalize signals for correlation calculation
                ref_norm = reference_sources[i] / (np.linalg.norm(reference_sources[i]) + 1e-12)
                est_norm = estimated_sources[j] / (np.linalg.norm(estimated_sources[j]) + 1e-12)
                
                # Use maximum cross-correlation
                cross_corr = correlate(ref_norm, est_norm, mode='full')
                corr_matrix[i, j] = np.max(np.abs(cross_corr))
        
        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-corr_matrix)  # Negative for maximization
        
        # Create permutation matrix
        perm_matrix = np.zeros((n_sources, n_sources))
        perm_matrix[row_ind, col_ind] = 1
        
        # Calculate scaling factors
        scale_factors = np.zeros(n_sources)
        reordered_estimated = perm_matrix @ estimated_sources
        
        for i in range(n_sources):
            if np.var(reordered_estimated[i]) > 1e-12:
                scale_factors[i] = np.dot(reference_sources[i], reordered_estimated[i]) / \
                                 (np.dot(reordered_estimated[i], reordered_estimated[i]) + 1e-12)
            else:
                scale_factors[i] = 1.0
                
        return perm_matrix, scale_factors
    
    def signal_to_interference_ratio(self, reference_sources, estimated_sources):
        """
        Calculate Signal-to-Interference Ratio (SIR) in dB.
        
        SIR measures how well the algorithm suppresses interference from other sources.
        Higher values indicate better separation.
        """
        perm_matrix, scale_factors = self._find_best_permutation(reference_sources, estimated_sources)
        reordered_estimated = perm_matrix @ estimated_sources
        
        # Apply scaling
        for i in range(len(scale_factors)):
            reordered_estimated[i] *= scale_factors[i]
        
        n_sources = reference_sources.shape[0]
        sir_values = np.zeros(n_sources)
        
        for i in range(n_sources):
            # Target signal power
            target_power = np.mean(reference_sources[i] ** 2)
            
            # Interference: difference between estimated and reference
            interference = reordered_estimated[i] - reference_sources[i]
            interference_power = np.mean(interference ** 2)
            
            # SIR in dB
            if interference_power > 1e-12:
                sir_values[i] = 10 * np.log10(target_power / interference_power)
            else:
                sir_values[i] = 100  # Very high SIR if no interference
                
        return sir_values
    
    def signal_to_artifacts_ratio(self, reference_sources, estimated_sources):
        """
        Calculate Signal-to-Artifacts Ratio (SAR) in dB.
        
        SAR measures the amount of artifacts introduced by the separation algorithm.
        Higher values indicate fewer artifacts.
        """
        perm_matrix, scale_factors = self._find_best_permutation(reference_sources, estimated_sources)
        reordered_estimated = perm_matrix @ estimated_sources
        
        # Apply scaling
        for i in range(len(scale_factors)):
            reordered_estimated[i] *= scale_factors[i]
        
        n_sources = reference_sources.shape[0]
        sar_values = np.zeros(n_sources)
        
        for i in range(n_sources):
            # Artifacts: high-frequency content difference
            ref_energy = np.mean(reference_sources[i] ** 2)
            error = reordered_estimated[i] - reference_sources[i]
            artifact_energy = np.mean(error ** 2)
            
            if artifact_energy > 1e-12:
                sar_values[i] = 10 * np.log10(ref_energy / artifact_energy)
            else:
                sar_values[i] = 100
                
        return sar_values
    
    def signal_to_distortion_ratio(self, reference_sources, estimated_sources):
        """
        Calculate Signal-to-Distortion Ratio (SDR) in dB.
        
        SDR is an overall measure combining interference and artifacts.
        Higher values indicate better overall separation quality.
        """
        perm_matrix, scale_factors = self._find_best_permutation(reference_sources, estimated_sources)
        reordered_estimated = perm_matrix @ estimated_sources
        
        # Apply scaling
        for i in range(len(scale_factors)):
            reordered_estimated[i] *= scale_factors[i]
        
        n_sources = reference_sources.shape[0]
        sdr_values = np.zeros(n_sources)
        
        for i in range(n_sources):
            signal_power = np.mean(reference_sources[i] ** 2)
            error = reordered_estimated[i] - reference_sources[i]
            distortion_power = np.mean(error ** 2)
            
            if distortion_power > 1e-12:
                sdr_values[i] = 10 * np.log10(signal_power / distortion_power)
            else:
                sdr_values[i] = 100
                
        return sdr_values
    
    def normalized_mean_square_error(self, reference_sources, estimated_sources):
        """
        Calculate Normalized Mean Square Error (NMSE).
        
        Lower values indicate better separation quality.
        """
        perm_matrix, scale_factors = self._find_best_permutation(reference_sources, estimated_sources)
        reordered_estimated = perm_matrix @ estimated_sources
        
        # Apply scaling
        for i in range(len(scale_factors)):
            reordered_estimated[i] *= scale_factors[i]
        
        n_sources = reference_sources.shape[0]
        nmse_values = np.zeros(n_sources)
        
        for i in range(n_sources):
            mse = np.mean((reference_sources[i] - reordered_estimated[i]) ** 2)
            signal_power = np.mean(reference_sources[i] ** 2)
            nmse_values[i] = mse / (signal_power + 1e-12)
            
        return nmse_values
    
    def correlation_coefficient(self, reference_sources, estimated_sources):
        """
        Calculate correlation coefficient between reference and estimated sources.
        
        Values closer to 1 (or -1) indicate better separation quality.
        """
        perm_matrix, scale_factors = self._find_best_permutation(reference_sources, estimated_sources)
        reordered_estimated = perm_matrix @ estimated_sources
        
        n_sources = reference_sources.shape[0]
        corr_coeffs = np.zeros(n_sources)
        
        for i in range(n_sources):
            ref_centered = reference_sources[i] - np.mean(reference_sources[i])
            est_centered = reordered_estimated[i] - np.mean(reordered_estimated[i])
            
            numerator = np.sum(ref_centered * est_centered)
            denominator = np.sqrt(np.sum(ref_centered**2) * np.sum(est_centered**2))
            
            if denominator > 1e-12:
                corr_coeffs[i] = abs(numerator / denominator)
            else:
                corr_coeffs[i] = 0
                
        return corr_coeffs
    
    def amari_error(self, true_mixing_matrix, estimated_unmixing_matrix):
        """
        Calculate Amari Error for mixing matrix estimation quality.
        
        Lower values indicate better matrix estimation.
        """
        if true_mixing_matrix is None or estimated_unmixing_matrix is None:
            return None
            
        # Product of true mixing and estimated unmixing
        P = estimated_unmixing_matrix @ true_mixing_matrix
        n = P.shape[0]
        
        # Row-wise normalization
        P_row = P / (np.max(np.abs(P), axis=1, keepdims=True) + 1e-12)
        # Column-wise normalization  
        P_col = P / (np.max(np.abs(P), axis=0, keepdims=True) + 1e-12)
        
        # Amari error calculation
        row_sum = np.sum(np.sum(np.abs(P_row), axis=1) - 1)
        col_sum = np.sum(np.sum(np.abs(P_col), axis=0) - 1)
        
        amari_err = (row_sum + col_sum) / (2 * n * (n - 1))
        return amari_err
    
    def evaluate_separation(self, reference_sources, estimated_sources, true_mixing_matrix=None):
        """
        Comprehensive evaluation of BSS performance.
        
        Args:
            reference_sources: Original source signals (n_sources, n_samples)
            estimated_sources: Separated signals (n_sources, n_samples)
            true_mixing_matrix: True mixing matrix (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Primary BSS metrics
        results['SIR'] = self.signal_to_interference_ratio(reference_sources, estimated_sources)
        results['SAR'] = self.signal_to_artifacts_ratio(reference_sources, estimated_sources)
        results['SDR'] = self.signal_to_distortion_ratio(reference_sources, estimated_sources)
        
        # Additional metrics
        results['NMSE'] = self.normalized_mean_square_error(reference_sources, estimated_sources)
        results['Correlation'] = self.correlation_coefficient(reference_sources, estimated_sources)
        
        # Matrix estimation quality (if available)
        if true_mixing_matrix is not None:
            # This would need the estimated unmixing matrix from your ICA
            # results['Amari_Error'] = self.amari_error(true_mixing_matrix, estimated_unmixing_matrix)
            pass
        
        # Summary statistics
        results['Mean_SIR'] = np.mean(results['SIR'])
        results['Mean_SAR'] = np.mean(results['SAR'])
        results['Mean_SDR'] = np.mean(results['SDR'])
        results['Mean_NMSE'] = np.mean(results['NMSE'])
        results['Mean_Correlation'] = np.mean(results['Correlation'])
        
        return results
    
    def print_results(self, results):
        """
        Print evaluation results in a formatted way.
        """
        print("\n" + "="*60)
        print("BLIND SOURCE SEPARATION EVALUATION RESULTS")
        print("="*60)
        
        metrics = ['SIR', 'SAR', 'SDR']
        for metric in metrics:
            if metric in results:
                print(f"\n{metric} (dB):")
                for i, val in enumerate(results[metric]):
                    print(f"  Source {i+1}: {val:.2f}")
                print(f"  Mean: {results[f'Mean_{metric}']:.2f}")
        
        print(f"\nNormalized MSE:")
        for i, val in enumerate(results['NMSE']):
            print(f"  Source {i+1}: {val:.4f}")
        print(f"  Mean: {results['Mean_NMSE']:.4f}")
        
        print(f"\nCorrelation Coefficient:")
        for i, val in enumerate(results['Correlation']):
            print(f"  Source {i+1}: {val:.4f}")
        print(f"  Mean: {results['Mean_Correlation']:.4f}")
        
        if 'Amari_Error' in results:
            print(f"\nAmari Error: {results['Amari_Error']:.4f}")
        
        print("\n" + "="*60)
        print("INTERPRETATION GUIDE:")
        print("SIR: Signal-to-Interference Ratio (higher is better, >10dB is good)")
        print("SAR: Signal-to-Artifacts Ratio (higher is better, >10dB is good)")
        print("SDR: Signal-to-Distortion Ratio (higher is better, >5dB is acceptable)")
        print("NMSE: Normalized Mean Square Error (lower is better, <0.1 is good)")
        print("Correlation: Correlation coefficient (closer to 1 is better, >0.8 is good)")
        print("="*60)