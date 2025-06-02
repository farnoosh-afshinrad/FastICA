import numpy as np


class NonLinearities:
    """Implementation of all non-linearity functions from Table I"""
    
    @staticmethod
    def logcosh(y, a1=1.0):
        """F(y) = (1/a1) log(cosh(a1*y))"""
        return (1/a1) * np.log(np.cosh(a1 * y))
    
    @staticmethod
    def logcosh_deriv(y, a1=1.0):
        """f(y) = tanh(a1*y)"""
        return np.tanh(a1 * y)
    
    @staticmethod
    def logcosh_deriv2(y, a1=1.0):
        """f'(y) = a1[1 - tanh²(a1*y)]"""
        return a1 * (1 - np.tanh(a1 * y)**2)
    
    @staticmethod
    def exp(y):
        """F(y) = -exp(-y²/2)"""
        return -np.exp(-y**2 / 2)
    
    @staticmethod
    def exp_deriv(y):
        """f(y) = y*exp(-y²/2)"""
        return y * np.exp(-y**2 / 2)
    
    @staticmethod
    def exp_deriv2(y):
        """f'(y) = (1 - y²)*exp(-y²/2)"""
        return (1 - y**2) * np.exp(-y**2 / 2)
    
    @staticmethod
    def pow4(y):
        """F(y) = y⁴"""
        return y**4
    
    @staticmethod
    def pow4_deriv(y):
        """f(y) = 4y³"""
        return 4 * y**3
    
    @staticmethod
    def pow4_deriv2(y):
        """f'(y) = 12y²"""
        return 12 * y**2


def compute_negentropy(y, fun='exp'):
    """
    Compute negentropy approximation J(y) ∝ {E[F(y)] - E[F(v)]}²
    where v is Gaussian variable with same variance as y
    
    Now includes proper proportionality constant for better approximation
    """
    # Generate Gaussian variable v with same variance
    v = np.random.randn(len(y))
    v = v * np.std(y)
    
    # Select non-linearity
    if fun == 'logcosh':
        F = NonLinearities.logcosh
        # Proportionality constant for logcosh (empirically determined)
        k = 0.375
    elif fun == 'exp':
        F = NonLinearities.exp
        # Proportionality constant for exp
        k = 0.25
    elif fun == 'pow4':
        F = NonLinearities.pow4
        # Proportionality constant for pow4
        k = 1/12
    else:
        raise ValueError(f"Unknown function: {fun}")
    
    # Compute negentropy approximation with proper constant
    negentropy = k * (np.mean(F(y)) - np.mean(F(v)))**2
    return negentropy


def gram_schmidt_orthogonalize(W, w, i):
    """
    Explicit Gram-Schmidt orthogonalization as mentioned in paper
    Orthogonalize w against rows 0 to i-1 of W
    """
    w_orth = w.copy()
    
    for j in range(i):
        # Project w onto W[j]
        projection = np.dot(w_orth, W[j]) * W[j]
        # Subtract projection
        w_orth = w_orth - projection
    
    # Normalize
    w_orth = w_orth / (np.linalg.norm(w_orth) + 1e-10)
    
    return w_orth


def symmetric_orthogonalization(W):
    """
    Symmetric orthogonalization as alternative to deflation
    W = W * (W^T * W)^(-1/2)
    """
    # Compute W^T * W
    WTW = W @ W.T
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(WTW)
    
    # Compute (W^T * W)^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
    WTW_inv_sqrt = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    
    # Apply symmetric orthogonalization
    W_orth = WTW_inv_sqrt @ W
    
    return W_orth


def fast_ica_newton(Z, n_components, fun='exp', tol=1e-6, max_iter=200, 
                    orthogonalization='deflation'):
    """
    FastICA with Newton iteration refinement as mentioned in paper
    
    Parameters:
    -----------
    orthogonalization: str, 'deflation' or 'symmetric'
        Method for orthogonalization. 'deflation' extracts components one by one,
        'symmetric' extracts all components simultaneously.
    """
    n_features, n_samples = Z.shape
    
    # Select non-linearity functions
    if fun == 'logcosh':
        f = NonLinearities.logcosh_deriv
        f_prime = NonLinearities.logcosh_deriv2
    elif fun == 'exp':
        f = NonLinearities.exp_deriv
        f_prime = NonLinearities.exp_deriv2
    elif fun == 'pow4':
        f = NonLinearities.pow4_deriv
        f_prime = NonLinearities.pow4_deriv2
    else:
        raise ValueError(f"Unknown function: {fun}")
    
    W = np.zeros((n_components, n_features))
    
    if orthogonalization == 'symmetric':
        # Initialize all components at once for symmetric orthogonalization
        for i in range(n_components):
            w = np.random.randn(n_features)
            W[i] = w / np.linalg.norm(w)
        
        print(f"\nExtracting {n_components} components using symmetric orthogonalization")
        
        for iteration in range(max_iter):
            W_old = W.copy()
            
            # Update all components
            for i in range(n_components):
                # Compute projections
                y = np.dot(W[i], Z)
                
                # Newton iteration update
                E_Zf = np.mean(Z * f(y), axis=1)
                E_fprime = np.mean(f_prime(y))
                
                W[i] = E_Zf - E_fprime * W[i]
            
            # Apply symmetric orthogonalization
            W = symmetric_orthogonalization(W)
            
            # Check convergence for all components
            converged = True
            for i in range(n_components):
                # Fixed convergence check as per paper: |⟨w, w_old⟩ - 1| < tolerance
                convergence = abs(abs(np.dot(W[i], W_old[i])) - 1)
                if convergence >= tol:
                    converged = False
                    break
            
            if converged:
                print(f"  All components converged at iteration {iteration+1}")
                break
    
    else:  # deflation
        for i in range(n_components):
            print(f"\nExtracting component {i+1}/{n_components}")
            
            # Random initialization with ||w(0)||₂ = 1
            w = np.random.randn(n_features)
            w = w / np.linalg.norm(w)
            
            converged = False
            
            for iteration in range(max_iter):
                w_old = w.copy()
                
                # Compute projections
                y = np.dot(w, Z)  # shape: (n_samples,)
                
                # Newton iteration update (from paper equations 14-17)
                E_Zf = np.mean(Z * f(y), axis=1)
                E_fprime = np.mean(f_prime(y))
                
                # Newton update
                w = E_Zf - E_fprime * w_old
                
                # Gram-Schmidt orthogonalization against previous components
                if i > 0:
                    w = gram_schmidt_orthogonalize(W, w, i)
                else:
                    # Just normalize for first component
                    w = w / (np.linalg.norm(w) + 1e-10)
                
                # Check for collapse and restart if needed
                norm_w = np.linalg.norm(w)
                if norm_w < 1e-12:
                    print(f"  Component collapsed at iteration {iteration+1}, restarting...")
                    w = np.random.randn(n_features)
                    w = w / np.linalg.norm(w)
                    continue
                
                # Fixed convergence check as per paper: |⟨w, w_old⟩ - 1| < tolerance
                convergence = abs(abs(np.dot(w, w_old)) - 1)
                
                if convergence < tol:
                    converged = True
                    print(f"  Converged at iteration {iteration+1}")
                    
                    # Compute and display negentropy for this component
                    y_final = np.dot(w, Z)
                    neg = compute_negentropy(y_final, fun)
                    print(f"  Negentropy J(y_{i+1}) = {neg:.6f}")
                    break
            
            if not converged:
                print(f"  Warning: Component {i+1} did not converge")
            
            W[i] = w
    
    # Extract sources
    S = W @ Z
    return W, S