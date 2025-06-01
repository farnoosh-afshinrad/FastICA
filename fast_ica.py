import numpy as np


# # --- non-linearities per Table I (negentropy approximations) ---
# def _g_pow3(x):         return x ** 3


# def _g_pow3_der(x):     return 3 * x ** 2


# def _g_tanh(x, a=1.0):   return np.tanh(a * x)


# def _g_tanh_der(x, a=1.0):
#     t = np.tanh(a * x)
#     return a * (1 - t ** 2)


# def _g_gauss(x):        return x * np.exp(-x ** 2 / 2)


# def _g_gauss_der(x):    return (1 - x ** 2) * np.exp(-x ** 2 / 2)




# def fast_ica(Z, n_components, fun='pow3', tol=1e-6, max_iter=2000):
#     """
#     Negentropy-based FastICA with restart-on-collapse.
#     Z: whitened data, shape (n_features, n_samples)
#     n_components: how many ICs to extract
#     fun: 'pow3' | 'tanh' | 'gauss'
#     """
#     n_features, n_samples = Z.shape

#     # bind g and its derivative
#     if fun == 'pow3':
#         g, g_der = _g_pow3, _g_pow3_der
#     elif fun == 'tanh':
#         g   = lambda x: _g_tanh(x, a=1.0)
#         g_der = lambda x: _g_tanh_der(x, a=1.0)
#     elif fun == 'gauss':
#         g, g_der = _g_gauss, _g_gauss_der
#     else:
#         raise ValueError(f"Unknown fun '{fun}'")

#     W = np.zeros((n_components, n_features), dtype=float)

#     for i in range(n_components):
#         # random unit start
#         w = np.random.randn(n_features)
#         w /= np.linalg.norm(w)

#         for _ in range(max_iter):
#             w_old = w.copy()

#             # 1) projection
#             proj = w @ Z                     # shape (n_samples,)
#             # 2) non-linearity
#             g_proj     = g(proj)
#             g_der_proj = g_der(proj)
#             # 3) fixed-point update
#             w = (Z @ g_proj) / n_samples - np.mean(g_der_proj) * w_old

#             # 4) deflation / orthogonalize
#             if i > 0:
#                 # subtract projection onto previously found W[:i]
#                 coeffs = W[:i] @ w          # shape (i,)
#                 w -= W[:i].T @ coeffs       # shape (n_features,)

#             # 5) guard against collapse
#             norm_w = np.linalg.norm(w)
#             if norm_w < 1e-12:
#                 # restart this component
#                 w = np.random.randn(n_features)
#                 w /= np.linalg.norm(w)
#                 continue

#             # 6) normalize
#             w /= norm_w

#             # 7) check convergence
#             if abs(abs(w @ w_old) - 1) < tol:
#                 break

#         W[i, :] = w

#     # recover sources
#     S = W @ Z
#     return W, S

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
    """
    # Generate Gaussian variable v with same variance
    v = np.random.randn(len(y))
    v = v * np.std(y)
    
    # Select non-linearity
    if fun == 'logcosh':
        F = NonLinearities.logcosh
    elif fun == 'exp':
        F = NonLinearities.exp
    elif fun == 'pow4':
        F = NonLinearities.pow4
    else:
        raise ValueError(f"Unknown function: {fun}")
    
    # Compute negentropy approximation
    negentropy = (np.mean(F(y)) - np.mean(F(v)))**2
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


def fast_ica_newton(Z, n_components, fun='exp', tol=1e-6, max_iter=200):
    """
    FastICA with Newton iteration refinement as mentioned in paper
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
    
    for i in range(n_components):
        print(f"\nExtracting component {i+1}/{n_components}")
        
        # Random initialization with ||u(0)||₂ = 1
        w = np.random.randn(n_features)
        w = w / np.linalg.norm(w)
        
        converged = False
        
        for iteration in range(max_iter):
            w_old = w.copy()
            
            # Compute projections
            y = np.dot(w, Z)  # shape: (n_samples,)
            
            # Newton iteration update (from paper equations 14-17)
            # u_i(k+1) = E[Zf(y_i)] - u_i(k)E[f'(y_i)]
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
            
            # Check convergence
            convergence = 1 - abs(np.dot(w, w_old))
            
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
