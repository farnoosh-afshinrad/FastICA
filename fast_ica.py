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

def _gram_schmidt_orthogonalize(W: np.ndarray, w: np.ndarray, n_prev: int) -> np.ndarray:
    """Classical Gram–Schmidt: project *w* onto orthogonal complement of rows < n_prev."""
    for j in range(n_prev):
        w -= np.dot(w, W[j]) * W[j]
    return w


def fast_ica_newton(
    Z: np.ndarray,
    n_components: int,
    *,
    fun: str = "exp",
    tol: float = 1e-6,
    max_iter: int = 200,
    # --------------------------------------------------------------------
    # Weight‑initialisation parameters -----------------------------------
    # --------------------------------------------------------------------
    apply_weight_init: bool = False,
    wi_sd_iters: int = 5,
    wi_sd_step: float = 0.3,
    random_state: int =42,
):
    """FastICA (deflation) with Newton refinement **and optional weight initialisation**.

    Parameters
    ----------
    Z : ndarray, shape (n_features, n_samples)
        Whitened, zero‑mean observation matrix *X̂*.
    n_components : int
        Number of sources to extract.  Must be ≤ n_features (after whitening).
    fun : {"logcosh", "exp", "pow4"}, default="exp"
        Non‑linearity *g*.
    tol : float, default=1e-6
        Convergence threshold (|⟨w, w_old⟩ − 1| < tol).
    max_iter : int, default=200
        Maximum Newton iterations per component.
        version can be added by using *_inv_sqrtm*.
    apply_weight_init : bool, default=False
        Activate the **Weight‑Initialised ICA** procedure:
            1. build a *matrix* W_init ∈ ℝ^{n_components × n_features} with entries
               ~ N(0, 1);
            2. subtract its mean  (W_init ← W_init − mean(W_init)); and
            3. run *wi_sd_iters* steps of *steepest descent*  W ← W − λ E[X g(WX)].
        The *i‑th* row of the resulting W_init is then used as the initial "w"
        for component *i* in the deflation loop.
    wi_sd_iters : int, default=5
        Number of steepest‑descent pre‑iterations.  Ignored if
        ``apply_weight_init`` is False.
    wi_sd_step : float, default=0.3
        Step size λ for the steepest‑descent pre‑iterations.
    random_state : int or None, default=None
        Deterministic seeding.

    Returns
    -------
    W : ndarray, shape (n_components, n_features)
        Estimated unmixing matrix.
    S : ndarray, shape (n_components, n_samples)
        Estimated source signals (S = W Z).
    """

    # --------------------------------------------------------------------
    # Reproducibility & basic checks -------------------------------------
    # --------------------------------------------------------------------
    rng = np.random.default_rng(random_state)
    n_features, n_samples = Z.shape
    if n_components > n_features:
        raise ValueError("n_components must be ≤ n_features (after whitening).")

    # --------------------------------------------------------------------
    # Select non‑linearity and derivatives -------------------------------
    # --------------------------------------------------------------------
    if fun == "logcosh":
        g = NonLinearities.logcosh
        g_deriv = NonLinearities.logcosh_deriv
        g_second = NonLinearities.logcosh_deriv2
    elif fun == "exp":
        g = NonLinearities.exp
        g_deriv = NonLinearities.exp_deriv
        g_second = NonLinearities.exp_deriv2
    elif fun == "pow4":
        g = NonLinearities.pow4
        g_deriv = NonLinearities.pow4_deriv
        g_second = NonLinearities.pow4_deriv2
    else:
        raise ValueError(f"Unknown fun={fun!r}.")

    # --------------------------------------------------------------------
    # Optional *matrix‑level* weight initialisation -----------------------
    # --------------------------------------------------------------------
    if apply_weight_init:
        # Step‑1: random W_init  --------------------------------------
        W_init = rng.standard_normal((n_components, n_features))
        # Step‑2: subtract mean of *entire* matrix  ------------------
        W_init -= W_init.mean()
        # Step‑3: *wi_sd_iters* steepest‑descent passes  -------------
        for _ in range(wi_sd_iters):
            Y = W_init @ Z                            # shape (m, n)
            # E[X g(WX)]  computed row‑wise -------------------------
            grad = np.empty_like(W_init)
            for j in range(n_components):
                grad[j] = np.mean(Z * g(Y[j]), axis=1)
            # gradient step -----------------------------------------
            W_init -= wi_sd_step * grad
            # row‑wise renormalisation to avoid explosion ------------
            W_init /= np.linalg.norm(W_init, axis=1, keepdims=True) + 1e-12
    else:
        W_init = rng.standard_normal((n_components, n_features))

    # Rows of W_init will serve as *initial* w's for components 0…m‑1

    # --------------------------------------------------------------------
    # Storage for final unmixing matrix ----------------------------------
    # --------------------------------------------------------------------
    W_est = np.zeros((n_components, n_features))

    # --------------------------------------------------------------------
    # Deflation loop ------------------------------------------------------
    # --------------------------------------------------------------------
    for i in range(n_components):
        w = W_init[i].copy()
        w /= np.linalg.norm(w) + 1e-12  # ensure on unit sphere

        for _ in range(max_iter):
            w_old = w.copy()

            y = w @ Z  # projections  (n_samples,)
            E_Zg_deriv = np.mean(Z * g_deriv(y), axis=1)  # (n_features,)
            E_g_second = np.mean(g_second(y))             # scalar
            w = E_Zg_deriv - E_g_second * w_old           # Newton / quasi‑Newton

            # Orthogonalise against already‑found vectors ------------
            if i > 0:
                w = _gram_schmidt_orthogonalize(W_est, w, i)

            # Always renormalise (algorithm step‑f) ------------------
            w /= np.linalg.norm(w) + 1e-12

            # Convergence check --------------------------------------
            if abs(abs(np.dot(w, w_old)) - 1.0) < tol:
                break
        else:
            print(f"[Warning] Component {i+1} did not converge within {max_iter} iterations.")

        # Fix sign to deterministic convention -----------------------
        w *= np.sign(w[np.argmax(np.abs(w))])

        W_est[i] = w

    # --------------------------------------------------------------------
    # Done – return sources ----------------------------------------------
    # --------------------------------------------------------------------
    S = W_est @ Z
    return W_est, S
