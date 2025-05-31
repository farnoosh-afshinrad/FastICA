import numpy as np


# --- non-linearities per Table I (negentropy approximations) ---
def _g_pow3(x):         return x ** 3


def _g_pow3_der(x):     return 3 * x ** 2


def _g_tanh(x, a=1.0):   return np.tanh(a * x)


def _g_tanh_der(x, a=1.0):
    t = np.tanh(a * x)
    return a * (1 - t ** 2)


def _g_gauss(x):        return x * np.exp(-x ** 2 / 2)


def _g_gauss_der(x):    return (1 - x ** 2) * np.exp(-x ** 2 / 2)




def fast_ica(Z, n_components, fun='pow3', tol=1e-6, max_iter=2000):
    """
    Negentropy-based FastICA with restart-on-collapse.
    Z: whitened data, shape (n_features, n_samples)
    n_components: how many ICs to extract
    fun: 'pow3' | 'tanh' | 'gauss'
    """
    n_features, n_samples = Z.shape

    # bind g and its derivative
    if fun == 'pow3':
        g, g_der = _g_pow3, _g_pow3_der
    elif fun == 'tanh':
        g   = lambda x: _g_tanh(x, a=1.0)
        g_der = lambda x: _g_tanh_der(x, a=1.0)
    elif fun == 'gauss':
        g, g_der = _g_gauss, _g_gauss_der
    else:
        raise ValueError(f"Unknown fun '{fun}'")

    W = np.zeros((n_components, n_features), dtype=float)

    for i in range(n_components):
        # random unit start
        w = np.random.randn(n_features)
        w /= np.linalg.norm(w)

        for _ in range(max_iter):
            w_old = w.copy()

            # 1) projection
            proj = w @ Z                     # shape (n_samples,)
            # 2) non-linearity
            g_proj     = g(proj)
            g_der_proj = g_der(proj)
            # 3) fixed-point update
            w = (Z @ g_proj) / n_samples - np.mean(g_der_proj) * w_old

            # 4) deflation / orthogonalize
            if i > 0:
                # subtract projection onto previously found W[:i]
                coeffs = W[:i] @ w          # shape (i,)
                w -= W[:i].T @ coeffs       # shape (n_features,)

            # 5) guard against collapse
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-12:
                # restart this component
                w = np.random.randn(n_features)
                w /= np.linalg.norm(w)
                continue

            # 6) normalize
            w /= norm_w

            # 7) check convergence
            if abs(abs(w @ w_old) - 1) < tol:
                break

        W[i, :] = w

    # recover sources
    S = W @ Z
    return W, S
