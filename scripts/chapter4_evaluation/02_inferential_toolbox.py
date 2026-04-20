import numpy as np
from scipy import stats as sp_stats


def select_block_length(x, max_lag=40):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    var0 = np.dot(x, x) / n

    if var0 == 0:
        return 5

    bound = 1.96 / np.sqrt(n)

    for lag in range(1, min(max_lag, n // 4)):
        r = np.dot(x[:-lag], x[lag:]) / ((n - lag) * var0)
        if abs(r) < bound:
            return max(3, min(20, lag))

    return 10


def block_bootstrap(x, statistic, n_boot=10000, block_len=None, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)

    if block_len is None:
        key = x if x.ndim == 1 else x[:, 0]
        block_len = select_block_length(key)

    p = 1.0 / block_len
    boot_stats = np.empty(n_boot)

    for _ in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = 0

        while i < n:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(p))
            length = min(length, n - i)
            idx[i:i + length] = (start + np.arange(length)) % n
            i += length

        sample = x[idx]
        boot_stats[_] = statistic(sample)

    point = statistic(x)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_stats, [alpha, 1 - alpha])

    return point, (float(lo), float(hi)), boot_stats


def diebold_mariano(e1, e2, h=1, loss="abs"):
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)

    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sq":
        d = e1 ** 2 - e2 ** 2
    else:
        raise ValueError(loss)

    T = len(d)
    d_bar = np.mean(d)

    gamma_0 = np.var(d, ddof=0)
    gamma = [gamma_0]

    for k in range(1, h):
        gk = np.mean((d[:-k] - d_bar) * (d[k:] - d_bar))
        gamma.append(gk)

    var_d = gamma[0] + 2 * sum(gamma[1:])
    var_d = max(var_d, 1e-12) / T

    dm = d_bar / np.sqrt(var_d)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * hln_factor
    p = 2 * (1 - sp_stats.t.cdf(abs(dm_hln), df=T - 1))

    return float(dm_hln), float(p)


print("DM test added")