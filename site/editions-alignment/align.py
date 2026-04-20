import numpy as np


def score(sim: float, gap: int, penalty_weight: float = 0.1) -> float:
    return sim - (penalty_weight * gap)


def align(
    sim_matrix: np.ndarray,
    penalty_weight: float = 0.1,
    null_threshold: float = 0.2,
    band: int = 10,
) -> list[tuple[int | None, int | None]]:
    M, N = sim_matrix.shape
    dp = np.full((M + 1, N + 1), -np.inf)
    dp[0, 0] = 0.0
    back = {}

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            best_score = -np.inf
            best_prev = None

            pi_min = max(0, i - band - 1)
            pj_min = max(0, j - band - 1)
            for pi in range(pi_min, i):
                for pj in range(pj_min, j):
                    if dp[pi, pj] == -np.inf:
                        continue
                    gap_a = (i - 1) - pi
                    gap_b = (j - 1) - pj
                    gap = max(gap_a, gap_b) - 1
                    gap = max(gap, 0)

                    sim = sim_matrix[i - 1, j - 1]
                    if sim < null_threshold:
                        continue

                    candidate = dp[pi, pj] + score(sim, gap, penalty_weight)
                    if candidate > best_score:
                        best_score = candidate
                        best_prev = (pi, pj)

            if dp[i - 1, j] > best_score:
                best_score = dp[i - 1, j]
                best_prev = (i - 1, j)

            if dp[i, j - 1] > best_score:
                best_score = dp[i, j - 1]
                best_prev = (i, j - 1)

            dp[i, j] = best_score
            back[(i, j)] = best_prev

    alignment = []
    i, j = M, N
    while i > 0 or j > 0:
        prev = back.get((i, j))
        if prev is None:
            break
        pi, pj = prev
        if pi < i and pj < j:
            alignment.append((i - 1, j - 1))
        elif pi < i:
            alignment.append((i - 1, None))
        else:
            alignment.append((None, j - 1))
        i, j = pi, pj

    alignment.reverse()
    return alignment
