import concurrent
import concurrent.futures
import multiprocessing
import time

import numpy as np
import ot
from tqdm import tqdm

from src.utils_solver import (
    Lmatrix2paths,
    adapted_empirical_measure,
    adapted_wasserstein_squared,
    nested,
    plot_V,
    quantization,
)


def do_sth(second):
    print(f"Sleep {second} second")
    time.sleep(second)
    return f"Done Sleeping {second} second..."


def nested_mp(mu_x, nu_y, v2q_x, v2q_y, q2v, markovian=False, verbose=True):
    T = len(mu_x)
    square_cost_matrix = (q2v[None, :] - q2v[None, :].T) ** 2

    V = [np.zeros([len(v2q_x[t]), len(v2q_y[t])]) for t in range(T)]
    if verbose:
        print("Nested backward induction .......")
    for t in range(T - 1, -1, -1):
        print(len(mu_x[t]), len(nu_y[t]))
        print(np.unique(np.array(mu_x[t].keys())))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(do_sth, range(3))

        tqdm_bar = tqdm(mu_x[t].items()) if verbose else mu_x[t].items()
        for k1, v1 in tqdm_bar:
            if verbose:
                tqdm_bar.set_description(f"Timestep {t}")
            for k2, v2 in nu_y[t].items():
                # list of probability of conditional distribution mu_x
                w1 = list(v1.values())
                w1 = np.array(w1) / sum(w1)
                # list of probability of conditional distribution nu_y
                w2 = list(v2.values())
                w2 = np.array(w2) / sum(w2)
                # list of quantized values of conditional distribution mu_x (nu_y)
                q1 = list(v1.keys())
                q2 = list(v2.keys())
                # square cost of the values indexed by quantized values: |q2v[q1] - q2v[q2]|^2
                cost = square_cost_matrix[np.ix_(q1, q2)]

                # At T-1: add V[T] = 0, otherwise add the V[t+1] already computed
                if t < T - 1:
                    if (
                        markovian
                    ):  # If markovian, for condition path (k1,q), only the last value q matters, and V[t+1] is indexed by the time re-quantization of q
                        q1s = [v2q_x[t + 1][q] for q in v1.keys()]
                        q2s = [v2q_y[t + 1][q] for q in v2.keys()]
                    else:  # If non-markovian, for condition path (k1,q), the V[t+1] is indexed by the time re-quantization of tuple (k1,q)
                        q1s = [v2q_x[t + 1][k1 + (q,)] for q in v1.keys()]
                        q2s = [v2q_y[t + 1][k2 + (q,)] for q in v2.keys()]
                    cost += V[t + 1][np.ix_(q1s, q2s)]

                # solve the OT problem with cost |x_t-y_t|^2 + V_{t+1}(x_{1:t},y_{1:t})
                V[t][v2q_x[t][k1], v2q_y[t][k2]] = ot.emd2(w1, w2, cost)

    AW_2square = V[0][0, 0]
    return AW_2square, V


if __name__ == "__main__":
    n_sample = 100
    normalize = False

    L = np.array([[1, 0, 0], [2, 4, 0], [3, 2, 1]])
    X, A = Lmatrix2paths(L, n_sample, seed=1, verbose=False)
    M = np.array([[1, 0, 0], [2, 3, 0], [3, 1, 2]])
    Y, B = Lmatrix2paths(M, n_sample, seed=1, verbose=False)

    adaptedX = adapted_empirical_measure(X, delta_n=0.1)
    adaptedY = adapted_empirical_measure(Y, delta_n=0.1)

    q2v, v2q, mu_x, nu_y, q2v_x, v2q_x, q2v_y, v2q_y = quantization(
        adaptedX, adaptedY, markovian=False, verbose=False
    )

    start_time = time.perf_counter()
    AW_2square, V = nested_mp(mu_x, nu_y, v2q_x, v2q_y, q2v, markovian=False)
    end_time = time.perf_counter()

    dist_bench = adapted_wasserstein_squared(A, B)
    print("Theoretical AW_2^2: ", dist_bench)
    print("Numerical AW_2^2: ", AW_2square)
    print("Elapsed time (Adapted OT): {:.4f} seconds".format(end_time - start_time))

    # secs = [5, 4, 3, 2, 1]

    # start = time.perf_counter()
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = executor.map(do_sth, secs)
    # finish = time.perf_counter()
    # print(f"Finished in {round(finish - start,2)} seconds")
