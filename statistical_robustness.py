import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils_solver import (
    Lmatrix2paths,
    adapted_empirical_measure,
    adapted_wasserstein_squared,
    nested,
    quantization,
)


def compute_AW2squares(L, M, markovian=False):
    AW_2squares = np.zeros([n, n_trial])
    for i, n_sample in enumerate(n_sample_list):
        print(f"Number of sample: {n_sample}")
        tqdm_bar = tqdm(np.arange(n_trial))
        for j in tqdm_bar:
            X, A = Lmatrix2paths(L, n_sample, seed=j, verbose=False)
            Y, B = Lmatrix2paths(M, n_sample, seed=j, verbose=False)
            delta_n = 1 / np.sqrt(n_sample)
            adaptedX = adapted_empirical_measure(X, delta_n=delta_n)
            adaptedY = adapted_empirical_measure(Y, delta_n=delta_n)
            q2v, v2q, mu_x, nu_y, q2v_x, v2q_x, q2v_y, v2q_y = quantization(
                adaptedX, adaptedY, markovian=markovian, verbose=False
            )
            AW_2square, V = nested(
                mu_x, nu_y, v2q_x, v2q_y, q2v, markovian=markovian, verbose=False
            )
            AW_2squares[i, j] = AW_2square

    AW_2bench = adapted_wasserstein_squared(A, B)

    return AW_2squares, AW_2bench


def plot_error(AW_2squares, AW_2bench, markovian=False):
    if markovian:
        file_name = "stat_markovian"
    else:
        file_name = "stat_non_markovian"

    errors = np.abs(AW_2squares - AW_2bench)
    with open(file_name + ".npy", "wb") as f:
        np.save(f, errors)

    mean_error = np.mean(errors, axis=1)
    std_error = np.std(errors, axis=1)

    # Plot mean error with shaded standard deviation
    plt.figure(figsize=(10, 5))
    plt.plot(n_sample_list, mean_error, label="Mean Error")
    plt.fill_between(
        n_sample_list,
        mean_error - std_error,
        mean_error + std_error,
        alpha=0.3,
        label="±1 Std Dev",
    )

    # Labels and title
    plt.xlabel("n_sample")
    plt.ylabel("error")
    plt.title("Error Plot with Number of Samples")
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name + ".png")
    plt.close()


if __name__ == "__main__":

    n = 5
    n_trial = 10
    n_sample_base = 1000
    n_sample_list = np.array([1000, 2000, 3000, 4000])

    # Markovian
    L = np.array([[1, 0, 0], [1, 2, 0], [2, 4, 2]])
    M = np.array([[1, 0, 0], [2, 1, 0], [2, 1, 2]])
    AW_2squares, AW_2bench = compute_AW2squares(L, M, markovian=False)
    plot_error(AW_2squares, AW_2bench, markovian=False)
