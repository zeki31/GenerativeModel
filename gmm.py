from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    dim = len(x)
    denom = 1 / np.sqrt((2 * np.pi) ** dim * det)
    y = denom * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y


def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y


def likelihood(xs, phis, mus, covs):
    eps = 1e-8  # Avoid log(0)
    ll = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        ll += np.log(y + eps)
    return ll / N


def plot_contour(w, mus, covs):
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])

            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * multivariate_normal(x, mu, cov)
    plt.contour(X, Y, Z)


class GMM:
    def __init__(self, xs):
        self.xs = xs

        self.phis = np.array([0.5, 0.5])
        self.mus = np.array([[0.0, 50.0], [0.0, 100.0]])
        self.covs = np.array([np.eye(2), np.eye(2)])

        self.K = len(self.phis)  # 2
        self.N = xs.shape[0]  # 272
        self.MAX_ITERS = 100
        self.THRESHOLD = 1e-4

    def fit(self):
        current_ll = likelihood(self.xs, self.phis, self.mus, self.covs)

        for iter in range(self.MAX_ITERS):
            # E-step
            qs = np.zeros((self.N, self.K))
            for n in range(self.N):
                x = self.xs[n]
                for k in range(self.K):
                    phi, mu, cov = self.phis[k], self.mus[k], self.covs[k]
                    qs[n, k] = phi * multivariate_normal(x, mu, cov)
                qs[n] /= gmm(x, self.phis, self.mus, self.covs)

            # M-step
            qs_sum = qs.sum(axis=0)
            for k in range(self.K):
                # 1. phis
                self.phis[k] = qs_sum[k] / self.N

                # 2. mus
                c = 0
                for n in range(self.N):
                    c += qs[n, k] * self.xs[n]
                self.mus[k] = c / qs_sum[k]

                # 3. covs
                c = 0
                for n in range(self.N):
                    z = self.xs[n] - self.mus[k]
                    z = z[:, np.newaxis]  # (2,) -> (2, 1)
                    c += qs[n, k] * (z @ z.T)
                self.covs[k] = c / qs_sum[k]

            # Check convergence
            logger.info(f"iter: {iter}, current_ll: {current_ll:.4f}")
            next_ll = likelihood(self.xs, self.phis, self.mus, self.covs)
            if abs(next_ll - current_ll) < self.THRESHOLD:
                logger.info(f"Converged at iteration {iter}.")
                break
            current_ll = next_ll

    def viz(self):
        plt.scatter(self.xs[:, 0], self.xs[:, 1])
        plot_contour(self.phis, self.mus, self.covs)
        plt.xlabel("Eruptions(Min)")
        plt.ylabel("Waiting(Min)")
        plt.show()

    def generate_and_viz(self, N):
        new_xs = np.zeros((N, 2))
        for n in range(N):
            k = np.random.choice(self.K, p=self.phis)
            new_xs[n] = np.random.multivariate_normal(self.mus[k], self.covs[k])
        plt.scatter(self.xs[:, 0], self.xs[:, 1], alpha=0.7, label="original")
        plt.scatter(new_xs[:, 0], new_xs[:, 1], alpha=0.7, label="generated")
        plt.legend()
        plt.xlabel("Eruptions(Min)")
        plt.ylabel("Waiting(Min)")
        plt.show()
        return new_xs


if __name__ == "__main__":
    data_path = Path("data/step05/old_faithful.txt")
    xs = np.loadtxt(data_path)
    logger.info(f"xs.shape: {xs.shape}")

    model = GMM(xs)
    model.fit()
    model.viz()
    model.generate_and_viz(N=500)
