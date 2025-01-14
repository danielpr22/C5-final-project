import numpy as np
import matplotlib.pyplot as plt


def v_map(u: np.array, v: np.array, X: np.array, Y: np.array, color: str):

    plt.streamplot(X, Y, u, v, color=color)
    plt.title("velocity field lines", fontsize=22)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
