import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit

mu = 14 #Number of interactions
N = 500 #Number of individuals

#Parameter values
sm = ss = sc = cs = cc = 0.2
h = 7
cm = 2
C = np.array([sm, ss, sc, cm, cs, cc, h], dtype=np.float32)

n = 100000 #Number of simulation steps
rel = 100 #Number of repetitions

#Simulation algorithm
@njit
def simulate(rel, n, N, C, mu):
    P_list = []
    S_list = []
    t_list = []

    for iter in range(rel):
        # Initial states
        if iter < rel // 2:
            X1 = np.random.randint(N // 3, N)
            X2 = np.random.randint(0, N - X1)
        else:
            X2 = np.random.randint(N // 3, N)
            X1 = np.random.randint(0, N - X2)
        X0 = N - X1 - X2

        T = 0
        Tprint = 0.01

        for _ in range(n):
            # Precompute terms
            X0_N = X0 / N
            X1_N = X1 / N
            X2_N = X2 / N
            X0X1_N2 = X0 * X1 / (N * N)
            X0X2_N2 = X0 * X2 / (N * N)
            X1X2_N2 = X1 * X2 / (N * N)

            # Propensities
            A = np.zeros(mu, dtype=np.float32)
            A[0] = A[1] = C[0] * X0_N
            A[2] = C[1] * X1_N
            A[3] = C[1] * X2_N
            A[4] = C[2] * X1_N
            A[5] = C[2] * X2_N
            A[6] = C[3] * X0X1_N2
            A[7] = C[3] * X0X2_N2
            A[8] = C[4] * X0X1_N2
            A[9] = C[4] * X0X2_N2
            A[10] = A[11] = C[5] * X1X2_N2
            A[12] = A[13] = C[6] * X1X2_N2

            A0 = np.sum(A)
            if A0 == 0:
                break

            T += np.log(1 / np.random.rand()) / A0

            if T > Tprint:
                polarization = (X1 - X2) / N
                speed = (X1 + X2) / N
                P_list.append(polarization)
                S_list.append(speed)
                t_list.append(T)
                Tprint += 0.01  # advance sampling time

            # Choose reaction
            R2 = np.random.rand() * A0
            mu_index = np.searchsorted(np.cumsum(A), R2)

            # Update state based on reaction index
            if mu_index in (0, 6) and X0 > 0:
                X0 -= 1; X1 += 1
            elif mu_index in (1, 7) and X0 > 0:
                X0 -= 1; X2 += 1
            elif mu_index in (2, 8, 12) and X1 > 0:
                X1 -= 1; X0 += 1
            elif mu_index in (3, 9, 13) and X2 > 0:
                X2 -= 1; X0 += 1
            elif mu_index in (4, 11) and X1 > 0:
                X1 -= 1; X2 += 1
            elif mu_index in (5, 10) and X2 > 0:
                X2 -= 1; X1 += 1

    return np.array(P_list), np.array(S_list), np.array(t_list)

# Run simulation
P_array, S_array, t_array = simulate(rel, n, N, C, mu)

# Visualization
sns.set(style="whitegrid")
plt.plot(t_array[:100000], P_array[:100000], label='Polarization', color='steelblue')
plt.plot(t_array[:100000], S_array[:100000], label='Speed', color='darkorange')
plt.xlim([0, 2000])
plt.ylim(-1, 1)
plt.xlabel("Time")
plt.ylabel("State")
plt.title("SSA - Sample Trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()