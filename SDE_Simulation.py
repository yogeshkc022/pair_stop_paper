import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

repetitions = 100 #Number of repetitions
n = 100000 #Number of simulation steps
del_t = 0.01

N = 500 #Number of individuals

#Parameter values
sm = ss = sc = cs = cc = 0.2
h = 7
cm = 2
rates = np.array([sm, ss, sc, cm, cs, cc, h], dtype=np.float32)

# Storing output
store = np.zeros((repetitions, n, 2), dtype=np.float32)

#Define drift and diffusion functions
@njit
def drift(m, v, rates):
    '''
    Drift function for m and v 
    Args:
        m, v: Dynamical variables m and v
        rates: List/array of interaction rates
    '''
    sm, ss, sc, cm, cs, cc, h = rates
    a1 = -(sm + 2 * sc) + (cm - cs)
    a2 = -(cm - cs)
    b1 = 2 * sm
    b2 = h / 2
    b3 = -(2 * sm + ss) + cm + cs
    b4 = -cm - cs - h / 2
    return (a1 + a2 * v) * m, b1 + b2 * m ** 2 + b3 * v + b4 * v ** 2

@njit
def diff(m, v, rates, N):
    '''
    Diffusion function for m and v 
    Args:
        m, v: Dynamical variables m and v
        rates: List/array of interaction rates
    '''
    sm, ss, sc, cm, cs, cc, h = rates
    f1 = 2 * sm
    f2 = -cc / 2 - 3 * h / 4
    f3 = -2 * sm + 3 * ss / 2 + sc + cm + cs
    f4 = cc / 2 + 3 * h / 4 - cm - cs
    e1 = 2 * sm
    e2 = -h / 2
    e3 = -2 * sm + ss + cm + cs
    e4 = h / 2 - cm - cs
    diff_m = (f1 + f2 * m ** 2 + f3 * v + f4 * v ** 2) / N
    diff_v = (e1 + e2 * m ** 2 + e3 * v + e4 * v ** 2) / N
    return diff_m, diff_v

#Simulation algorithm
@njit(parallel=True)
def simulate(store, rates, repetitions, n, del_t, N):
    for rep in prange(repetitions):
        x = np.zeros((n, 2), dtype=np.float32)
        x[0, 0] = np.random.uniform(0, 1) if rep < 50 else np.random.uniform(-1, 0)
        x[0, 1] = np.random.uniform(0, 1)

        for i in range(n - 1):
            m, v = x[i]
            dm, dv = drift(m, v, rates)
            Dm, Dv = diff(m, v, rates, N)

            if np.isfinite(dm) and np.isfinite(dv) and np.isfinite(Dm) and np.isfinite(Dv):
                x[i + 1, 0] = m + dm * del_t + np.random.normal(0, 1) * np.sqrt(max(Dm, 0) * del_t)
                x[i + 1, 1] = v + dv * del_t + np.random.normal(0, 1) * np.sqrt(max(Dv, 0) * del_t)

        store[rep] = x

#Run simulation
simulate(store, rates, repetitions, n, del_t, N)

# Visualization
plt.figure()
plt.plot(np.arange(n), store[0, :, 0], label='Polarization', color='steelblue')
plt.plot(np.arange(n), store[0, :, 1], label='Speed', color='darkorange')
plt.xlim([0, 10000])
plt.ylim(-1, 1)
plt.xlabel('Time step')
plt.ylabel('State')
plt.title('SDE - Sample Trajectory')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()