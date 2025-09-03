#Note: This code only simulates stable points of the system. Unstable points are manually plotted
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import pandas as pd
import seaborn as sns
from scipy.integrate import odeint

#Parameter values
sm = ss = sc = cs = cc = 0.2
h = 7

#Bifurcation parameter range
cm_range = np.linspace(0, 5, 100)
# Time range
t = np.linspace(0, 1000, 100000)
#Intial points 
x0_range = [[0.1, 0.8], [0.1, 0.1], [0.8, 0.8], [0.8, 0.1],
            [-0.1, 0.8], [-0.1, 0.1], [-0.8, 0.8], [-0.8, 0.1]]

#Define drift function
def drift(x,t,cm):
    '''
    Drift function for m and v 
    Args:
        m, v: Dynamical variables m and v
        t: time
        cm: Bifurcation parameter
    '''
    a1 = -(sm + 2 * sc) + (cm - cs)
    a2 = -(cm - cs)

    b1 = 2 * sm
    b2 = h / 2
    b3 = -(2 * sm + ss) + cm + cs
    b4 = -cm -cs - h/2

    drift_m = (a1 + a2 * x[1]) * x[0]
    drift_v = b1 + b2 * x[0] ** 2 + b3 * x[1] + b4 * x[1] ** 2
    x = [drift_m, drift_v]
    return x

#Initialize plot
fig1, axm = plt.subplots()
fig2, axv = plt.subplots()

axm.set(xlabel='$c_M$',ylabel='Steady state polarization')
axv.set(xlabel='$c_M$',ylabel='Steady state average speed')
#Simulation step
for cm in cm_range:
    for x0 in x0_range:
        x = odeint(drift, x0, t, args=(cm,))
        axm.scatter(cm, x[-1][0], color='blue', zorder=3, s=50)
        axv.scatter(cm, x[-1][1], color='blue', zorder=3, s=50)

plt.show()

