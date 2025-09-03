import numpy as np
import matplotlib.pyplot as plt

#Define drift functions
def dm(m, v, rates):
    '''
    Drift function for m.    Args:
        m, v: Dynamical variables m and v
        rates: List/array of interaction rates
    '''
    sm, ss, sc, cm, cs, h = rates    
    a1 = - (sm + 2 * sc) + (cm - cs)
    a2 = - (cm - cs)
    return (a1 + a2 * v) * m

def dv(m, v, rates):
    '''
    Drift function for v.    Args:
        m, v: Dynamical variables m and v
        rates: List/array of interaction rates
    '''    
    sm, ss, sc, cm, cs, h = rates    
    b1 = 2 * sm
    b2 = h / 2
    b3 = - (2 * sm + ss) + (cm + cs)
    b4 = - (cm + cs) - h / 2    
    return b1 + b2 * m ** 2 + b3 * v + b4 * v ** 2

#parameter values
sm = ss = sc = cs = 0.2
cm = 1
h = 7
rates = np.array([sm, ss, sc, cm, cs, h])


mbins, vbins = np.linspace(-1, 1, 100), np.linspace(0, 1, 100)
mm, vv = np.meshgrid(mbins, vbins)
dm_ = dm(mm, vv, rates)
dv_ = dv(mm, vv, rates)

#Visualzation
plt.streamplot(mbins, vbins, dm_, dv_, linewidth=1.5,color = 'Black')
plt.xlabel('Order parameter(m)')
plt.ylabel('Average speed(v)')
plt.xlim(-1,1)
plt.ylim(0,1)
plt.show()