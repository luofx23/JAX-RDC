import numpy as np

Rg = 8.314463
P0 = 10 * 101325
T0 = 300
R0 = 369
rho0 = P0/(R0*T0)
M0 = Rg/R0
e0 = P0/rho0
u0 = np.sqrt(P0/rho0)
x0 = 0.08
t0 = x0/u0