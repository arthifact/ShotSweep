"""OpenFWI 2D acquisition, identical for all ten families: 70 x 70 cells of 10 m, five sources and 70 receivers 10 m deep,
1 s recorded at 1 ms, 15 Hz Ricker source. Velocities lie in 1500-4500 m/s."""
import numpy as np
DX = 10.0; DT = 1e-3; NX = NZ = 70; NT = 1000; F0 = 15.0
SRC_X = np.array([0, 17, 34, 52, 69]) * DX          # source positions (m)
DATUM = 10.0                                          # source and receiver depth (m)
VMIN, VMAX = 1500.0, 4500.0
