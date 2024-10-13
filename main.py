"""

Final project for the C5 course

In this final project the aim is to modelize the combustion of methane in a counterflow configuration

Authors: Samuel HOYOS, Georges SAIFI, Daniel PERALES RIOS

"""

from constants.constants import Lx, Ly, nb_points, L_slot, U_slot, L_coflow, U_coflow, dt
import equations as eq
import numpy as np

# Mesh definition

x = np.linspace(0, Lx, nb_points)
y = np.linspace(0, Ly, nb_points)


# Initial conditions & boundary conditions

u_field = np.zeros((nb_points, nb_points))
v_field = u_field

# The index for the first velocity is equal to (c.L_slot/c.Lx) * c.nb_points
v_field[nb_points - 1, 0:int((L_slot/Lx) * nb_points)] = U_slot
v_field[0, 0:int((L_slot/Lx) * nb_points)] = -U_slot

v_field[0, int((L_slot/Lx) * nb_points) + 1:int((L_coflow/Lx) * nb_points)] = U_coflow
v_field[nb_points - 1, int((L_slot/Lx) * nb_points) + 1:int((L_coflow/Lx) * nb_points)] = -U_coflow

print(eq.fractional_step(u_field, v_field, dt))

