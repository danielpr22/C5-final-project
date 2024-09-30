"""

Final project for the C5 course

Authors: Samuel HOYOS, Georges SAIFI, Daniel PERALES RIOS

"""

from constants import constants as c
import numpy as np

# Mesh definition

x = np.linspace(0, c.Lx, c.nb_points)
y = np.linspace(0, c.Ly, c.nb_points)


# Initial conditions & boundary conditions

u_field = np.zeros((c.nb_points, c.nb_points))
v_field = u_field

# The index for the first velocity is equal to (c.L_slot/c.Lx) * c.nb_points
v_field[c.nb_points - 1, 0:int((c.L_slot/c.Lx) * c.nb_points)] = c.U_slot
v_field[0, 0:int((c.L_slot/c.Lx) * c.nb_points)] = -c.U_slot
print(v_field)

v_field[0, int((c.L_slot/c.Lx) * c.nb_points) + 1:int((c.L_coflow/c.Lx) * c.nb_points)] = c.U_coflow
v_field[c.nb_points - 1, int((c.L_slot/c.Lx) * c.nb_points) + 1:int((c.L_coflow/c.Lx) * c.nb_points)] = -c.U_coflow