#############
# Constants #
#############

nb_points = 64
dt = 1e-6
final_time = 1
nb_timesteps = int(final_time / dt)
Lx, Ly = 2e-3, 2e-3
L_slot, L_coflow= 0.5e-3, 0.5e-3
U_slot, U_coflow = 1.0, 0.2
T_slot, T_coflow = 300, 300
dx = Lx / (nb_points - 1)
dy = Ly / (nb_points - 1)
max_iter_sor = 10000  # Maximum number of iterations for achieving convergence in the SOR method
omega = 1.5  # Parameter for the Successive Overrelaxation Method (SOR), it must be between 1 and 2

##################################
# Tolerances for the convergence #
##################################

tolerance_sor = 1e-7 # Tolerance for the convergence of the SOR algorithm
tolerance_sys = 1e-5 # Tolerance for the convergence of the whole system

#############
# Chemistry #
#############

rho = 1.1614 # Fluid density
nu = 15e-6 # Kinematic viscosity
D = nu # Schmidt number
a = nu # Prandtl number
A = 1.1e8
T_a = 10000
c_p = 1200 # J/(kg * K)
W_N2 = 0.02802  # kg/mol
W_O2 = 0.031999  # kg/mol
W_CH4 = 0.016042  # kg/mol
W_H2O = 0.018015  # kg/mol
W_CO2 = 0.044009  # kg/mol

nu_ch4 = -1
nu_o2 = -2
nu_n2 = 0
nu_h2o = 2
nu_co2 = 1

h_n2 = 0
h_o2 = 0
h_ch4 = -74.9 # kJ/mol
h_h2o = -241.818 # k/mol
h_co2 = -393.52 # kJ/mol