class constants:

    nb_points = 64

    dt = 1e-2

    Lx = 2e-3

    Ly = 2e-3

    L_slot = 0.5e-3

    L_coflow = 0.5e-3

    U_slot = 1.0

    T_slot = 300

    U_coflow = 0.2

    T_coflow = 300

    # Fluid density
    rho = 1.1614

    # Kimematic viscosity
    eta = 15e-6

    dx = Lx / (nb_points - 1)

    dy = Ly / (nb_points - 1)

    nu = 15e-6

    tolerance = 1e-6
