import math
import numpy as np
from fipy import CellVariable, TransientTerm, DiffusionTerm, Variable, PeriodicGrid2D
import configparser as cp

config = cp.ConfigParser()
config.read('default.ini')
CFL = config.getfloat('main', 'CFL')


def calc_density_fvm(n, dx, t, D, xp, np_max, vegf):
    # Check constants for calculation
    if D == 0 and xp == 0:
        return n

    # Preparing mesh of nx, ny size and dx step size
    ny, nx = np.shape(n)
    dy = dx
    mesh = PeriodicGrid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    phi = CellVariable(name="np",
                       mesh=mesh,
                       value=0.)

    # Calculate time step for CFL condition
    if D != 0:
        dt = CFL * dx / D
    else:
        dt = t * 0.45 * dx * dx / (2 * 0.01)

    # Calculate number of steps for the simulation
    steps = math.ceil(t / dt)

    # Create variables of the PDE based on the created mesh
    ncells = CellVariable(mesh=mesh, value=1., hasOld=1, name='cells')
    ncells.value = np.resize(n, nx * ny)

    # Create a time variable and dependant variable for the simulation
    time = Variable(0.)

    # Define initial conditions of the simulation
    phi.value = np.resize(n, nx * ny)

    # Define the PDE equation for the solver
    eq = TransientTerm() == DiffusionTerm(coeff=D)

    # Run the solution steps times
    for step in range(steps):
        eq.solve(var=phi, dt=dt)
        time.setValue(time.value + dt)

    return np.resize(phi, (nx, ny))
