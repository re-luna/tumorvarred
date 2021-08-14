import random
from math import sqrt
from scipy.stats import norm
import numpy as np
import configparser as cp
import json

# Initialization of configparser for reading config.ini
config = cp.ConfigParser()
config.read('default.ini')

# Agent based model for cells
"""
Enumeration for the constants in array form
"""

Ncell = 0
Ccell = 1
Ecell = 2

"""
Enumeration for Cells list array
"""

P = 0
X = 1
Y = 2
PHI = 3
P53 = 4
Z = 5
GAMMA = 6
VEGF_INT = 7
GEN = 8
ID = 8

"""
Table of the ABM for cells, taken from Table 1 in section 2.2 Mean-field description.
"""

Xp = np.asarray(json.loads(config.get('abm', 'XP')))  # Sensitivity for ECells [cm^2/min/nM]
Gen_max = np.asarray(json.loads(config.get('abm', 'GEN_MAX')))  # Max generation for cell division [times]
C_phi = np.asarray(json.loads(config.get('abm', 'C_PHI')))  # Constant for cell speed growth [mmHg]
C_VEGF = np.asarray(json.loads(config.get('abm', 'C_VEGF')))  # Constant for cell speed VEGF [mmHg]
C_p53 = np.asarray(json.loads(config.get('abm', 'C_P53')))  # Constant for cell P53 [mmHg]
Tp_min = np.asarray(json.loads(config.get('abm', 'TP_MIN')))  # Time for cell division [min]
Z_h = config.getfloat('abm', 'Z_H')  # Constant for normal environment []
Z_l = config.getfloat('abm', 'Z_L')  # Constant for harsh environment []
N_thr = config.getfloat('abm', 'N_THR')  # Density threshold for cancer cells []
O2_thr = config.getfloat('abm', 'O2_THR')  # Oxygen threshold for normal cells [mmHg]
VEGF_thr = np.asarray(json.loads(config.get('abm', 'VEGF_THR')))  # Vegf threshold [nM]
C = np.asarray(json.loads(config.get('abm', 'C')))  # Constants for P53 model [1/min]
J5 = config.getfloat('abm', 'J5')  # Constant for vegf model [1/min]
A = config.getfloat('abm', 'A')  # Linear increasing constant for hypoxia [1/min]
B = config.getfloat('abm', 'B')  # Exponential decay constant for hypoxia [1/min]

"""
Table of the ABM for cells, taken from Table 1 in section 2.2 Mean-field description.
"""

Dp = np.asarray(json.loads(config.get('abm', 'DP')))  # Brownian motion constant for cells [cm^2/min]
I_p = np.asarray(json.loads(config.get('abm', 'I_P')))  # Initial quantity of cells [# particles]
Qi = np.asarray(json.loads(config.get('abm', 'QI')))  #
Delta_t = config.getfloat('abm', 'DELTA_T')  # Time step of the multi-scale model [min]
np_max = np.asarray(json.loads(config.get('abm', 'NP_MAX')))  # Max density of particles for control [# particles]
Delta_x = np.asarray(json.loads(config.get('abm', 'DELTA_X')))  # Spatial separation [cm]
a = [0.5 * Delta_x[Ncell], 0.25 * Delta_x[Ccell]]  # [cm]
b = [0.5 * Delta_x[Ncell], 0.25 * Delta_x[Ccell]]  # [cm]

"""
Other constants
"""
P_MAX = config.getfloat('abm', 'P_MAX')
V_SPROUT = config.getfloat('abm', 'V_SPROUT')
POS_PERIODIC = config.getint('abm', 'POS_PERIODIC')
APOPTOSIS_MODEL = config.getint('abm', 'APOPTOSIS_MODEL')
O2_THR_L = config.getfloat('abm', 'O2_THR_L')
T_DEATH = config.getint('abm', 'T_DEATH')
MOORE_DIVISION = config.getint('abm', 'MOORE_DIVISION')

"""
Definition of functions
"""


def spawn_cells(lcells, quantity, p, distribution, x, y, dx, dy, father):
    """
    Spawn new cells of the specific type in the given cell list

    :param lcells: List of cells
    :param quantity: Quantity of cells
    :param p: Type of cells
    :param distribution: Distribution of the cells "random"
    :param x: Initial position of x
    :param y: Initial position of y
    :param dx: Distance of spawning on x
    :param dy: Distance of spawning on y
    :param father: Cell for inherit characteristics
    :return:
    """
    if type(quantity) == np.ndarray:
        print("is an array")
        if not quantity.any():
            return lcells
        else:
            lcells = np.append(lcells, father, axis=0)
            print("cell created")
            return lcells
    rows, columns = np.shape(lcells)
    lcells = np.vstack((lcells, father))
    if distribution == "random":
        lcells[rows:, P] = np.ones(quantity) * p
        lcells[rows:, X] = x + 50 * np.random.normal(0.25, 0.05, quantity)
        lcells[rows:, Y] = y + 50 * np.random.normal(0.25, 0.05, quantity)

    if distribution == "uniform":
        lcells[rows:, P] = np.ones(quantity) * p
        lcells[rows:, X] = np.random.uniform(x, 50 - 0.000001, quantity)
        lcells[rows:, Y] = np.random.uniform(y, 50 - 0.000001, quantity)
        """
        lcells[rows:, P] = np.ones(quantity) * p
        x_dist = int(quantity / dx)
        y_pos = np.linspace(y, y + dy - 0.0000001, dx)
        for i in range(dx):
            lcells[rows + i * x_dist:rows + (i + 1) * x_dist, X] = np.linspace(x, x + dx - 0.00000001, x_dist)
            lcells[rows + i * x_dist:rows + (i + 1) * x_dist, Y] = y_pos[i]
        # lcells[rows:, Y] = np.linspace(y, y + dy - 0.0000001, quantity)
        """

    if distribution == "circle_random":
        lcells[rows:, P] = np.ones(quantity) * p
        length = np.sqrt(np.random.uniform(0, 1, quantity))
        angle = np.pi * np.random.uniform(0, 2, quantity)
        lcells[rows:, X] = x + dx * length * np.cos(angle)
        lcells[rows:, Y] = y + dy * length * np.sin(angle)

    if distribution == "circle_uniform":
        lcells[rows:, P] = np.ones(quantity) * p
        angle = np.pi * np.linspace(0, 2 * dy, quantity)
        length = np.sqrt(np.linspace(0, 1, quantity))
        lcells[rows:, X] = x + dx * length * np.cos(angle)
        lcells[rows:, Y] = y + dy * length * np.sin(angle)

    return lcells


def cells_to_matrix(lcells, mcells):
    """
    Allocate the cells from the given list to a given lattice matrix
    :param lcells: List of the cells in the format [Type,X,Y,0,...,0]
    :param mcells: Lattice matrix of the size (lx,ly)
    :return: gives the matrix with the cells allocated
    """

    x, y = np.shape(mcells)
    mcells = np.zeros((x, y))
    qcells = int(np.size(lcells) / 9)

    for i in range(qcells):
        mcells[int(lcells[i][X]), int(lcells[i][Y])] += 1
    return mcells


def update_position(lcells, dt, spatial_delta, lx, ly, vegf, mcells, angiogenesis):
    """
    Updates the positions (x,y) of the cells
    :param lcells: List of the cells in the format [P,X,Y,0,...,0]
    :param dt: Time step of the movement in minutes
    :param spatial_delta: Space interval in cm
    :param lx: Lattice x size (int)
    :param ly: Lattice y size (int)
    :param vegf: Vegf matrix (Array)
    :return: return the list of the cells updated
    """


    n3 = cells_to_matrix(lcells[lcells[:, P] == Ecell, :], mcells)
    grad_vegf = np.gradient(vegf)
    xp_scale = sqrt(Xp[Ecell]) / spatial_delta


    for i in range(dt):

    # Directed movement
        if angiogenesis == 1:
            lcells = correct_position(lcells, lx, ly)
            lcells[:, X] += xp_scale * grad_vegf[0][lcells[:, X].astype(int), lcells[:, Y].astype(int)] * \
                            (1 - (n3[lcells[:, X].astype(int), lcells[:, Y].astype(int)]) /
                             np_max[lcells[:, P].astype(int)]) * (lcells[:, P] == Ecell)

            # Periodic lattice and symmetrical condition for the movement
            lcells = correct_position(lcells, lx, ly)

            lcells[:, Y] += xp_scale * grad_vegf[1][lcells[:, X].astype(int), lcells[:, Y].astype(int)] * \
                            (1 - (n3[lcells[:, X].astype(int), lcells[:, Y].astype(int)]) /
                             np_max[lcells[:, P].astype(int)]) * (lcells[:, P] == Ecell)

            # Periodic lattice and symmetrical condition for the movement
            lcells = correct_position(lcells, lx, ly)

    # Brownian motion of the Non-Normal cells

        # Cancer cells
        l_size_x, l_size_y = np.shape(lcells)
        lcells[:, X] += np.random.normal(0, sqrt(Dp[Ccell]) / spatial_delta, l_size_x) * (
                    (lcells[:, P] == Ccell) | (lcells[:, P] == Ecell))
        lcells[:, Y] += np.random.normal(0, sqrt(Dp[Ccell]) / spatial_delta, l_size_x) * (
                    (lcells[:, P] == Ccell) | (lcells[:, P] == Ecell))
        # lcells[:, X] += np.random.normal(0, sqrt(Dp[Ecell]) / spatial_delta, l_size_x) * (lcells[:, P] == Ecell)
        # lcells[:, Y] += np.random.normal(0, sqrt(Dp[Ecell]) / spatial_delta, l_size_x) * (lcells[:, P] == Ecell)

    # Periodic lattice and symmetrical condition for the movement
    lcells = correct_position(lcells, lx, ly)

    return lcells


def correct_position(lcells, lx, ly):
    # Periodic lattice and symmetrical condition for the movement
    if POS_PERIODIC == 1:
        lcells[:, X] += (lcells[:, X] >= lx) * (- lx)
        lcells[:, Y] += (lcells[:, Y] >= ly) * (- ly)
        lcells[:, X] += (lcells[:, X] < 0.0) * lx
        lcells[:, Y] += (lcells[:, Y] < 0.0) * ly
    else:
        lcells[:, X] += (lcells[:, X] >= lx) * (lx - lcells[:, X] - 0.00001)
        lcells[:, Y] += (lcells[:, Y] >= ly) * (ly - lcells[:, Y] - 0.00001)
        lcells[:, X] -= (lcells[:, X] < 0.0) * lcells[:, X]
        lcells[:, Y] -= (lcells[:, Y] < 0.0) * lcells[:, Y]

    return lcells


def update_cell_division(lcells, o2, dt):
    """
    Updates the size of the given cell list using eq (6) and sup. material
    :param lcells: List of cells
    :param o2: Oxygen distribution in lattice array shape
    :param dt: time step in minutes
    :return: Returns the new list of cells
    """
    # Calculate vector form of Oxygen for the cells
    o2_cell = o2[lcells[:, X].astype(int), lcells[:, Y].astype(int)]

    # Prepare heaviside function calculation according the generation of every cell
    heaviside = np.heaviside(Gen_max[lcells[:, P].astype(int)] - lcells[:, GEN], 1) * (lcells[:, P] != Ecell)
    if APOPTOSIS_MODEL == 1:
        heaviside = heaviside * (lcells[:, P] == Ccell) * (lcells[:, Z] == 0) + heaviside * (lcells[:, P] == Ncell)

    # Calculates the new value of the cell growth for cellular division
    lcells[:, PHI] += dt * o2_cell / (Tp_min[lcells[:, P].astype(int)] *
                                      (C_phi[lcells[:, P].astype(int)] + o2_cell)) * heaviside

    # Get cells' list index that could be divide
    div_cells = np.copy(lcells[lcells[:, PHI] >= 1.0, :])

    # Get an index list of cells ready for division
    r = 0
    if div_cells.size != 0:
        r, c = lcells.shape
        index_list = np.arange(r)
        index_list += (- 1 - index_list) * (lcells[:, PHI] < 1.0)
        index_list = np.copy(index_list[index_list > -1])
        new_cells = np.empty((0, 9))

        n1 = Qi[Ncell] * cells_to_matrix(lcells[lcells[:, P] == Ncell, :], o2)
        n2 = Qi[Ncell] * cells_to_matrix(lcells[lcells[:, P] == Ccell, :], o2)
        n_cells = n1 + n2

        r = index_list.size

        # Division process
        lx, ly = np.shape(o2)
        for i in range(r):
            if MOORE_DIVISION == 1:

                # Initialize random Moore's neighborhood
                ri = [1, 2, 3, 4, 5, 6, 7, 8]
                np.random.shuffle(ri)
                ri = np.insert(ri, 0, 0)
                moores = [[0, 0],
                          [-1, -1],
                          [-1, 0],
                          [-1, 1],
                          [0, -1],
                          [0, 1],
                          [1, -1],
                          [1, 0],
                          [1, 1]]

                # Check for space
                for j in range(9):

                    # Calculates the position of the space to be check
                    if POS_PERIODIC == 1:
                        x, y = latticeWrapIdx([int(lcells[index_list[i]][X]) + moores[ri[j]][0],
                                               int(lcells[index_list[i]][Y]) + moores[ri[j]][1]],
                                              (lx, ly))
                    else:
                        x, y = [int(lcells[index_list[i]][X]) + moores[ri[j]][0],
                                int(lcells[index_list[i]][Y]) + moores[ri[j]][1]]
                        if x >= lx: x = lx - 1
                        if y >= ly: y = ly - 1
                        if x < 0: x = 0
                        if y < 0: y = 0

                    position = n_cells[x, y]

                    # If the space is free allocates the new cell in that square
                    if position < np_max[int(lcells[index_list[i]][P])]:
                        new_cells = np.vstack((new_cells, lcells[index_list[i]]))

                        # Locates the new cell at the center of the free space and overrides Gen param
                        new_cells[int(new_cells.size / 9) - 1, X:PHI] = [x + 0.5, y + 0.5]
                        new_cells[int(new_cells.size / 9) - 1, GEN] = 0

                        # Add a generation to the parent cell
                        lcells[index_list[i], GEN] += 1

                        # Add the weight of the cell to the cell lattice
                        n_cells[x, y] += Qi[int(new_cells[int(new_cells.size / 9) - 1, P])]
            else:
                x, y = [int(lcells[index_list[i]][X]),
                        int(lcells[index_list[i]][Y])]
                position = n_cells[x, y]

                # If the space is free allocates the new cell in that square
                if position < np_max[int(lcells[index_list[i]][P])]:
                    new_cells = np.vstack((new_cells, lcells[index_list[i]]))

                    # Locates the new cell at the center of the free space and overrides Gen param
                    new_cells[int(new_cells.size / 9) - 1, X:PHI] = [x + 0.5, y + 0.5]
                    new_cells[int(new_cells.size / 9) - 1, GEN] = 0

                    # Add a generation to the parent cell
                    lcells[index_list[i], GEN] += 1

                    # Add the weight of the cell to the cell lattice
                    n_cells[x, y] += Qi[int(new_cells[int(new_cells.size / 9) - 1, P])]

    # Append cells copying all attributes
    if div_cells.size != 0:
        lcells = np.vstack((lcells, new_cells))

    # Clear the division condition
    lcells[:, PHI] = lcells[:, PHI] * (lcells[:, PHI] < 1.0)

    return lcells


def update_intracellular(lcells, o2, dt):
    """
    Updates the intracellular model of the cells, P53 and VEGF_Int
    :param lcells: List of the cells in arrays form
    :param o2: Oxygen distribution in lattice array shape
    :param dt: Time step in minutes for the simulation
    :return:
    """

    # Calculates the vector form of the oxygen for every cell
    o2_cell = o2[lcells[:, X].astype(int), lcells[:, Y].astype(int)]

    # Calculates P53 and Vegf_int for every cell according to eq (9)
    lcells[:, P53] += dt * (C[0] - (C[1] * o2_cell * lcells[:, P53]) / (C_p53[lcells[:, P].astype(int)] + o2_cell))
    lcells[:, VEGF_INT] += dt * (C[2] - (C[3] * lcells[:, P53] * lcells[:, VEGF_INT]) / (J5 + lcells[:, VEGF_INT]) -
                                 (C[4] * o2_cell * lcells[:, VEGF_INT]) /
                                 (C_VEGF[lcells[:, P].astype(int)] + o2_cell)) * (lcells[:, P] != Ecell)

    return lcells


def update_apoptosis(lcells, o2, dt, mcells):
    """
    Calculates the Z and Gamma values and then updates the list cell
    :param lcells: List of cells
    :param o2: Oxygen distribution in lattice array shape
    :param dt: Time step for update
    :param mcells: Lattice size array for allocate the cells
    :return: List of cells updated
    """

    # Calculate apoptosis function for normal cells
    n1 = cells_to_matrix(lcells[lcells[:, P] == Ncell, :], mcells)
    n2 = cells_to_matrix(lcells[lcells[:, P] == Ccell, :], mcells)
    rho_normal = n1 / (n1 + n2 + 0.001)
    rho_normal_cell = rho_normal[lcells[:, X].astype(int), lcells[:, Y].astype(int)]

    lcells[:, GAMMA] += np.heaviside(lcells[:, P53] - Z_l * np.heaviside(N_thr - rho_normal_cell, 1) -
                                     Z_h * np.heaviside(rho_normal_cell - N_thr, 1), 1) * (lcells[:, P] == Ncell)
    lcells = lcells[lcells[:, GAMMA] < 1, :]

    # Calculate hypoxia and apoptosis state for cancer cells
    o2_cell = o2[lcells[:, X].astype(int), lcells[:, Y].astype(int)]

    if APOPTOSIS_MODEL == 0:
        lcells[:, Z] += dt * (A * np.heaviside(O2_thr - o2_cell, 1) -
                              dt * B * lcells[:, Z] * np.heaviside(o2_cell - O2_thr, 1)) * (lcells[:, P] == Ccell)
        lcells[:, GAMMA] += np.heaviside(lcells[:, Z] - 1, 1) * (lcells[:, P] == Ccell)

    if APOPTOSIS_MODEL == 1:
        lcells[:, Z] += dt*np.heaviside(O2_thr - o2_cell, 1) * np.heaviside(O2_THR_L - o2_cell, 1) * (lcells[:, P] == Ccell)
        lcells[:, Z] -= lcells[:, Z] * np.heaviside( o2_cell - O2_THR_L, 1) * (lcells[:, P] == Ccell)
        lcells[:, GAMMA] += np.heaviside(lcells[:, Z] - T_DEATH, 1) * (lcells[:, P] == Ccell)



    # Kill cells in apoptosis from the cell list
    lcells = lcells[lcells[:, GAMMA] < 1, :]
    return lcells


def latticeWrapIdx(index, lattice_shape):
    """
        Returns periodic lattice index for a given iterable index
        :param index: List of cells
        :param lattice_shape: Oxygen distribution in lattice array shape
        :return: Modified indexes
    """
    if not hasattr(index, '__iter__'): return index  # handle integer slices
    if len(index) != len(lattice_shape): return index  # must reference a scalar
    if any(type(i) == slice for i in index): return index  # slices not supported
    if len(index) == len(lattice_shape):  # periodic indexing of scalars
        mod_index = tuple(((i % s + s) % s for i, s in zip(index, lattice_shape)))
        return mod_index
    raise ValueError('Unexpected index: {}'.format(index))


"""
Angiogenesis functions definitions
"""


def angiogenesis(lcells, excluded_positions, vessels, vegf, dt, a_nodes):
    # Recall endothelial cells list
    lx, ly = np.shape(vegf)
    n3_list = lcells[lcells[:, P] == Ecell, :]
    n3_list = np.append(n3_list, excluded_positions, axis=0)
    n3 = np.zeros((lx, ly))

    # Exclusion radius due Endothelial cells
    qcells = int(np.size(n3_list) / 9)
    for i in range(qcells):
        n3[int(n3_list[i][X]), int(n3_list[i][Y])] = 1

        # Exclusion radius calculation
        for j in range(5):
            for k in range(5):
                x, y = latticeWrapIdx([int(n3_list[i][X]) + j - 1,
                                       int(n3_list[i][Y]) + k - 1],
                                      np.shape(vegf))
                n3[x, y] = 1

    # Exclusion radius due stalk cells
    rows_an, columns_an = a_nodes.shape
    for i in range(rows_an):
        n3[int(a_nodes[i][X]), int(a_nodes[i][Y])] = 1

        # Exclusion radius calculation
        for j in range(5):
            for k in range(5):
                x, y = latticeWrapIdx([int(a_nodes[i][0]) + j - 1,
                                       int(a_nodes[i][1]) + k - 1],
                                      np.shape(vegf))
                n3[x, y] = 1

    n3 = 1 - n3

    # Generate a lx,ly matrix with random values and calculate the sprouting probability
    p_sprout = dt * P_MAX * n3 * vessels * vegf / (V_SPROUT + vegf)
    p_random = np.random.rand(lx, ly)
    (m_x, m_y) = np.where(np.asarray(p_sprout) > p_random)
    id_counter = 0
    if not np.size(m_x) == 0:
        sprouts = np.zeros((1, 9))
        if n3_list.size != 0:
            id_counter += np.amax(n3_list[:, ID]) + 1

        # Choose the random cell position to be introduced
        random_sprout = random.randint(0, np.size(m_x)-1)
        sprouts[0, :3] = [Ecell, m_x[random_sprout], m_y[random_sprout]]
        sprouts[0, ID] = id_counter

        lcells = np.append(lcells, sprouts, axis=0)
        excluded_positions = np.append(excluded_positions, sprouts, axis=0)

    # Periodic lattice and symmetrical condition for the movement
    lcells = correct_position(lcells, lx, ly)
    return lcells, excluded_positions


def anastomosis(lcells, nodes, segments, a_nodes, a_segments):
    Ecells = lcells[lcells[:, P] == Ecell, :]
    if Ecells.size == 0:
        return lcells
    r, c = Ecells.shape
    L = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            L[i, j] = np.linalg.norm(Ecells[i, X:PHI]-Ecells[j, X:PHI])
    L += np.eye(r, dtype=int)
    if np.amin(L) < np.amax(segments[:, 5] * .004):
        ind = np.unravel_index(np.argmin(L, axis=None), L.shape)
        
    return lcells, nodes, segments

def tip_prunning(lcells, dt):
    lcells[:, Z] += dt * (lcells[:, P] == Ecell)
    lcells[:, GAMMA] += np.heaviside(lcells[:, Z] - T_DEATH, 1) * (lcells[:, P] == Ecell)
    lcells = lcells[lcells[:, GAMMA] < 1, :]
    return lcells