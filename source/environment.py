import math
import numpy as np
from fipy import CellVariable, TransientTerm, DiffusionTerm, Variable, PeriodicGrid2D, ImplicitSourceTerm
from fipy.tools import numerix
import configparser as cp
import json
from PIL import Image, ImageDraw
from xml.dom import minidom

# Initialization of configparser for reading config.ini
config = cp.ConfigParser()
config.read('default.ini')

# Oxygen and VEGF diffusion models
"""
Enumeration for the elements in list form
"""
O2 = 0
VEGF = 1

bX = 0
bY = 1
bT = 2
bP = 3

sNi = 0
sNo = 1
sBN = 2
sX = 3
sY = 4
sR = 5
sL = 6
sQ = 7
sDP = 8
sTw = 9
sKm = 10
sSm = 11
sTau = 12
sSh = 13
sID = 14
sEID = 15

nX = 0
nY = 1
nP = 2
nID = 3
nEID = 4

"""
Table of environmental constants, taken from Table 2 in section 2.4 Environment.
"""
D = np.asarray(json.loads(config.get('environment', 'D')))  # Diffusion coefficient DO2=[Oxygen, VEGF] [cm^2/min]
P = np.asarray(json.loads(
    config.get('environment', 'P')))  # Permeability of the oxygen thru the vessels PO2=[Oxygen, VEGF] [cm/min]
Decay = np.asarray(
    json.loads(config.get('environment', 'Decay')))  # Decay constant of VEGF Decay=[Oxygen, VEFG] [min^-1]
KO2 = config.getfloat('environment', 'KO2')  # Consume rate of Oxygen by the cells [min-1]
KVEGF = config.getfloat('environment', 'KVEGF')  # Consume rate of VEGF by the cells [min-1]
O2ref = config.getfloat('environment', 'O2ref')  # Oxygen concentration reference in a blood vessel mmHg
Hin = config.getfloat('environment', 'H_IN')  # Hematocrit inflow node constant

"""
Vessels distribution constants for initial conditions construction.
"""
# Sizes are lattice grid-wise.
VESSEL_RADIUS = config.getfloat('environment', 'VESSEL_RADIUS')  # Radius of the vessel
VESSEL_DISTANCE = config.getint('environment', 'VESSEL_DISTANCE')  # Distance between vessels
VESSEL_START = config.getint('environment',
                             'VESSEL_START')  # Position in the lattice where the first vessels will spawn
VESSEL_FILE = config['environment'].get('VESSEL_FILE')
Q_INIT_SEGMENTS = config.getint('environment', 'Q_INIT_SEGMENTS')
V_DX = config.getint('environment', 'V_DX')

"""
Blood flow and angiogenesis constants
"""
EPS_T = config.getfloat('environment', 'EPS_T')
MU0 = config.getfloat('environment', 'MU0')
KMV = config.getfloat('environment', 'KMV')
KM0 = config.getfloat('environment', 'KM0')
KS = config.getfloat('environment', 'KS')
KP = config.getfloat('environment', 'KP')
V0 = config.getfloat('environment', 'V0')
Q_REF = config.getfloat('environment', 'Q_REF')
TAU_REF = config.getfloat('environment', 'TAU_REF')
R_TOL = config.getfloat('environment', 'R_TOL')
PO2 = config.getfloat('environment', 'PO2')

"""
Post definition constants
"""
O2kb = O2ref  # Quotient of O2ref/Hin for speeding up purposes
K_TW = 1333.22 / 100000

"""
Others
"""
CFL = config.getfloat('main', 'CFL')


def Ht(vesels):
    """
    *Distribution function of Hematocrit*
    Still pending to implement the right distribution of the hematocrit function
    """
    htc = vesels * .45
    return htc


def nv(vessel):
    """Surface area occupied by the vessel and new sprouts"""
    nvo = np.copy(vessel)
    return nvo


def o2blood(vessel):
    """
    *Calculates the oxygen concentration in a blood vessel*
    returns the O2 blood concentration in the lattice size
    """
    result = O2kb * Ht(vessel)
    return result


def build_vessels(lattice, SPATIAL_DELTA, VESSEL_RADII_ADAPT):
    """
    Build vessels
    :param lattice: Lattice of the environment
    :return: return lattice size matrix distribution
    """
    segments = np.empty((0, 14))
    nodes = np.empty((0, 3))

    if VESSEL_RADII_ADAPT == 1:
        vessel_xml = minidom.parse(VESSEL_FILE)
        nodes_xml = vessel_xml.getElementsByTagName('node')
        connections_xml = vessel_xml.getElementsByTagName('connection')
        boundary_nodes = np.empty((0, 3))
        initial_connections = np.empty((0, 2))

        for elem in nodes_xml:
            new_boundary_node = [float(elem.getElementsByTagName("x")[0].firstChild.data),
                                 float(elem.getElementsByTagName("y")[0].firstChild.data),
                                 float(elem.getElementsByTagName("p")[0].firstChild.data)]
            boundary_nodes = np.vstack((boundary_nodes, new_boundary_node))

        for elem in connections_xml:
            new_connection = [int(elem.getElementsByTagName("in_node")[0].firstChild.data),
                              int(elem.getElementsByTagName("out_node")[0].firstChild.data)]
            initial_connections = np.vstack((initial_connections, new_connection))

        for conn in initial_connections:
            new_nodes = np.zeros((Q_INIT_SEGMENTS - 1, 3))
            new_nodes[:, 0] = np.linspace(
                (boundary_nodes[conn[1].astype(int)][0].astype(int) - boundary_nodes[conn[0].astype(int)][0].astype(
                    int)) / Q_INIT_SEGMENTS + boundary_nodes[conn[0].astype(int)][0].astype(int),
                boundary_nodes[conn[1].astype(int)][0].astype(int), Q_INIT_SEGMENTS - 1, endpoint=False)
            new_nodes[:, 1] = np.linspace(
                (boundary_nodes[conn[1].astype(int)][1].astype(int) - boundary_nodes[conn[0].astype(int)][1].astype(
                    int)) / Q_INIT_SEGMENTS + boundary_nodes[conn[0].astype(int)][1].astype(int),
                boundary_nodes[conn[1].astype(int)][1].astype(int), Q_INIT_SEGMENTS - 1, endpoint=False)
            rows, columns = np.shape(nodes)
            nodes = np.vstack((nodes, new_nodes))
            new_segments = np.zeros((Q_INIT_SEGMENTS, 14))
            new_segments[:, sNi] = np.arange(rows - 1, rows + Q_INIT_SEGMENTS - 1)
            new_segments[:, sNo] = new_segments[:, 0] + 1
            new_segments[0, sNi] = rows
            new_segments[Q_INIT_SEGMENTS - 1, sNo] = new_segments[Q_INIT_SEGMENTS - 1, 0]
            new_segments[0, sBN] = boundary_nodes[conn[0].astype(int)][2]
            new_segments[Q_INIT_SEGMENTS - 1, sBN] = boundary_nodes[conn[1].astype(int)][2]
            new_segments[0, sX] = boundary_nodes[conn[0].astype(int)][0]
            new_segments[Q_INIT_SEGMENTS - 1, sX] = boundary_nodes[conn[1].astype(int)][0]
            new_segments[0, sY] = boundary_nodes[conn[0].astype(int)][1]
            new_segments[Q_INIT_SEGMENTS - 1, sY] = boundary_nodes[conn[1].astype(int)][1]
            new_segments[:, sR] = VESSEL_RADIUS
            new_segments[:, sL] = SPATIAL_DELTA
            segments = np.vstack((segments, new_segments))

        vessel = segments_to_nv(lattice, segments, nodes, SPATIAL_DELTA)
    else:
        vessel = np.copy(lattice)
        vessel[VESSEL_START, :] = 2 * np.pi * VESSEL_RADIUS / 100000 / SPATIAL_DELTA ** 2
        vessel[VESSEL_START + VESSEL_DISTANCE, :] = 2 * np.pi * VESSEL_RADIUS / 100000 / SPATIAL_DELTA ** 2
        #vessel = vessel * 0.0001 / SPATIAL_DELTA


    return vessel, segments, nodes


def update_vessels(vessels, SPATIAL_DELTA):
    return vessels


def segments_to_nv(nv, segments, nodes, SPATIAL_DELTA):
    nx, ny = np.shape(nv)

    # creating new Image object
    img = Image.new("F", (nx * V_DX, ny * V_DX))

    # create line image
    img1 = ImageDraw.Draw(img)

    K_SECTION = 2 * np.pi / 100000 / SPATIAL_DELTA ** 2 * (1 / V_DX)
    for seg in segments:
        img1.line([(nodes[seg[sNi].astype(int), nX] * V_DX, (nodes[seg[sNi].astype(int), nY]) * V_DX),
                   ((seg[sX] + nodes[seg[sNo].astype(int), nX] * (seg[sBN].astype(int).astype(int) == 0)) * V_DX,
                    (seg[sY] + nodes[seg[sNo].astype(int), nY] * (seg[sBN].astype(int).astype(int) == 0)) * V_DX)],
                  fill=K_SECTION * seg[sR])

    pix = np.array(img)
    nv = np.zeros((50, 50))
    for i in range(50):
        for j in range(50):
            nv[i, j] = np.sum(pix[i * V_DX:i * V_DX + V_DX, j * V_DX:j * V_DX + V_DX])

    return nv


def get_nv_geometry(nv, segments, nodes, SPATIAL_DELTA):
    nx, ny = np.shape(nv)

    # creating new Image object
    img = Image.new("F", (nx * V_DX, ny * V_DX))

    # create line image
    img1 = ImageDraw.Draw(img)

    K_SECTION = 2 * np.pi / 100000 / SPATIAL_DELTA ** 2 * (1 / V_DX)
    for seg in segments:
        img1.line([(nodes[seg[sNi].astype(int), nX] * V_DX, (nodes[seg[sNi].astype(int), nY]) * V_DX),
                   ((seg[sX] + nodes[seg[sNo].astype(int), nX] * (seg[sBN].astype(int).astype(int) == 0)) * V_DX,
                    (seg[sY] + nodes[seg[sNo].astype(int), nY] * (seg[sBN].astype(int).astype(int) == 0)) * V_DX)],
                  fill=K_SECTION * seg[sR])

    pix = np.array(img)

    return pix


def update_r(segments, nodes, vegf):
    rows_s, columns_s = segments.shape
    rows_n, columns_n = nodes.shape
    enum_s = np.arange(rows_s)
    r_old = 0
    while np.absolute(np.amax(segments[:, sR]) - r_old) / np.amax(segments[:, sR]) > R_TOL:
        matrix_a = np.zeros((rows_s + rows_n, rows_s + rows_n))
        vector_b = np.zeros(rows_s + rows_n)

        matrix_a[enum_s[:], segments[:, sNi].astype(int)] = np.pi * np.power(segments[:, sR], 4) / (
                8 * MU0 * mu_rel(segments[:, sR], Hin) * segments[:, sL])
        matrix_a[enum_s[:], segments[:, sNo].astype(int)] += (-1 + np.sign(segments[:, sBN])) * np.pi * np.power(
            segments[:, sR], 4) / (8 * MU0 * mu_rel(segments[:, sR], Hin) * segments[:, sL])
        matrix_a[enum_s[:], rows_n + enum_s[:]] = -1
        matrix_a[rows_s + segments[:, sNi].astype(int), rows_n + enum_s[:]] = 1
        matrix_a[rows_s + segments[:, sNo].astype(int), rows_n + enum_s[:]] += -1 + np.sign(segments[:, sBN])
        vector_b[enum_s[:]] = segments[:, 2] * np.pi * np.power(segments[:, sR], 4) / (
                8 * MU0 * mu_rel(segments[:, sR], Hin) * segments[:, sL])
        x = np.linalg.solve(matrix_a, vector_b)

        nodes[:, nP] = x[:rows_n]
        segments[:, sQ] = np.abs(x[rows_n:] * 0.000000000001)

        segments[:, sDP] = np.absolute(
            nodes[segments[:, sNi].astype(int), nP] - nodes[segments[:, sNo].astype(int), nP] * (
                    segments[:, sBN] == 0) - np.absolute(segments[:, sBN]))

        # Calculates tau_w of the segment
        segments[:, sTw] = K_TW * segments[:, sDP] * segments[:, sR] / segments[:, sL]

        # Calculates km
        segments[:, sKm] = KM0 * (1 + KMV * vegf[
            nodes[segments[:, sNi].astype(int), nX].astype(int), nodes[segments[:, sNi].astype(int), nY].astype(
                int)] / (V0 + vegf[
            nodes[segments[:, sNi].astype(int), nX].astype(int), nodes[segments[:, sNi].astype(int), nY].astype(int)]))

        # Calculates Sm
        segments[:, sSm] = segments[:, sKm] * np.log10(Q_REF / (segments[:, sQ] * Hin) + 1)

        # Calculates tau
        segments[:, sTau] = 100 - 86 * np.exp(-5000 * np.power(np.log10(np.log10(PO2)), 5.4))

        # Calculates Sh
        segments[:, sSh] = np.log10(segments[:, sTw] + TAU_REF) - KP * np.log10(segments[:, sTau])

        # Calculates new radii
        r_old = np.amax(segments[:, sR])
        segments[:, sR] += EPS_T * segments[:, sR] * (segments[:, sSh] + segments[:, sSm] - KS)

    return segments, nodes


def calc_o2_cd(o20, dx, t, vessel, cells):
    """
    *Calculates the O2 distribution on the lattice grid-wise using forward euler scheme with central difference*

    o20, initial condition of the Oxygen distribution in the lattice size matrix
    dx, is the spatial delta for accuracy of the solution
    t, time in minutes of the simulation, the standard for O2 is 30 min
    cells, are the amount of cells in the lattice
    """
    dx2, dy2 = dx * dx, dx * dx
    dt = dx2 * dy2 / (2 * D[O2] * (dx2 + dy2))
    print(dt)
    nsteps = int(np.round(t / dt))
    print(nsteps)
    u0 = np.copy(o20)
    u = np.copy(u0)
    for i in range(nsteps):
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D[O2] * dt * (
                (u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[:-2, 1:-1]) / dx2
                + (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2]) / dy2)
        u = u + P[O2] * vessel * dt * (O2blood(vessel) - u0) + dt * KO2 * u0 * cells
        u0 = u.copy()
    return u


def test_function(x, y):
    print(x)
    print(x.__sizeof__())
    print(y)
    print(y.__sizeof__())
    return 0


def calc_o2_fvm(o20, dx, t, vessel, cells):
    """
    Calculates the O2 distribution on the lattice grid-wise
    :param o20: initial condition of the oxygen distribution in the lattice size matrix
    :param dx: Spatial delta
    :param t: time of the simulation in minutes
    :param vessel: Distribution of the vessels in lattice size matrix
    :param cells: cells density matrix
    :return:
    """

    # Preparing mesh of nx, ny size and dx step size
    ny, nx = np.shape(o20)
    dy = dx
    mesh = PeriodicGrid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    phi = CellVariable(name="O2",
                       mesh=mesh,
                       value=0.)

    # Calculate time step for CFL condition
    dt = CFL * dx ** 2 / D[O2]

    # Calculate number of steps for the simulation
    steps = math.ceil(t / dt)

    # Create variables of the PDE based on the created mesh
    nvessel = CellVariable(mesh=mesh, value=0., hasOld=1, name='vessels')
    nvessel.value = np.resize(vessel, nx * ny)
    ncells = CellVariable(mesh=mesh, value=0., hasOld=1, name='cells')
    ncells.value = np.resize(cells, nx * ny)

    # Create a time variable and dependant variable for the simulation
    time = Variable(0.)

    # Define initial conditions of the simulation
    phi.value = np.resize(o20, nx * ny)

    # Define the PDE equation for the solver
    eq = TransientTerm() == DiffusionTerm(coeff=D[O2]) + P[O2] * nvessel * O2kb + ImplicitSourceTerm(-P[O2] * nvessel + KO2 * ncells)

    # Run the solution steps times
    phi_old = 0

    for step in range(steps):
        eq.solve(var=phi, dt=dt)
        time.setValue(time.value + dt)
        error = np.absolute(phi - phi_old)
        phi_old = np.asarray(phi)
        if np.asarray(error).max() < 0.01:
            break

    return np.resize(phi, (nx, ny))


def calc_svegf(cells):
    result = KVEGF * np.copy(cells)
    return result


def calc_vegf_cd(vegf0, dx, t, vessel, cells):
    """
    *Calculates the VEGF distribution on the lattice grid-wise*

    o20, initial condition of the Oxygen distribution in the lattice size matrix
    dx, is the spatial delta for accuracy of the solution
    t, time in minutes of the simulation, the standard for O2 is 30 min
    cells, are the amount of cells in the lattice
    """
    dx2, dy2 = dx * dx, dx * dx
    # dt = dx2 * dy2 / (2 * D[VEGF] * (dx2 + dy2))
    dt = 0.025 * dx / D[VEGF]

    nsteps = int(np.round(t / dt))
    print(nsteps)
    u0 = np.copy(vegf0)
    u = np.copy(u0)

    for i in range(nsteps):
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D[VEGF] * dt * (
                (u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[:-2, 1:-1]) / dx2
                + (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2]) / dy2)
        u = u - P[VEGF] * vessel * dt * u0 - dt * Decay[VEGF] * u0 + dt * KVEGF * u0 * cells
        u0 = u.copy()
    return u


def calc_vegf_fvm(vegf0, dx, t, vessel, vegf_int):
    """
    Calculates the VEGF distribution on the lattice grid-wise

    :param vegf0: initial condition of the VEGF distribution in the lattice size matrix
    :param dx: Spatial delta
    :param t: Time of the simulation in minutes
    :param vessel: Distribution of the vessels in lattice size matrix
    :param vegf_int: Internal vegf of cells in mesh shape
    :return:
    """

    # Preparing mesh of nx, ny size and dx step size
    ny, nx = np.shape(vegf0)
    dy = dx
    mesh = PeriodicGrid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    phi = CellVariable(name="VEGF",
                       mesh=mesh,
                       value=0.)

    # Calculate time step for CFL condition
    dt = CFL * dx ** 2 / D[VEGF]

    # Calculate number of steps for the simulation
    steps = math.ceil(t / dt)

    # Create variables of the PDE based on the created mesh
    nvessel = CellVariable(mesh=mesh, value=1., hasOld=1, name='vessels')
    nvessel.value = np.resize(vessel, nx * ny)
    nvegf_int = CellVariable(mesh=mesh, value=1., hasOld=1, name='cells')
    nvegf_int.value = np.resize(vegf_int, nx * ny)

    # Create a time variable and dependant variable for the simulation
    time = Variable(0.)

    # Define initial conditions of the simulation
    phi.value = np.resize(vegf0, nx * ny)

    # Define the PDE equation for the solver
    eq = TransientTerm() == DiffusionTerm(coeff=D[VEGF]) + ImplicitSourceTerm(- P[VEGF] * nvessel) + \
         KVEGF * nvegf_int + ImplicitSourceTerm(-Decay[VEGF])

    # Run the solution steps times
    phi_old = 0
    for step in range(steps):
        eq.solve(var=phi, dt=dt)
        time.setValue(time.value + dt)
        error = phi - phi_old
        phi_old = np.asarray(phi)
        if np.asarray(error).max() < 0.001:
            break

    return np.resize(phi, (nx, ny))


"""
Blood flow and vessel radii adaptation functions definition
"""


def mu_rel(R, H):
    C_r = C(R)
    sol = (1 + (mu45(R) - 1) * ((1 - H) ** C_r - 1) / ((1 - 0.45) ** C_r - 1) * (2 * R / (2 * R - 1.1)) ** 2) * (
            2 * R / (2 * R - 1.1)) ** 2
    return sol


def mu45(R):
    return 6 * np.exp(-0.17 * R) + 3.2 - 2.44 * np.exp(-0.06 * np.power(2 * R, 0.645))


def C(R):
    return (0.8 + np.exp(-0.15 * R)) * (-1 + 1 / (1 + 10 ** (-11) * np.power((2 * R), 12))) + 1 / (
            1 + 10 ** (-11) * np.power((2 * R), 12))


def Length(P1, P2):
    return math.sqrt((P1[1] - P2[1]) ** 2 + (P1[0] - P2[0]) ** 2)


"""
Angiogenesis functions definition
"""


def closest_node(nodes, a_node):
    L = np.linalg.norm(nodes[:, :2] - a_node[:2], axis=1)
    ind = np.unravel_index(np.argmin(L, axis=None), L.shape)
    return ind[0]


def merge_tips(nodes, segments, a_nodes, tipcells, SPATIAL_DELTA):
    la0_nodes = a_nodes[a_nodes[:, nEID] == tipcells[0].astype(int)]
    la1_nodes = a_nodes[a_nodes[:, nEID] == tipcells[1].astype(int)]

    # Setting the starting node of the new blood vessel
    r_seg, c_seg = segments.shape
    r_nod, c_nod = nodes.shape
    r_nod_old = r_nod
    new_nodes1 = la0_nodes[1:, :3]
    new_nodes2 = la1_nodes[1:, :3]
    index_init = closest_node(nodes, la0_nodes[0])
    index_last = closest_node(nodes, la1_nodes[0])

    P1 = nodes[index_init, nP]
    P2 = nodes[index_last, nP]
    if P1 < P2:
        la0_nodes = a_nodes[a_nodes[:, nEID] == tipcells[1].astype(int)]
        la1_nodes = a_nodes[a_nodes[:, nEID] == tipcells[0].astype(int)]
        new_nodes1 = la0_nodes[1:, :3]
        new_nodes2 = la1_nodes[1:, :3]
        index_init = closest_node(nodes, la0_nodes[0])
        index_last = closest_node(nodes, la1_nodes[0])

    nodes = np.vstack((nodes, new_nodes1))
    nodes = np.vstack((nodes, new_nodes2[::-1]))
    r_nod, c_nod = nodes.shape
    new_segments = np.zeros((r_nod - r_nod_old, c_seg))
    new_segments[:, sNo] = np.arange(r_nod_old, r_nod)
    new_segments[:, sNi] = new_segments[:, sNo] - 1
    new_segments[0, sNi] = index_init
    new_segments[:, sR] = VESSEL_RADIUS
    new_segments[:, sL] = np.linalg.norm(
        nodes[new_segments[:, 0].astype(int), 0:2] - nodes[new_segments[:, 1].astype(int), 0:2],
        axis=1) * SPATIAL_DELTA
    segments = np.vstack((segments, new_segments))
    new_segments = np.zeros((1, c_seg))
    new_segments[0, sNo] = index_last
    new_segments[0, sNi] = segments[-1:, sNo]
    new_segments[0, sR] = VESSEL_RADIUS
    new_segments[0, sL] = np.linalg.norm(
        nodes[new_segments[0, 0].astype(int), 0:2] - nodes[new_segments[0, 1].astype(int), 0:2]) * SPATIAL_DELTA
    segments = np.vstack((segments, new_segments))

    return nodes, segments
