import logging
import matplotlib.pyplot as plt
import numpy as np
from source import abm as abm
from source import environment as env
from source import varred as var
import time
import gc
import configparser as cp
import os
import pickle

# Initialization of configparser for reading config.ini
config = cp.ConfigParser()
config.read('default.ini')

# Simulation variables
start_time = time.time()  # Time Variable for execution time
STEPS = config['main'].getint('STEPS')
DT = config['main'].getint('DT')
SPATIAL_DELTA = config['main'].getfloat('SPATIAL_DELTA')

# Environmental simulation parameters
VESSEL_RADII_ADAPT = config.getint('main', 'VESSEL_RADII_ADAPT')
ANGIOGENESIS = config['main'].getint('ANGIOGENESIS')
RESTART_O2_CONDITIONS = config['main'].getint('RESTART_O2_CONDITIONS')
RESTART_VEGF_CONDITIONS = config['main'].getint('RESTART_VEGF_CONDITIONS')

# Cancer size and shape parameters
CANCER_X = config['main'].getint('CANCER_X')
CANCER_Y = config['main'].getint('CANCER_Y')
CANCER_SIZE_X = config['main'].getint('CANCER_SIZE_X')
CANCER_SIZE_Y = config['main'].getint('CANCER_SIZE_Y')
CANCER_DISTRIBUTION = config['main'].get('CANCER_DISTRIBUTION')
NORMAL_DISTRIBUTION = config['main'].get('NORMAL_DISTRIBUTION')

# Plotting parameters
SHOW_PLOT = config['main'].getint('SHOW_PLOT')
SAVE_PLOT = config['main'].getint('SAVE_PLOT')
SAVE_PATH = str(config['main'].get('SAVE_PATH'))
SAVE_NAME = str(config['main'].get('SAVE_NAME'))
SAVE_STEPS = config['main'].getint('SAVE_STEPS')

# Initialization of save path
i = 0
if SAVE_PLOT & (not os.path.exists(SAVE_PATH + i.__str__())):
    os.makedirs(SAVE_PATH + i.__str__())
else:
    while os.path.exists(SAVE_PATH + i.__str__()):
        i += 1
    os.makedirs(SAVE_PATH + i.__str__())
SAVE_PATH = SAVE_PATH + i.__str__() + '/'

logging.basicConfig(filename=SAVE_PATH + 'log.log', level=logging.INFO)
logging.info('Started at ' + str(time.time()))

# Initialization of lattice
lx = config['main'].getint('LX')
ly = config['main'].getint('LY')
lattice = np.zeros((lx, ly))
density_data = np.zeros((14, STEPS, lx, ly))

# Initialization of blood flow variables
segments = np.empty((0, 14))
nodes = np.empty((0, 3))

# Initialization of angiogenesis variables
a_segments = np.empty((0, 16))
a_nodes = np.empty((0, 5))
flag_geom_changed = 0
excluded_positions = np.empty((0, 9))

# Initialization of the environment
nv, segments, nodes = env.build_vessels(lattice, SPATIAL_DELTA, VESSEL_RADII_ADAPT)  # Position of the vessels
mcells = np.zeros((lx, ly))  # Lattice of the cell distribution
lcells = np.empty((0, 9))  # List of the cells [CTYPE,X,Y,PHI,P53,GAMMA,Z,VEGF,GEN]
lcells_stalk = np.empty((0, 9))  # List of stalk cells for angiogenesis

# Initialize Oxygen and vegf distribution in the environment
o2 = np.ones((lx, ly))*0
o2 = env.calc_o2_fvm(o2, SPATIAL_DELTA, DT, nv, mcells)
vegf = mcells

# Spawn of the cells in list form
q_cells = lcells.size / 9
lcells = abm.spawn_cells(lcells, abm.I_p[abm.Ncell], abm.Ncell,
                         NORMAL_DISTRIBUTION, 0, 0, lx, ly, np.zeros((abm.I_p[abm.Ncell], 9)))
new_cells = lcells.size / 9 - q_cells

if new_cells != 0:
    print(str(new_cells) + ' Normal cells were created')

q_cells = lcells.size / 9
lcells = abm.spawn_cells(lcells, abm.I_p[abm.Ccell], abm.Ccell,
                         CANCER_DISTRIBUTION, CANCER_X, CANCER_Y, CANCER_SIZE_X, CANCER_SIZE_Y,
                         np.zeros((abm.I_p[abm.Ccell], 9)))
new_cells = lcells.size / 9 - q_cells

if new_cells != 0:
    print(str(new_cells) + ' Cancer cells were created')

# Calculates Stochastic density Eq (25)
n1_stochastic = abm.Qi[abm.Ncell] * abm.cells_to_matrix(lcells[lcells[:, abm.P] == abm.Ncell, :], mcells)
n2_stochastic = abm.Qi[abm.Ccell] * abm.cells_to_matrix(lcells[lcells[:, abm.P] == abm.Ccell, :], mcells)
n3_stochastic = abm.Qi[abm.Ecell] * abm.cells_to_matrix(lcells[lcells[:, abm.P] == abm.Ecell, :], mcells)
np_stochastic = n1_stochastic + n2_stochastic + n3_stochastic

n1_controlled = n1_stochastic
n2_controlled = n2_stochastic
n3_controlled = n3_stochastic

n1_varred = n1_stochastic
n2_varred = n2_stochastic
n3_varred = n3_stochastic
np_varred = n1_varred + n2_varred + n3_varred

n1_coarse = n1_stochastic
n2_coarse = n2_stochastic
n3_coarse = n3_stochastic
np_coarse = n1_coarse + n2_coarse + n3_coarse

vegf_old = np.zeros((lx, ly))

for i in range(STEPS):

    # Updating vessels with sprouts from angiogenesis process
    if VESSEL_RADII_ADAPT == 1:
        segments, nodes = env.update_r(segments, nodes, vegf)
        nv = env.segments_to_nv(nv, segments, nodes, SPATIAL_DELTA)

    # Updates the environment
    if RESTART_O2_CONDITIONS == 1:
        o2 = np.zeros(o2.shape)

    if RESTART_VEGF_CONDITIONS == 1:
        vegf = np.zeros(vegf.shape)

    o2 = env.calc_o2_fvm(o2, SPATIAL_DELTA, DT, nv, np_coarse)
    vegf_int = abm.cells_to_matrix(lcells[lcells[:, abm.VEGF_INT] >= abm.VEGF_thr[lcells[:, abm.P].astype(int)], :], np_coarse)

    #vegf_cell = abm.cells_to_matrix(lcells[lcells[:, abm.VEGF_INT] >= abm.VEGF_thr[lcells[:, abm.P].astype(int)], :]*abm.Qi[lcells[:, abm.P].astype(int)], np_coarse)

    if np.amax(vegf_int) != 0:
        vegf = env.calc_vegf_fvm(vegf, SPATIAL_DELTA, DT, nv + n3_stochastic, vegf_int)

    # Angionesis process evolution
    if ANGIOGENESIS == 1:
        q_cells = lcells.size / 9
        lcells, excluded_positions = abm.angiogenesis(lcells, excluded_positions, nv, vegf, DT, a_nodes)

        sprout_cells = lcells.size / 9 - q_cells
        if sprout_cells != 0:
            print(str(sprout_cells) + ' sprouts were created')
            logging.info(str(sprout_cells) + ' sprouts were created')


    Ecells = lcells[lcells[:, abm.P] == abm.Ecell, :]
    if ANGIOGENESIS == 1:
        rows, columns = np.shape(Ecells)
        if rows != 0:
            a_nodes_counter = 0
            if a_nodes.size != 0:
                a_nodes_counter = np.amax(a_nodes[:, env.nID])
            new_a_segments = np.zeros((rows, 16))
            new_a_nodes = np.zeros((rows, 5))
            new_a_nodes[:, env.nX] = Ecells[:, abm.Y]
            new_a_nodes[:, env.nY] = Ecells[:, abm.X]
            new_a_nodes[:, env.nID] = a_nodes_counter + np.arange(rows) + 1
            new_a_nodes[:, env.nEID] = Ecells[:, abm.ID]
            a_nodes = np.vstack((a_nodes, new_a_nodes))
            k = 0
            a_segments_counter = 0
            if a_segments.size != 0:
                a_segments_counter = np.amax(a_segments[:, env.sID])
            for stalk in Ecells:
                la_nodes = a_nodes[a_nodes[:, env.nEID] == stalk[abm.ID], :]
                rows_la, columns_la = la_nodes.shape
                new_a_segments[k, env.sNo] = la_nodes[rows_la - 1, env.nID]
                if rows_la > 1:
                    new_a_segments[k, env.sNi] = la_nodes[rows_la - 2, env.nID]
                else:
                    sprout_position = (stalk[abm.Y], stalk[abm.X])
                    new_a_segments[k, env.sNi] = -env.closest_node(nodes, sprout_position)
                new_a_segments[k, env.sID] = k + a_segments_counter
                new_a_segments[k, env.sEID] = stalk[abm.ID]
                k += 1
            a_segments = np.vstack((a_segments, new_a_segments))

    # ABM loop
    dt_stochastic = 1.0
    # for j in range(int(dt / dt_stochastic)):

    # Evolves position of the cells
    lcells = abm.update_position(lcells, DT, SPATIAL_DELTA, lx, ly, vegf, mcells, ANGIOGENESIS)

    # Anastomosis process
    if ANGIOGENESIS == 1 and Ecells.size != 0:

        # Computes tip cells prunning timer
        lcells = abm.tip_prunning(lcells, DT)

        # lcells, nodes, segments = abm.anastomosis(lcells, nodes, segments, a_nodes, a_segments)
        r, c = Ecells.shape
        L = np.zeros((r, r))
        for j in range(r):
            for k in range(r):
                L[j, k] = np.linalg.norm(Ecells[j, abm.X:abm.PHI] - Ecells[k, abm.X:abm.PHI])
        L += np.eye(r, dtype=int)*10
        if np.amin(L) < np.amax(segments[:, 5] * 0.0001 / .004):
            ind = np.unravel_index(np.argmin(L, axis=None), L.shape)
            ind = (Ecells[ind[0], abm.ID].astype(int), Ecells[ind[1], abm.ID].astype(int))

            nodes, segments = env.merge_tips(nodes, segments, a_nodes, ind, SPATIAL_DELTA)

            #a_segments = a_segments[a_segments[:, env.sEID] != ind[0].astype(int)]
            #a_segments = a_segments[a_segments[:, env.sEID] != ind[1].astype(int)]
            #a_nodes = a_nodes[a_nodes[:, env.nEID] != ind[0].astype(int)]
            #a_nodes = a_nodes[a_nodes[:, env.nEID] != ind[1].astype(int)]
            Ecells = lcells[lcells[:, abm.P] == abm.Ecell, :]
            lcells = lcells[lcells[:, abm.P] != abm.Ecell, :]
            Ecells = Ecells[Ecells[:, abm.ID] != ind[0].astype(int), :]
            Ecells = Ecells[Ecells[:, abm.ID] != ind[1].astype(int), :]
            lcells = np.vstack((Ecells, lcells))
            print("new blood vessel was introduced")
            flag_geom_changed = 1

    # Calculates Coarse density Eq (21)
    n1_coarse = abm.Qi[abm.Ncell] * abm.cells_to_matrix(lcells[lcells[:, abm.P] == abm.Ncell, :], np_stochastic)
    n2_coarse = abm.Qi[abm.Ccell] * abm.cells_to_matrix(lcells[lcells[:, abm.P] == abm.Ccell, :], np_stochastic)
    n3_coarse = abm.Qi[abm.Ecell] * abm.cells_to_matrix(lcells[lcells[:, abm.P] == abm.Ecell, :], np_stochastic)
    np_coarse = n1_coarse + n2_coarse + n3_coarse

    for j in range(int(DT / dt_stochastic)):

        # Cell division process
        q_cells = lcells.size / 9
        lcells = abm.update_cell_division(lcells, o2, dt_stochastic)

        # Calculates quantity of cells divided
        div_cells = lcells.size / 9 - q_cells
        if div_cells != 0:
            print(str(div_cells) + ' cells were divided')
            logging.info(str(div_cells) + ' cells were divided')

        q_cells = lcells.size / 9
        lcells = abm.update_intracellular(lcells, o2, dt_stochastic)
        lcells = abm.update_apoptosis(lcells, o2, dt_stochastic, np_stochastic)

        # Calculate quantity of killed cells
        killed_cells = q_cells - lcells.size / 9
        if killed_cells != 0:
            print(str(killed_cells) + ' cells were killed')
            logging.info(str(killed_cells) + ' cells were killed')

    # Preparing densities for Variance reduction algorithm
    n1_stochastic = abm.Qi[abm.Ncell] * abm.cells_to_matrix(lcells[lcells[:, abm.P] == abm.Ncell, :], np_stochastic)
    n2_stochastic = abm.Qi[abm.Ccell] * abm.cells_to_matrix(lcells[lcells[:, abm.P] == abm.Ccell, :], np_stochastic)
    n3_stochastic = abm.Qi[abm.Ncell] * abm.cells_to_matrix(lcells[lcells[:, abm.P] == abm.Ecell, :], np_stochastic)
    np_stochastic = n1_stochastic + n2_stochastic + n3_stochastic

    # Calculates the advection - diffusion equation (13)
    n1_diffusion = var.calc_density_fvm(n1_coarse, 0.004, DT, abm.Dp[abm.Ncell],
                                        abm.Xp[abm.Ncell], abm.np_max[abm.Ncell], vegf - vegf_old)
    n2_diffusion = var.calc_density_fvm(n2_coarse, 0.004, DT, abm.Dp[abm.Ccell],
                                        abm.Xp[abm.Ccell], abm.np_max[abm.Ccell], vegf - vegf_old)
    n3_diffusion = var.calc_density_fvm(n3_coarse, 0.004, DT, abm.Dp[abm.Ccell],
                                        abm.Xp[abm.Ccell], abm.np_max[abm.Ccell], vegf - vegf_old)
    np_diffusion = n1_diffusion + n2_coarse + n3_diffusion
    vegf_old = vegf

    # Calculates controlled
    n1_controlled = n1_varred * np.exp(-DT * n1_diffusion)
    n2_controlled = n2_varred * np.exp(-DT * n2_diffusion)
    n3_controlled = n2_varred * np.exp(-DT * n3_diffusion)

    # Calculates Variance reduced density
    n1_varred = n1_varred * np.exp(-DT * n1_diffusion) + n1_stochastic - n1_coarse
    n2_varred = n2_varred * np.exp(-DT * n1_diffusion) + n2_stochastic - n2_coarse
    n3_varred = n3_varred * np.exp(-DT * n1_diffusion) + n3_stochastic - n3_coarse
    np_varred = n1_varred + n2_varred + n3_varred

    print(' %s seconds lapsed, %s min of simulation' % ((time.time() - start_time), DT * i))
    logging.info(' %s seconds lapsed, %s min of simulation' % ((time.time() - start_time), DT * i))

    # Save all result arrays to file
    density_data[0, i, :, :] = n1_stochastic
    density_data[1, i, :, :] = n2_stochastic
    density_data[2, i, :, :] = n3_stochastic
    density_data[3, i, :, :] = n1_coarse
    density_data[4, i, :, :] = n2_coarse
    density_data[5, i, :, :] = n3_coarse
    density_data[6, i, :, :] = n1_diffusion
    density_data[7, i, :, :] = n2_diffusion
    density_data[8, i, :, :] = n3_diffusion
    density_data[9, i, :, :] = n1_varred
    density_data[10, i, :, :] = n2_varred
    density_data[11, i, :, :] = n3_varred
    density_data[12, i, :, :] = o2
    density_data[13, i, :, :] = vegf

    # Heatmaps for data visualization
    if (SHOW_PLOT or ((SAVE_PLOT == 1) & (0 == (i % SAVE_STEPS))) or (i == (STEPS - 1))) and ANGIOGENESIS == 0:
        # Closes all previous figures for memory clearing purposes
        plt.close('all')
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        im1 = axs[0, 0].imshow(np.transpose(o2), cmap='Reds', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[0, 0].title.set_text(r'$O_2(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        axs[0, 0].set_xlabel('Length in cm')
        axs[0, 0].set_ylabel('Width in cm')
        fig.colorbar(im1, ax=axs[0, 0], label="mmHg")

        im2 = axs[0, 1].imshow(np.transpose(np_stochastic), cmap='Greys', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[0, 1].title.set_text(r'$n_p(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        axs[0, 1].set_xlabel('Length in cm')
        axs[0, 1].set_ylabel('Width in cm')
        fig.colorbar(im2, ax=axs[0, 1], label="number of cells")

        im3 = axs[1, 0].imshow(np.transpose(n1_stochastic), cmap='Greys', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[1, 0].title.set_text(r'$n_1(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        axs[1, 0].set_xlabel('Length in cm')
        axs[1, 0].set_ylabel('Width in cm')
        fig.colorbar(im2, ax=axs[1, 0], label="number of cells")

        im4 = axs[1, 1].imshow(np.transpose(n2_stochastic), cmap='Greys', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[1, 1].title.set_text(r'$n_2(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        axs[1, 1].set_xlabel('Length in cm')
        axs[1, 1].set_ylabel('Width in cm')
        fig.colorbar(im2, ax=axs[1, 1], label="number of cells")

    if (SHOW_PLOT or ((SAVE_PLOT == 1) & (0 == (i % SAVE_STEPS))) or (i == (STEPS - 1))) and ANGIOGENESIS == 1:
        # Closes all previous figures for memory clearing purposes
        plt.close('all')
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        im1 = axs[0, 0].imshow(np.transpose(o2), cmap='Reds', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[0, 0].title.set_text(r'$O_2(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        fig.colorbar(im1, ax=axs[0, 0])

        im2 = axs[0, 1].imshow(np.transpose(vegf), cmap='Greys', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[0, 1].title.set_text(r'$V(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        fig.colorbar(im2, ax=axs[0, 1])

        im3 = axs[0, 2].imshow(np.transpose(np_stochastic), cmap='Greys', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[0, 2].title.set_text(r'$n_p(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        fig.colorbar(im3, ax=axs[0, 2])

        im4 = axs[1, 0].imshow(np.transpose(n3_stochastic), cmap='Greys', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[1, 0].title.set_text(r'$n_3(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        fig.colorbar(im4, ax=axs[1, 0])

        im5 = axs[1, 1].imshow(np.transpose(n1_stochastic), cmap='Greys', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[1, 1].title.set_text(r'$n_1(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        fig.colorbar(im5, ax=axs[1, 1])

        im6 = axs[1, 2].imshow(np.transpose(n2_stochastic), cmap='Greys', interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs[1, 2].title.set_text(r'$n_2(x, t=' + str(int((i + 1) * DT)) + ' min)$')
        fig.colorbar(im6, ax=axs[1, 2])


    if (SAVE_PLOT == 1) & ((0 == (i % SAVE_STEPS)) or (i == (STEPS - 1))):
        if not os.path.exists(SAVE_PATH + SAVE_NAME + str(i) + '/'):
            os.makedirs(SAVE_PATH + SAVE_NAME + str(i) + '/')
        plt.savefig(SAVE_PATH + SAVE_NAME + str(i) + '_O2.png')

        # Save plots information in csv files
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/O2.csv', o2, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/vegf.csv', vegf, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n1.csv', n1_stochastic, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n2.csv', n2_stochastic, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n3.csv', n3_stochastic, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/np.csv', np_stochastic, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n1_varred.csv', n1_varred, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n2_varred.csv', n2_varred, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n3_varred.csv', n3_varred, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/np_varred.csv', np_varred, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n1_diffusion.csv', n1_diffusion, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n2_diffusion.csv', n2_diffusion, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n3_diffusion.csv', n3_diffusion, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/np_diffusion.csv', np_diffusion, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n1_coarse.csv', n1_coarse, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n2_coarse.csv', n2_coarse, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/n3_coarse.csv', n3_coarse, delimiter=';')
        np.savetxt(SAVE_PATH + SAVE_NAME + str(i) + '/np_coarse.csv', np_coarse, delimiter=';')

    if SHOW_PLOT == 1:
        plt.show()

    if ANGIOGENESIS == 1 and flag_geom_changed == 1:
        flag_geom_changed = 0
        fig, axs = plt.subplots(1, 1, figsize=(9.6, 7.2))
        im1 = axs.imshow(np.transpose(env.get_nv_geometry(nv,segments,nodes,SPATIAL_DELTA)), interpolation='nearest',
                               extent=[0, ly * SPATIAL_DELTA, 0, lx * SPATIAL_DELTA])
        axs.title.set_text(r'$n_v(x, t=' + str(int(i * DT)) + ' min)$')
        fig.colorbar(im1, ax=axs)
        plt.savefig(SAVE_PATH + SAVE_NAME + str(i) + '_nv.png')

    # Release memory resources
    gc.collect()
    logging.info('--- Data saved at step %s ---' % i)

density_handler = open(SAVE_PATH + '/density_data.obj', 'wb')
pickle.dump(density_data, density_handler)
density_handler.close()

# Control execution time and finished indicator
print('--- %s seconds ---' % (time.time() - start_time))
logging.info('--- %s seconds ---' % (time.time() - start_time))
print('done!')
logging.info('done!' + str(time.time()))
