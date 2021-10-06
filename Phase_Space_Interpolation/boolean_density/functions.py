import numpy as np
import openpmd_api as io


def data_set(path, iteration_id=2000):
    # path = "D:/TUD/TU_Dresden/WiSe2021/Thesis/Data_s/004_KHI/simOutput/openPMD/simOutput_%06T.bp"
    series = io.Series(path, io.Access_Type.read_only)

    i = series.iterations[iteration_id]
    electrons = i.particles['e']
    ions = i.particles['i']
    position_x = electrons["position"]["x"].load_chunk()
    position_y = electrons["position"]["y"].load_chunk()
    position_z = electrons["position"]["z"].load_chunk()
    positionOffset_x = electrons["positionOffset"]["x"].load_chunk()
    positionOffset_y = electrons["positionOffset"]["y"].load_chunk()
    positionOffset_z = electrons["positionOffset"]["z"].load_chunk()
    weighting = electrons["weighting"][io.Mesh_Record_Component.SCALAR].load_chunk()
    # ions
    ions_position_x = ions["position"]["x"].load_chunk()
    ions_position_y = ions["position"]["y"].load_chunk()
    ions_position_z = ions["position"]["z"].load_chunk()
    ions_positionOffset_x = ions["positionOffset"]["x"].load_chunk()
    ions_positionOffset_y = ions["positionOffset"]["y"].load_chunk()
    ions_positionOffset_z = ions["positionOffset"]["z"].load_chunk()
    ions_weighting = ions["weighting"][io.Mesh_Record_Component.SCALAR].load_chunk()
    series.flush()

    electron_position = np.stack(
        (positionOffset_x, positionOffset_y, positionOffset_z, positionOffset_x + position_x,
         positionOffset_y + position_y,
         positionOffset_z + position_z,
         weighting), axis=1)
    ions_position = np.stack(
        (ions_positionOffset_x, ions_positionOffset_y, ions_positionOffset_z, ions_positionOffset_x + ions_position_x,
         ions_positionOffset_y + ions_position_y,
         ions_positionOffset_z + ions_position_z,
         ions_weighting), axis=1)
    data_dict = {'electron': electron_position, 'ion': ions_position}
    return data_dict

def iteration():
    path = "D:/TUD/TU_Dresden/WiSe2021/Thesis/other/Data_s/004_KHI/simOutput/openPMD/simOutput_%06T.bp"
    series = io.Series(path, io.Access_Type.read_only)
    iterations = []
    for iter_id in series.iterations:
        iterations.append(iter_id)
    return iterations

def assignment_function(distance):
    cumulative_assignment_function = 0
    for k in range(0, len(distance)):
        if abs(distance[k]) < 0.5:
            assignment_func = ((3 / 4) - (distance[k] ** 2))
        elif 0.5 < abs(distance[k]) < 1.5:
            assignment_func = (1 / 2) * ((3 / 2) - abs(distance[k])) ** 2
        else:
            assignment_func = 0
    cumulative_assignment_function = cumulative_assignment_function + assignment_func
    return cumulative_assignment_function


def small_search_space(particle_position, particles__offset_x, particles__offset_y, particles__offset_z):
    mask_x = (particle_position[:, 0] >= particles__offset_x - 1)
    output = particle_position[mask_x, :]
    mask_xx = (output[:, 0] <= particles__offset_x + 1)
    output = output[mask_xx, :]

    mask_y = (output[:, 1] >= particles__offset_y - 1)
    output = output[mask_y, :]
    mask_yy = (output[:, 1] <= particles__offset_y + 1)
    output = output[mask_yy, :]

    mask_z = (output[:, 2] >= particles__offset_z - 1)
    output = output[mask_z, :]
    mask_zz = (output[:, 2] <= particles__offset_z + 1)
    output = output[mask_zz, :]
    particle_position = output
    return particle_position

