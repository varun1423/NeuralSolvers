import numpy as np
import openpmd_api as io


def data_set(path, iteration_id):
    """
    this function returns a dictionary with electron data and ion data for particular iteration.

    """
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
    """
    returns a list of all iterations available in the pic data

    """
    path = "D:/TUD/TU_Dresden/WiSe2021/Thesis/other/Data_s/004_KHI/simOutput/openPMD/simOutput_%06T.bp"
    series = io.Series(path, io.Access_Type.read_only)
    iterations = []
    for iter_id in series.iterations:
        iterations.append(iter_id)
    return iterations


def small_search_space(particle_position, particles__offset_x, particles__offset_y, particles__offset_z): #cell pos
    """
    it takes 7D data [cell postions X,Y,Z, position X,Y,Z, weighting] and cell positions for which we are calculating the density.
    and find all particles in (+2 cells and -2 cells) which may contribute to the cell density. 

    returns an array of neighboring particles

    """
    mask_x = (particle_position[:, 0] >= particles__offset_x - 2)
    output = particle_position[mask_x, :]
    mask_xx = (output[:, 0] <= particles__offset_x + 2)
    output = output[mask_xx, :]

    mask_y = (output[:, 1] >= particles__offset_y - 2)
    output = output[mask_y, :]
    mask_yy = (output[:, 1] <= particles__offset_y + 2)
    output = output[mask_yy, :]

    mask_z = (output[:, 2] >= particles__offset_z - 2)
    output = output[mask_z, :]
    mask_zz = (output[:, 2] <= particles__offset_z + 2)
    output = output[mask_zz, :]
    particle_position = output
    return particle_position


def distances_from_cell(cell_position_x, cell_position_y, cell_position_z,neighbouring_cells):
    """
    this takes neighboring particles as input 
    and 
    returns an array with particle distances in X,Y,Z  from the particular cell position for which we are finding the density. 
    
    """
    x_distance = neighbouring_cells[:,3]-cell_position_x
    y_distance = neighbouring_cells[:,4]-cell_position_y
    z_distance = neighbouring_cells[:,5]-cell_position_z
    x_distance = x_distance.reshape(x_distance.shape[0],1) 
    y_distance = y_distance.reshape(y_distance.shape[0],1)
    z_distance = z_distance.reshape(z_distance.shape[0],1)
    weights = neighbouring_cells[:,6].reshape(neighbouring_cells[:,6].shape[0],1)
    neighbouring_cells_distances = np.concatenate((x_distance,y_distance, z_distance, weights), axis=1)
    return neighbouring_cells_distances


def assignment_function_with_weights(neighbouring_cells_distances,idx_col):
    """
    this functions takes already calculated distances and correspponding weighting factor as input
    and 
    returns, weighting multiplied by assignment function value for X, Y, Z. idx_col parameter is used to control which (X,Y or Z) assignment function is calculated.
    
    ASSINMENT FUNCTION:
    /*       -
    *       |  3/4 - x^2                  if |x|<1/2
    * W(x)=<|  1/2*(3/2 - |x|)^2          if 1/2<=|x|<3/2
    *       |  0                          otherwise
    *       -
    */
    """
    cumulative_assignment_function = 0
    weighthing_factor = 0

    for k in range(0, len(neighbouring_cells_distances[:,0])):
        if abs(neighbouring_cells_distances[k,idx_col]) < 0.5:
            assignment_func = ((3 / 4) - (neighbouring_cells_distances[k,idx_col] ** 2))
            weighthing_factor += neighbouring_cells_distances[k,3]
        elif 0.5 <= abs(neighbouring_cells_distances[k,idx_col]) < 1.5:
            assignment_func = (1 / 2) * ((3 / 2) - abs(neighbouring_cells_distances[k,idx_col])) ** 2
            weighthing_factor += neighbouring_cells_distances[k,3]
        else:
            assignment_func = 0
        cumulative_assignment_function += assignment_func
    return cumulative_assignment_function*weighthing_factor






