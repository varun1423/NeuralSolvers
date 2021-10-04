# %%
import numpy as np
import functions
from tqdm import tqdm

iteration_id = 2000
path = "D:/TUD/TU_Dresden/WiSe2021/Thesis/other/Data_s/004_KHI/simOutput/openPMD/simOutput_%06T.bp"
particle_position = functions.data_set(path, iteration_id)

# %%
# initialize empty list
dicts = {}
density = []
print('for loop started...')

for particle in range(0, len(particle_position[:, 0:3])):
    to_list = []
    particles__offset_x = particle_position[particle, 0:3][0]
    particles__offset_y = particle_position[particle, 0:3][1]
    particles__offset_z = particle_position[particle, 0:3][2]
    small_search = functions.small_search_space(particle_position, particles__offset_x, particles__offset_y,
                                                particles__offset_z)
    for i in range(0, len(small_search[:, 0:3])):
        for y in [particles__offset_y, particles__offset_y - 1, particles__offset_y + 1]:
            '''
            for particular cell get all the neighbour in  X and Z
            '''
            rows = np.where(
                (
                        ((small_search[i, 0:3][0] == particles__offset_x - 1) or (
                                    small_search[i, 0:3][0] == particles__offset_x + 1) or (
                                     small_search[i, 0:3][0] == particles__offset_x))
                        and
                        (small_search[i, 0:3][1] == y)
                        and
                        (small_search[i, 0:3][2] == particles__offset_z)
                )
                or
                (
                        ((small_search[i, 0:3][0] == particles__offset_x))
                        and
                        (small_search[i, 0:3][1] == y)
                        and
                        ((small_search[i, 0:3][2] == particles__offset_z - 1) or (
                                    small_search[i, 0:3][2] == particles__offset_z + 1))
                )
                or
                (
                        ((small_search[i, 0:3][0] == particles__offset_x + 1))
                        and
                        (small_search[i, 0:3][1] == y)
                        and
                        ((small_search[i, 0:3][2] == particles__offset_z - 1) or (
                                    small_search[i, 0:3][2] == particles__offset_z + 1))
                )
                or
                (
                        ((small_search[i, 0:3][0] == particles__offset_x - 1))
                        and
                        (small_search[i, 0:3][1] == y)
                        and
                        ((small_search[i, 0:3][2] == particles__offset_z - 1) or (
                                    small_search[i, 0:3][2] == particles__offset_z + 1))
                ),
                small_search[i, 0:], 0)
            to_list.append(rows)

    neighbouring_cells = np.vstack(to_list)  # convert list to array
    neighbouring_cells = neighbouring_cells[np.any(neighbouring_cells != 0, axis=1)]  # drop irrelevant zero elements
    neighbouring_cells = np.unique(neighbouring_cells[:, 0:], axis=0)
    dicts[particle] = neighbouring_cells
    #bar.set_postfix({'particle number': particle, 'has neighbouring cells': neighbouring_cells.shape[0]})
    del to_list

    x_distance = particles__offset_x - dicts[particle][:, 3]
    y_distance = particles__offset_y - dicts[particle][:, 4]
    z_distance = particles__offset_z - dicts[particle][:, 5]

    weighting = particle_position[particle, 6]
    x_density = functions.assignment_function(x_distance)
    y_density = functions.assignment_function(y_distance)
    z_density = functions.assignment_function(z_distance)
    xw_density = weighting * functions.assignment_function(x_distance)
    yw_density = weighting * functions.assignment_function(y_distance)
    zw_density = weighting * functions.assignment_function(z_distance)
    density.append([x_density, y_density, z_density, xw_density, yw_density, zw_density])
    if particle%25000 == 0:
        calculated_density = np.stack(density, axis=0)
        np.save(f'density_intermediate' + str(iteration_id), calculated_density)
        print('')
        print(f'Intermediate file saved at particle: {particle}')
calculated_density = np.stack(density, axis=0)
print("")
print(f'completed, the shape of density is = {calculated_density.shape}')
print(f'Saving *.npy file...')
np.save(f'density_' + str(iteration_id), calculated_density)
