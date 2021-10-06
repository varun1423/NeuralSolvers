# %%
import numpy as np
import functions
import time

start_time = time.time()

#path = "D:/TUD/TU_Dresden/WiSe2021/Thesis/other/Data_s/004_KHI/simOutput/openPMD/simOutput_%06T.bp"
path = "/warm_archive/ws/s5960712-ml_currentDeposition/runs/004_KHI/simOutput/openPMD/simOutput_%06T.bp"

iterations_list = functions.iteration()
iterations_list = iterations_list[0:4]

for iter_id in iterations_list:
    data = functions.data_set(path, iter_id)

    for element in ['electron', 'ion']:
        particle_position = data[element]
        # break
        density = []
        for particle in range(0, len(particle_position[:, 0:3])):  # len(particle_position[:, 0:3])
            particles__offset_x = particle_position[particle, 0:3][0]
            particles__offset_y = particle_position[particle, 0:3][1]
            particles__offset_z = particle_position[particle, 0:3][2]

            neighbouring_cells = functions.small_search_space(particle_position, particles__offset_x,
                                                              particles__offset_y, particles__offset_z)

            x_distance = particles__offset_x - neighbouring_cells[:, 3]
            y_distance = particles__offset_y - neighbouring_cells[:, 4]
            z_distance = particles__offset_z - neighbouring_cells[:, 5]
            weighting = particle_position[particle, 6]

            x_density = functions.assignment_function(x_distance)
            y_density = functions.assignment_function(y_distance)
            z_density = functions.assignment_function(z_distance)
            x_density_w = x_density * weighting
            y_density_w = y_density * weighting
            z_density_w = z_density * weighting
            density.append([x_density, y_density, z_density, x_density_w, y_density_w, z_density_w])
            if particle % 25000 == 0:
                calculated_density = np.stack(density, axis=0)
                np.save('Calculated_Density/density_intermediate_' + element + '_' + str(iter_id), calculated_density)
        calculated_density = np.stack(density, axis=0)
        np.save('Calculated_Density/Calculated_density_' + element + '_' + str(iter_id), calculated_density)
    print('Density calculation completed for:{}'.format(iter_id))
time_taken = (time.time() - start_time) / 3600
print("time taken:{} hours".format(time_taken))
