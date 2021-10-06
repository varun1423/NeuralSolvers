# %%
import numpy as np
import functions
import time
start_time = time.time()

iteration_id = 2000
path = "/warm_archive/ws/s5960712-ml_currentDeposition/runs/004_KHI/simOutput/openPMD/simOutput_%06T.bp"
particle_position = functions.data_set(path, iteration_id)

# %%
# initialize empty list
dicts = {}
density = []

for particle in range(0, len(particle_position[:, 0:3])): 
    particles__offset_x = particle_position[particle,0:3][0]
    particles__offset_y = particle_position[particle,0:3][1]
    particles__offset_z = particle_position[particle,0:3][2]
    
    neighbouring_cells = functions.small_search_space(particle_position, particles__offset_x,particles__offset_y,particles__offset_z)
    dicts[particle] = neighbouring_cells

    x_distance = particles__offset_x-dicts[particle][:,3]
    y_distance = particles__offset_y-dicts[particle][:,4]
    z_distance = particles__offset_z-dicts[particle][:,5]
    weighting = particle_position[particle,6]

    x_density = functions.assignment_function(x_distance)
    y_density = functions.assignment_function(y_distance)
    z_density = functions.assignment_function(z_distance)
    x_density_w = x_density * weighting 
    y_density_w = y_density * weighting
    z_density_w = z_density * weighting
    density.append([x_density, y_density, z_density, x_density_w, y_density_w, z_density_w])
    if particle%5000 == 0:
        print('')
        print('at particle:'.format(particle))
    if particle%25000 == 0:
        calculated_density = np.stack(density, axis=0)
        np.save('density_intermediate',calculated_density)
        print('')
        print('Intermediate file saved at particle'.format(particle))
calculated_density = np.stack((density),axis=0)
np.save('density_intermediate',calculated_density)

time_taken = (time.time() - start_time)/3600
print("time taken:{}".format(time_taken))
