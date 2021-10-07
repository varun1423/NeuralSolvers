# %%
import numpy as np
import functions
import time

start_time = time.time()

path = "D:/TUD/TU_Dresden/WiSe2021/Thesis/other/Data_s/004_KHI/simOutput/openPMD/simOutput_%06T.bp"
#path = "/warm_archive/ws/s5960712-ml_currentDeposition/runs/004_KHI/simOutput/openPMD/simOutput_%06T.bp"
outpath = 'Calculated_Density/Calculated_density_'

iterations_list = functions.iteration()
iter_id = iterations_list[-1]
data = functions.data_set(path, iter_id)

particle_position = data['electron']
cells = particle_position[:,0:3] 
unique_cells = np.unique(cells, axis=0) #get all the cells unique cells

density = []
for cell in range(0,len(unique_cells[:,0:3])):#unique_cells[:,0:3]
    cell_position_x = unique_cells[cell, 0:3][0]
    cell_position_y = unique_cells[cell, 0:3][1]
    cell_position_z = unique_cells[cell, 0:3][2]
#calculates neighboring cells
    neighbouring_cells = functions.small_search_space(particle_position, cell_position_x, cell_position_y, cell_position_z)
# calculates distances in X, Y Z direction from the cell position
    neighbouring_cells_distances = functions.distances_from_cell(cell_position_x, cell_position_y, cell_position_z, neighbouring_cells)
#calculates assignment function, it is already multiplied by weights 
    x_component = functions.assignment_function_with_weights(neighbouring_cells_distances,0)
    y_component = functions.assignment_function_with_weights(neighbouring_cells_distances,1)
    z_component = functions.assignment_function_with_weights(neighbouring_cells_distances,2)
#density at cell
    density_at_cell = x_component * y_component * z_component
    density.append([density_at_cell])
calculated_density = np.stack(density, axis=0)
to_save = np.concatenate((unique_cells[0:cell+1,:], calculated_density), axis=1)
np.save( outpath + str(iter_id) , to_save)

time_taken = (time.time() - start_time) / 3600
print("time taken:{} hours".format(time_taken))
# %%