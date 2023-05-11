import os
from python.optimisation_funcs import surface_points_normals, autodecoder, single_prediction

def input_prep (component, input_dir):
    for file in [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f[0] != "."]:
        points, normals, offsurface_points = surface_points_normals.generate(file)
        best_latent_vector = autodecoder.get_latent_vector(points, normals, offsurface_points, None, component)
        verts, faces = autodecoder.get_verts_faces(best_latent_vector, None, component)

        # if component == "bulkhead":
            
        # if component == "u-bending":

def target_prep (iput_dir):
    for file in [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f[0] != "."]:
        return