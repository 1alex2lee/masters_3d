import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from random import randint
from scipy.optimize import minimize

# Data structures and functions for rendering
from pytorch3d import ops
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)

from pyqtgraph.opengl import MeshData, GLMeshItem
from pyqtgraph import AxisItem, GradientEditorItem

from python.optimisation_funcs import manufacturingSurrogateModels_bulkhead
from python.optimisation_funcs import manufacturingSurrogateModels_ubending

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#hyperparameters surrogate model
batch_size = 4
num_channel_thinning = (np.array([4,8,16,32,64,128,256,512])).astype(np.int64)

#load trained NN3 manufacturing constraints surrogate models
#load trained model
thinningModel = manufacturingSurrogateModels_ubending.ResUNet_Thinning(num_channel_thinning,batch_size)
thinningModel = thinningModel.to(device)
thinningModel.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/ResSEUNet_512_B4_2000_COS0.0_LRFix0.0002_E4B6D4_NewShape_08Feb23_best.pkl",map_location=device))
thinningModel.eval()

loadedInputForDisplacementModelImages_original = np.load("python/optimisation_funcs/model_confirugrations/u-bending/ModelPreparation/NN2_ManufacturingSurrogate/UBending_models_newgeo/InputTestOriginalAndNN_Feb23.npy") #for blank shape 
sampleNo = randint(0, len(loadedInputForDisplacementModelImages_original)) #initial design
loadedInputForDisplacementModelImages = loadedInputForDisplacementModelImages_original[sampleNo].copy()
surrogateModelInput = torch.tensor(loadedInputForDisplacementModelImages).float().to(device).unsqueeze(0)
gridOfOnes = torch.ones_like(surrogateModelInput.squeeze()[2])

def get_max_thinning (BHF, friction, clearance, thickness):
    surrogateModelInput[:, 2, :, :] = BHF * gridOfOnes #BHF
    surrogateModelInput[:, 3, :, :] = friction * gridOfOnes #friction
    surrogateModelInput[:, 4, :, :] = clearance * gridOfOnes #clearance
    surrogateModelInput[:, 5, :, :] = thickness * gridOfOnes #thickness

    thinningField = thinningModel(surrogateModelInput)[..., :-10].detach().numpy()
    thinningField *= 100

    return thinningField.max()

def monotonic_line_of_best_fit(x, y):
    # Sort the points based on x-coordinate
    sorted_points = sorted(zip(x, y), key=lambda p: p[0])
    x_sorted, y_sorted = zip(*sorted_points)
    
    # Define the objective function to minimize
    def objective(coefficients):
        a, b = coefficients
        y_predicted = a * np.array(x_sorted) + b
        residuals = np.array(y_sorted) - y_predicted
        return np.sum(residuals**2)
    
    # Define the monotonic constraint
    def constraint(coefficients):
        a, _ = coefficients
        return a
    
    # Perform the optimization
    initial_guess = [0.0, 0.0]
    bounds = [(None, None), (None, None)]
    constraints = [{'type': 'ineq', 'fun': constraint}]
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    
    # Extract the optimal coefficients
    a_opt, b_opt = result.x
    
    # Generate the points on the line of best fit
    x_fit = np.linspace(min(x_sorted), max(x_sorted), num=len(x_sorted))
    y_fit = a_opt * x_fit + b_opt
    
    return x_fit, y_fit

def load (component, var1, var2, window, bhf, friction, clearance, thickness):
    if "u-bending" in component.lower():
        print(component, var1, var2, "sensitivity requested")

        if "max thinning" in var2.lower():
            num_steps = 20
            clip = 0.2 # clip either side of bound
            # Set variable boundaries
            maxBHF = 59
            maxFriction = 0.199
            maxClearance = 1.49
            maxThickness = 2.99

            minBHF = 5.2
            minFriction = 0.1
            minClearance = 1.1
            minThickness = 0.51

            # midBHF = minBHF + (maxBHF-minBHF)/2
            # midFriction = minFriction + (maxFriction-minFriction)/2
            # midClearance = minClearance + (maxClearance-minClearance)/2
            # midThickness = minThickness + (maxThickness-minThickness)/2

            if "blank holding force" in var1.lower():
                var_linspace = np.linspace(minBHF + (maxBHF-minBHF)*clip, maxBHF - (maxBHF-minBHF)*clip, num_steps)
                max_thinnings = []
                for v in var_linspace:
                    max_thinnings.append(get_max_thinning(v, friction, clearance, thickness))
                    # print(v, "bhf thinning added")
            if "friction" in var1.lower():
                var_linspace = np.linspace(minFriction + (maxFriction-minFriction)*clip, maxFriction - (maxFriction-minFriction)*clip, num_steps)
                max_thinnings = []
                for v in var_linspace:
                    max_thinnings.append(get_max_thinning(bhf, v, clearance, thickness))
            if "clearance" in var1.lower():
                var_linspace = np.linspace(minClearance + (maxClearance-minClearance)*clip, maxClearance - (maxClearance-minClearance)*clip, num_steps)
                max_thinnings = []
                for v in var_linspace:
                    max_thinnings.append(get_max_thinning(bhf, friction, v, thickness))
            if "thickness" in var1.lower():
                var_linspace = np.linspace(minThickness + (maxThickness-minThickness)*clip, maxThickness - (maxThickness-minThickness)*clip, num_steps)
                max_thinnings = []
                for v in var_linspace:
                    max_thinnings.append(get_max_thinning(bhf, maxFriction, clearance, v))
                    
            x, y = monotonic_line_of_best_fit(var_linspace, max_thinnings)
            window.canvas.axes.clear()
            window.canvas.axes.plot(x, y)
            window.canvas.axes.set_xlabel(var1)
            window.canvas.axes.set_ylabel(var2)
            window.canvas.axes.set_ylim(0,20)
            # window.canvas.axes.legend(["Chamfer Loss", "Height Loss","Manufacturing Constraint Loss", "Similarity Loss"])
            window.canvas.axes.set_title(f"Change of {var2} over {var1}")
            window.canvas.draw()
            print("graph plotted")