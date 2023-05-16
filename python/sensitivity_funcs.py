import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d

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

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load (component, var1, var2, window):
    if component == "u-bending":
        print(component, var1, var2, "sensitivity requested")
        #hyperparameters surrogate model
        batch_size = 4
        num_channel_thinning = (np.array([4,8,16,32,64,128,256,512])).astype(np.int64)

        #load trained NN3 manufacturing constraints surrogate models
        #load trained model
        thinningModel = manufacturingSurrogateModels_ubending.ResUNet_Thinning(num_channel_thinning,batch_size)
        thinningModel = thinningModel.to(device)
        thinningModel.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/ResSEUNet_512_B4_2000_COS0.0_LRFix0.0002_E4B6D4_NewShape_08Feb23_best.pkl",map_location=device))
        thinningModel.eval()

        sampleNo = 157 #initial design

        loadedInputForDisplacementModelImages_original = np.load("python/optimisation_funcs/model_confirugrations/u-bending/ModelPreparation/NN2_ManufacturingSurrogate/UBending_models_newgeo/InputTestOriginalAndNN_Feb23.npy") #for blank shape 
        loadedInputForDisplacementModelImages = loadedInputForDisplacementModelImages_original[sampleNo].copy()
        surrogateModelInput = torch.tensor(loadedInputForDisplacementModelImages).float().to(device).unsqueeze(0)
        gridOfOnes = torch.ones_like(surrogateModelInput.squeeze()[2])

        def get_max_thinning (BHF, friction, clearance, thickness):
            BHF = torch.tensor(surrogateModelInput.squeeze()[2].mean(), requires_grad=True)
            friction = torch.tensor(surrogateModelInput.squeeze()[3].mean(), requires_grad=True)
            clearance = torch.tensor(surrogateModelInput.squeeze()[4].mean(), requires_grad=True)
            thickness = torch.tensor(surrogateModelInput.squeeze()[5].mean(), requires_grad=True)

            surrogateModelInput[:, 2, :, :] = BHF * gridOfOnes #BHF
            surrogateModelInput[:, 3, :, :] = friction * gridOfOnes #friction
            surrogateModelInput[:, 4, :, :] = clearance * gridOfOnes #clearance
            surrogateModelInput[:, 5, :, :] = thickness * gridOfOnes #thickness

            thinningField = thinningModel(surrogateModelInput)[..., :-10].detach().numpy()

            return thinningField.max()
        
        if var2 == "max thinning":
            num_steps = 20
            # Set variable boundaries
            maxBHF = 59
            maxFriction = 0.199
            maxClearance = 1.49
            maxThickness = 2.99

            minBHF = 5.2
            minFriction = 0.1
            minClearance = 1.1
            minThickness = 0.51

            midBHF = minBHF + (maxBHF-minBHF)/2
            midFriction = minFriction + (maxFriction-minFriction)/2
            midClearance = minClearance + (maxClearance-minClearance)/2
            midThickness = minThickness + (maxThickness-minThickness)/2

            if var1 == "bhf":
                var_linspace = np.linspace(minBHF, maxBHF, num_steps)
                max_thinnings = []
                for v in var_linspace:
                    max_thinnings.append(get_max_thinning(v, midFriction, midClearance, midThickness))
                    print(v, "bhf thinning added")
            if var1 == "friction":
                var_linspace = np.linspace(minFriction, maxFriction, num_steps)
                max_thinnings = []
                for v in var_linspace:
                    max_thinnings.append(get_max_thinning(midBHF, v, midClearance, midThickness))
            if var1 == "clearance":
                var_linspace = np.linspace(minClearance, maxClearance, num_steps)
                max_thinnings = []
                for v in var_linspace:
                    max_thinnings.append(get_max_thinning(midBHF, midFriction, v, midThickness))
            if var1 == "thickness":
                var_linspace = np.linspace(minThickness, maxThickness, num_steps)
                max_thinnings = []
                for v in var_linspace:
                    max_thinnings.append(get_max_thinning(midBHF, midFriction, midClearance, v))
                    
            
            window.canvas.axes.clear()
            window.canvas.axes.plot(max_thinnings, var_linspace)
            window.canvas.axes.set_xlabel(var2)
            window.canvas.axes.set_ylabel(var1)
            # window.canvas.axes.legend(["Chamfer Loss", "Height Loss","Manufacturing Constraint Loss", "Similarity Loss"])
            window.canvas.axes.set_title(f"Sensitivity of {var1} on {var2}")
            window.canvas.draw()
            print("graph plotted")