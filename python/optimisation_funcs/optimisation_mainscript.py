import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
import pytorch3d
from skimage import measure
import imageio

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
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from torchvision import models

import trimesh

from PyQt6.QtCore import QObject, pyqtSignal

from python.optimisation_funcs import autodecoder
from python.optimisation_funcs.implicitShapeNetworkModels import *
from python.optimisation_funcs.plotting import *
from python.optimisation_funcs import manufacturingSurrogateModels_bulkhead
from python.optimisation_funcs import manufacturingSurrogateModels_ubending
from python.optimisation_funcs import U_ResSEUNet_512_B4_2000_LRFix00004_E6B6D6_Th


class worker(QObject):
    def __init__(self, num_iterations, file, component, window):
        super().__init__()
    # def __init__ (self, num_iterations=100, *args, **kwargs):
        self.num_iterations = num_iterations
        self.file = file
        self.component = component.lower()
        self.window = window
        self.cancelled = False
        self.window.cancel.connect(self.stop)

    finished = pyqtSignal(str)
    progress = pyqtSignal(int)

    def stop (self):
        print("stop requested")
        self.cancelled = True

    def run (self):

        if self.component == "bulkhead":

            #hyperparameters optimisation
            random_seed = 37
            num_iterations = self.num_iterations
            refLengthDies = 600 #mm

            chamfer_lambda = 1e2 #lambda0
            latentSimilarityLambda = 1e1 #lambda1
            manufacturingConstraint_lambda = 5e1 #lambda2 5e1
            latent_Lambda = 1e-2 #lambda3 #1e-4
            height_Lambda = 1e1 # lambda4
            normalsAlpha = 1 #beta

            learningRate = 1e-5 #<------------------- KEY POINT: The LR must be low enough such that gradient updates are not too large, which for the optimisation platform, may result in fast divergance since unrealistic geometries might be generated where NN3 is unable to predict on
            LrReductionInterval = 500

            allLatentVectorsPath = os.path.join("temp", "best_latent_vector", "Shape_1.pkl")
            testingSetLatentVectorsPath = os.path.join("temp", "best_latent_vector", "Shape_SHAPENAME.pkl")

            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(device)

            #load decoder
            latentVectorLength = 256
            hiddenLayerSizes = 512

            networkSettings = {
                "dims" : [hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes],
                "skip_in" : [4],
                "geometric_init" : True,
                "radius_init" : 1,
                "beta" : 100,
            }
            decoder = ImplicitNet(z_dim = latentVectorLength,
                                dims = networkSettings["dims"],
                                skip_in = networkSettings["skip_in"],
                                geometric_init = networkSettings["geometric_init"],
                                radius_init = networkSettings["radius_init"],
                                beta = networkSettings["beta"]).to(device)

            # if component.lower() == "bulkhead":
            #     decoder.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/bulkhead/NN1_ImplicitRepresentationDies_FinalTrained_NN1.pkl", map_location=device))
            # if component.lower() == "u-bending":
            #     decoder.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/NN1_FinalTrained.pkl", map_location=device))



            decoder.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/bulkhead/NN1_ImplicitRepresentationDies_FinalTrained_NN1.pkl", map_location=device))
            #load trained NN3 manufacturing constraints surrogate model
            #load trained model
            #hyperparameters surrogate model
            print("bulkhead thinning model loaded")
            batch_size = 20
            num_channel = (np.array([16,32,64,128,256,512])).astype(np.int64)
            thinningModel = manufacturingSurrogateModels_bulkhead.ResUNet(num_channel,batch_size)
            thinningModel = thinningModel.to(device)
            thinningModel.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/bulkhead/NN2_ManufacturingSurrogateModel_BestModel.pkl", map_location=device))
            
            thinningModel.eval()

            #------------------------------------------------------------------------------------------------------------------------------
            #Rasteriser 1: Rasteriser settings for top view orthographic projection - used to generate images for loss during optimisation
            #------------------------------------------------------------------------------------------------------------------------------
            CAMERA_DISTANCE = 1
            AZIMUTH = 0
            ELEVATION = 0
            IMAGE_SIZE = 256

            R, T = look_at_view_transform(dist = CAMERA_DISTANCE, elev = ELEVATION, azim = AZIMUTH) 
            cameras = FoVOrthographicCameras(max_y = 1, min_y = 0, max_x = 0, min_x = -1, device=device, R=R, T=T) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu

            # Here we set the output image to be of size 256 x 256
            raster_settings = RasterizationSettings(image_size = IMAGE_SIZE, blur_radius = 1e-16, faces_per_pixel = 1) #set a small blur radius to avoid -1 regions in the image (not captured by camera rays). See PyTorch3D documentation

            # Initialize rasterizer by using a MeshRasterizer class and store depth map
            rasterizer_forImageLoss = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

            #------------------------------------------------------------------------------------------------------------------------------
            #Rasteriser 2: Renderer settings, including rasteriser, for storing .gif of optimised mesh
            #------------------------------------------------------------------------------------------------------------------------------
            CAMERA_DISTANCE = 3
            AZIMUTH = 0
            ELEVATION = -60
            IMAGE_SIZE = 512

            rendererDevice = device

            R, T = look_at_view_transform(dist = CAMERA_DISTANCE, elev = ELEVATION, azim = AZIMUTH) 
            cameras = FoVOrthographicCameras(max_y = 0.4, min_y = -1.2, max_x = 0.7, min_x = -0.7, device=rendererDevice, R=R, T=T) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu

            # Here we set the output image to be of size 256 x 256
            raster_settings = RasterizationSettings(image_size = IMAGE_SIZE, blur_radius = 1e-16, faces_per_pixel = 1) #set a small blur radius to avoid -1 regions in the image (not captured by camera rays). See PyTorch3D documentation

            # Initialize rasterizer by using a MeshRasterizer class and store depth map
            rasterizer_forGIF = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

            #place point light infront of object.
            lights = PointLights(device=rendererDevice, location=[[2, -1, 5]])
            # lights = PointLights(device=rendererDevice, location=[[4, -1, 5]])

            #creae shader.The textured phong shader will interpolate the texture uv coordinates 
            #for each vertex, sample from a texture image and apply the Phong lighting model
            shader = SoftPhongShader(device=rendererDevice, cameras=cameras, lights=lights)

            materials = Materials(device=rendererDevice, shininess=5.0)

            # Create a phong renderer by composing a rasterizer and a shader. 
            renderer = MeshRenderer(rasterizer=rasterizer_forGIF, shader=shader)

            # Mesh colour
            rgb = [192/255, 192/255, 192/255]

            def process_image(multiChannelImage):
                multiChannelImage = multiChannelImage.squeeze()
                image_out_export = multiChannelImage[..., :3].detach().cpu().numpy().transpose((0, 1, 2)) # [image_size, image_size, RGB]
                alpha_out_export = multiChannelImage[..., 3].detach().cpu().numpy()
                image_out_export = np.concatenate( (image_out_export, alpha_out_export[:,:,np.newaxis]), -1 )

                I = image_out_export

                mn = I.min()
                mx = I.max()

                mx -= mn

                I = ((I - mn)/mx) * 255
                return I.astype(np.uint8)

            #------------------------------------------------------------------------
            #MAIN OPTIMISATION CODE
            #------------------------------------------------------------------------
            random.seed(random_seed)
            torch.random.manual_seed(random_seed)
            np.random.seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False  
            torch.backends.cudnn.enabled   = False

            runningChamferLoss = []
            runningHeightLoss = []
            runningBackwardLoss = []
            runningTotalBackwardLoss = []
            runningSimilarityMSELoss = []
            runningLatentLoss = []
            runningManufacturingConstraintLoss = []
            runningTotalLoss = []
            runningIterations = []
            runningMaxThinning = []

            imagesForGIF = []
            thinningGIFNames = []
            thinningObjectiveGIFNames = []

            latentVectorsForPlotting = []
            thinningFieldsForPlotting = []

            plotGIFhere = False

            backwardLossPreviousStep = torch.tensor(1000)

            ######################################################################################
            #PREPARATION BEFORE OPTIMISATION
            ######################################################################################

            #==================== TARGET VOLUME (NO RADII - IDEAL SHAPE) ==================== 
            #load STL mesh (CAD)
            # targetVolumePath = os.path.join("temp", "Optimisation", "OptimisationInputs", "IdealGeometriesSTLInputs", "Bulkhead3_Ideal2.STL")
            targetVolumePath = self.file
            targetVolume = trimesh.load_mesh(targetVolumePath, force = "mesh")

            #translate and scale the target mesh (unit side length)
            meshNodes = np.array(targetVolume.vertices)

            targetHeight = meshNodes[:,2].max() - meshNodes[:,2].min()

            meshNodes = rotateAboutZAxis(90, meshNodes)

            #translate
            meshNodes[:,0] = meshNodes[:,0] - meshNodes[:,0].min()
            meshNodes[:,1] = meshNodes[:,1] - meshNodes[:,1].min()
            meshNodes[:,2] = meshNodes[:,2] - meshNodes[:,2].max()
            meshNodes[:,2] = meshNodes[:,2] + (meshNodes[:,2].max() - meshNodes[:,2].min()) / 2 #offset Z dimention to be exactly in the centre of a unit cube
            #scale
            scalingFactor = meshNodes[:,0].max() - meshNodes[:,0].min() #X side length
            meshNodes = meshNodes/scalingFactor
            targetVolume.vertices = meshNodes

            targetVolumeVerts, targetVolumeFaces = torch.from_numpy(targetVolume.vertices).float(), torch.from_numpy(targetVolume.faces) #convert to pytorch tensors
            targetVolumeMeshPytorch3d = Meshes(verts=[targetVolumeVerts], faces=[targetVolumeFaces]) #create a pytorch3d Meshes object
            sampledTargetVolumePoints = ops.sample_points_from_meshes(meshes=targetVolumeMeshPytorch3d, num_samples=30000, return_normals=False).to(device) #randomly sample mesh surface with normals
            #==================== TARGET VOLUME (NO RADII - IDEAL SHAPE) ==================== 


            #==================== LOAD INITIAL GEOMETRY AND SET MANUFACTURING CONSTRAINT ==================== 
            #set constraint
            maxAllowableThinning = 0.15

            #load die data (latent vector)
            sampleNo = 0 #initial design
            latentPath = testingSetLatentVectorsPath.replace("SHAPENAME", str(sampleNo+1))
            latentForOptimization = torch.load(latentPath, map_location=device) #this will be updated during optimisation
            latentOfTargetDesign = torch.load(latentPath, map_location=device) # changed target design = starting geometry

            #load target die latent vector
            # sampleNo = 68 #target design
            # latentPath = trainingSetLatentVectorsPath.replace("SHAPENAME", str(sampleNo+1))
            # latentOfTargetDesign = torch.load(latentPath, map_location=device).detach() #latent vector of initial design which will not be updated

            #assemble uniform grid
            marchingCubesResolution = 90
            X, Y, Z = np.mgrid[0:1:complex(marchingCubesResolution), 0:1:complex(marchingCubesResolution), -0.5:0.5:complex(marchingCubesResolution)]
            inputPoints = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            inputPoints = torch.tensor(inputPoints).float().to(device) 
            numGridPoints = inputPoints.shape[0]

            optimizer = torch.optim.Adam([latentForOptimization], lr = learningRate) 

            #==================== LOAD INITIAL GEOMETRY AND SET MANUFACTURING CONSTRAINT ==================== 

            # trainingSetLatentVectors = torch.vstack(torch.load(allLatentVectorsPath)[0]).detach().to(device)
            trainingSetLatentVectors = torch.load(allLatentVectorsPath)[0].detach().to(device) # changed
            latentVectorFromPreviousIteration = torch.clone(latentForOptimization.detach())

            ########################################################################################
            # ----- Optimisation Loop ------
            ########################################################################################
            for e in range(num_iterations):

                ######################################################################################
                # UPDATE PROGRESS IN QT
                ######################################################################################

                self.progress.emit(e + 1)

                if self.cancelled:
                    self.finished.emit("Stopped early!")
                    break


                optimizer.zero_grad() #set all gradients to 0 so they do not accumulate

                ######################################################################################
                #Part 1: FORWARD PASS TO OBTAIN RECONSTRUCTED DIE GEOMETRY
                ######################################################################################

                #-------------------------------------------------------------------------------------
                #1.1: Predict SDF on uniform grid vertices using the trained network
                #-------------------------------------------------------------------------------------

                z = []
                for i,pnts in enumerate(torch.split(inputPoints,100000,dim=0)):

                    latentInputs = latentForOptimization.expand(pnts.shape[0], -1)
                    predictedSDF = decoder(latentInputs, pnts)
                    predictedSDF = predictedSDF.detach().cpu().numpy().squeeze()
                    z.append(predictedSDF)

                z = np.concatenate(z,axis=0)
                z = z.astype(np.float64)
                z = z.reshape(marchingCubesResolution, marchingCubesResolution, marchingCubesResolution)

                #-------------------------------------------------------------------------------------
                #1.2: Run marching cubes to extract the mesh
                #-------------------------------------------------------------------------------------
                verts, faces, _, _ = measure.marching_cubes(volume=z,level=0)
                # verts, faces, _, _ = measure.marching_cubes(volume=z,level=0.1) # changed

                #-------------------------------------------------------------------------------------
                #1.3: Transform the output of MC to match the training data
                #     I.e., the configuration that the network was trained on
                #-------------------------------------------------------------------------------------
                #rotate
                # verts = rotateAboutYAxis(90 + 180, verts)
                verts = rotateAboutZAxis(90, verts)

                #translate
                verts[:,0] = verts[:,0] - verts[:,0].min()
                verts[:,1] = verts[:,1] - verts[:,1].min()
                verts[:,2] = verts[:,2] - verts[:,2].max()
                verts[:,2] = verts[:,2] + (verts[:,2].max() - verts[:,2].min()) / 2 #offset Z dimention to be exactly in the centre of a unit cube

                #scale
                verts = verts/(verts[:,0].max() - verts[:,0].min()) #ensure side length of 1. Can replace this denominator with some pre-defined reference length
                normalisedHeight = verts[:,2].max() - verts[:,2].min()

                ######################################################################################
                #Part 2: COMPUTE OBJECTIVE FUNCTION (minimise chamfer distance)
                ######################################################################################
                xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype = torch.float32, device = device)  #replace the above with this
                faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device = device) 
                mesh = Meshes(verts=[xyz_upstream], faces=[faces_upstream]) #pytorch3d "Meshes" object

                #chamfer loss
                sampledReconstructedGeometryPoints = ops.sample_points_from_meshes(meshes=mesh, num_samples=30000, return_normals=False).to(device) #randomly sample mesh surface with normals
                loss_chamfer, _ =  chamfer_distance(sampledTargetVolumePoints, sampledReconstructedGeometryPoints)
                loss_chamfer = chamfer_lambda * loss_chamfer
                runningChamferLoss.append(loss_chamfer.detach().cpu().numpy())

                #height loss
                currentHeight = torch.max(xyz_upstream[:,2]) - torch.min(xyz_upstream[:,2])
                loss_height = torch.nn.functional.relu(currentHeight - targetHeight/refLengthDies) #targetHeight is height in mm of the target (ideal) geometry, which has no radii
                loss_height = height_Lambda * loss_height
                runningHeightLoss.append(loss_height.detach().cpu().numpy())

                ######################################################################################
                #Part 3: COMPUTE MANUFACTURING PERFORMANCE (maximum thinning)
                ######################################################################################

                #-------------------------------------------------------------------------------------
                #3.1: Project to image
                #-------------------------------------------------------------------------------------
                fragments = rasterizer_forImageLoss(mesh)
                depthMap = fragments.zbuf.squeeze() #no need for a renderer just use the rasterizer to obtain the depth map
                depthMap = depthMap.to(device)

                #correct the scaling due to camera view
                depthMap = depthMap - depthMap.min() #ensure min value of 0
                depthMap = depthMap / depthMap.max() #scale between 0 and 1
                depthMap = depthMap * normalisedHeight #correct the hight
                depthMap = depthMap - normalisedHeight/2 #offset Z direction so that Z=0 is exactly at the mid height - matches training SDF data
                depthMap = torch.rot90(depthMap, 3)

                #correct the Z dimention; min value of 0 and mm scale proved better for manufacturing surrogate model performance.
                #but a normalised -0.5 to 0.5 scale was better for NN1 performance, the implicit shape representation network
                depthMap = (depthMap - depthMap.min())*refLengthDies 

                #store first depth map for calculating image similarity to it
                if e == 0:
                    firstDepthMap = depthMap.detach()

                #-------------------------------------------------------------------------------------
                #3.2: Using NN3, calculate manufacturing performance loss
                #-------------------------------------------------------------------------------------
                thinningField = thinningModel(depthMap.unsqueeze(0).unsqueeze(0))
                maxThinning = thinningField.max()
                runningMaxThinning.append(maxThinning.detach().cpu().numpy())

                # if e == 600:
                #   manufacturingConstraint_lambda = 100 * manufacturingConstraint_lambda

                manufacturingConstraintLoss = manufacturingConstraint_lambda * F.relu(maxThinning - maxAllowableThinning)

                ######################################################################################
                #Part 3: ASSEMBLE LOSS FUNCTION AND BACKWARD PASS
                ######################################################################################

                #----------
                #Total loss
                #----------
                Loss = loss_chamfer + loss_height + manufacturingConstraintLoss

                #-------------------------------------------------------------------------------------
                #2.1: Store upstream gradients
                #-------------------------------------------------------------------------------------
                Loss.backward()
                dL_dv_i = xyz_upstream.grad #note: gradients are only calculated with respect to leaf nodes in the computational graph
                
                #-------------------------------------------------------------------------------------
                #2.2: Take care of weird stuff possibly happening
                #-------------------------------------------------------------------------------------
                dL_dv_i[torch.isnan(dL_dv_i)] = 0 
                # percentiles = torch.tensor(np.percentile(dL_dv_i.cpu().numpy(), 99.9, axis=0)).to(device)
                # dL_dv_i = torch.clamp(dL_dv_i, min = -percentiles, max = percentiles) #clip gradient components at percentiles 

                #-------------------------------------------------------------------------------------
                #2.3: Use vertices to compute full backward pass
                #-------------------------------------------------------------------------------------
                optimizer.zero_grad()
                xyz = torch.tensor(verts.astype(float), requires_grad = True, dtype = torch.float32, device = device) #predicted surface points
                latentInputs = latentForOptimization.expand(xyz.shape[0], -1)

                #-------------------------------------------------------------------------------------
                #2.4: First compute surface normals
                #-------------------------------------------------------------------------------------
                predictedSDF = decoder(latentInputs, xyz) #<==================== IMPORTANT: ensure scale of xyz is in line with the training data for the network
                loss_normals = torch.sum(predictedSDF) #some gradient invarient function, e.g., sum or mean, since autograd requires single scalar to calculate gradients of
                loss_normals.backward(retain_graph = True)
                normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1) #xyz.grad[i] is the partial derivatives of the SDF with respect to each x y and z coordinate of point i. Its normalised to get the unit normal vector, since SDF prediction is not perfect.

                #weighting normals (give more priority to non-flat areas such as radii)
                nonVerticalNormalsIdx = ((normals[:,2] <= 0.99) + 0) #give a large weight to normals on the radius and on the sidewall
                verticalNormalsIdx = ((normals[:,2] > 0.99) + 0) #give a low weight to normals on the flat surfaces
                normalsWeights = verticalNormalsIdx + normalsAlpha*nonVerticalNormalsIdx
                normals = normals * normalsWeights.unsqueeze(-1)

                #-------------------------------------------------------------------------------------
                #2.5: Now assemble inflow derivative
                #-------------------------------------------------------------------------------------
                optimizer.zero_grad()
                dL_ds_i = -torch.matmul(dL_dv_i.to(device).unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1) #dot product, normals are 100% correct because we visualised them

                #-------------------------------------------------------------------------------------
                #2.6: Finally assemble full backward pass
                #-------------------------------------------------------------------------------------
                similarityLoss = latentSimilarityLambda * torch.mean((latentOfTargetDesign - latentForOptimization).pow(2))
                #   similarityLoss = 0 # changed

                # loss_latent = latent_Lambda * torch.norm(latentForOptimization) #ensures shapes are not too far away from ones learnt during training. 

                #Here computing addional latent regulariation term. This ensures shapes are not too far away from ones learnt during training.
                #Further info section 6.8.4 from DeepMesh paper. 
                dist = torch.norm(trainingSetLatentVectors - latentVectorFromPreviousIteration, dim=1, p=None) #find distance from current latent vec to all vecs in training set
                #   knn = dist.topk(10, largest=False) #pick the top 10 highest
                knn = dist.topk(1, largest=False) #pick the top 10 highest changed
                nearestTrainingSetLatentVectors = trainingSetLatentVectors[knn.indices]
                additionalLatentReg = torch.mean(torch.norm((nearestTrainingSetLatentVectors - latentForOptimization).pow(2), dim=1))

                latentVectorFromPreviousIteration = torch.clone(latentForOptimization.detach()) #store for next iteration

                # if maxThinning.detach() < maxAllowableThinning:
                #   loss_latent = 0
                # else:
                #   loss_latent = latent_Lambda * additionalLatentReg

                loss_latent = latent_Lambda * additionalLatentReg

                loss_customBackwardPass = torch.mean(dL_ds_i * predictedSDF) #dot product
                loss_backward = loss_customBackwardPass + similarityLoss + loss_latent #<====================================================== SIMILARITY LOSS ADDED HERE #############################################
                loss_backward.backward() #calculates ds_dz (using predictedSDF) and finally dL_dz by the above chain rule

                #----------
                #Log
                #----------
                runningSimilarityMSELoss.append(similarityLoss.detach().cpu().numpy())
                runningManufacturingConstraintLoss.append(manufacturingConstraintLoss.detach().cpu().numpy())
                runningIterations.append(e)
                runningTotalBackwardLoss.append(loss_backward.detach().cpu().numpy())
                runningBackwardLoss.append(loss_customBackwardPass.detach().cpu().numpy())
                runningLatentLoss.append(loss_latent.detach().cpu().numpy())

                #-------------------------------------------------------------------------------------
                #2.7: Stopping criteria
                #------------------------------------------------------------------------------------- 

                # if (torch.abs(backwardLossPreviousStep - Loss.detach().cpu()) <= 0.01*(1+torch.abs(backwardLossPreviousStep))) and manufacturingConstraintLoss == 0 :
                #   break

                # backwardLossPreviousStep = Loss.detach().cpu()
                
                #-------------------------------------------------------------------------------------
                #2.8: Update parameters using the above custom back propagation and update LR if desired
                #-------------------------------------------------------------------------------------
                optimizer.step()

                # adjust_learning_rate(learningRate, optimizer, e)
                # print("Learning rate:", optimizer.param_groups[0]["lr"])

                ######################################################################################
                #Part 4: VISULISE 
                ######################################################################################

                if (e == 0) or plotGIFhere or (int(e+1) % 10 == 0):

                    trimeshMesh = trimesh.Trimesh(vertices=verts, faces=faces)
                    trimesh.repair.fix_inversion(trimeshMesh) #need this to flip the normals to get a good rendered image

                    verts = trimeshMesh.vertices
                    faces = trimeshMesh.faces

                    #-------------------------------------------------------------------------------------
                    #4.1: Rotate to match camera view (trial and error) and define tensors for plotting
                    #-------------------------------------------------------------------------------------

                    verts_dr = rotateAboutZAxis(-90-45, verts)

                    verts_dr = torch.tensor(verts_dr[None, :, :].copy(), dtype=torch.float32, requires_grad = False).to(device)  # (num_vertices, XYZ) -> (batch_size=1, num_vertices, XYZ)
                    faces_dr = torch.tensor(faces[None, :, :].copy()).to(device)

                    #-------------------------------------------------------------------------------------
                    #4.2: Hand crafted color map (vertex textures)
                    #-------------------------------------------------------------------------------------
                    
                    textures_dr = torch.tensor(rgb).to(device)*torch.ones(verts_dr.shape[1], 3, dtype=torch.float32).unsqueeze(0).to(device)
                    meshTextures = TexturesVertex(verts_features=textures_dr).to(device)

                    #-------------------------------------------------------------------------------------
                    #4.3: Define mesh and render it to an image, then plot and save
                    #-------------------------------------------------------------------------------------

                    meshForRendering = Meshes(verts = [verts_dr.squeeze()], faces = [faces_dr.squeeze()], textures = meshTextures)
                    imageForGIF = renderer(meshForRendering, materials=materials)

                    imagesForGIF.append(process_image(imageForGIF)) #forgot to store the alpha value

                    # plt.figure(figsize=(10, 10))
                    # plt.imshow(imageForGIF[0, ..., :3].cpu().numpy())
                    # plt.grid("off")
                    # plt.axis("off")
                    # plt.show()
                
                #-------------------------------------------------------------------------------------
                #4.4: Plot optimisation progress
                #-------------------------------------------------------------------------------------
                # if int(e+1) % 10 == 0:

                #plot reconstucted image
                # depthMapForPlotting = depthMap.detach().cpu().numpy() 
                # plt.imshow(depthMapForPlotting, cmap='jet')
                # cbar = plt.colorbar(orientation="vertical")
                # cbar.ax.tick_params(labelsize=20)
                # plt.axis("off")
                # plt.title("Depth map of reconstructed shape")
                # plt.show()

                
                #for thinning field prediction
                # print("Predicted manufacturing performance")
                # thinningFieldForPlotting = thinningField.detach().cpu().numpy().squeeze()
                # plt.imshow(thinningFieldForPlotting, cmap='jet', vmax=0.15, vmin = -0.1)
                # # plt.colorbar(fraction=0.046, pad=0.04)
                # cbar = plt.colorbar(orientation="vertical")
                # cbar.ax.tick_params(labelsize=20)
                # plt.axis("off")
                # fileName = os.path.join(outputPath, "thinning_" + str(e) + ".png")
                # thinningGIFNames.append(fileName)
                # plt.savefig(fileName)
                # plt.show()


                #for dL_dv_i gradient distributions
                # plt.figure(figsize=(20,3))
                # plt.subplot(1,3,1)
                # plt.hist(dL_dv_i[:,0].detach().cpu(), bins=200)
                # plt.xlabel("X component of dL_dv_i")
                # plt.ylabel("Count")

                # plt.subplot(1,3,2)
                # plt.hist(dL_dv_i[:,1].detach().cpu(), bins=200)
                # plt.xlabel("Y component of dL_dv_i")
                # plt.ylabel("Count")

                # plt.subplot(1,3,3)
                # plt.hist(dL_dv_i[:,2].detach().cpu(), bins=200)
                # plt.xlabel("Z component of dL_dv_i")
                # plt.ylabel("Count")
                # plt.show()


                #plot losses
                # plt.plot(runningIterations, runningChamferLoss)
                # plt.plot(runningIterations, runningHeightLoss)
                # plt.plot(runningIterations, runningManufacturingConstraintLoss)
                # plt.plot(runningIterations, runningSimilarityMSELoss)
                # plt.xlabel("Iterations")
                # plt.ylabel("Losses")
                # plt.legend(["Chamfer Loss", "Height Loss","Manufacturing Constraint Loss", "Similarity Loss"])
                # plt.title("Performance history")
                # plt.show()
                self.window.canvas.axes.plot(runningIterations, runningChamferLoss)
                self.window.canvas.axes.plot(runningIterations, runningHeightLoss)
                self.window.canvas.axes.plot(runningIterations, runningManufacturingConstraintLoss)
                self.window.canvas.axes.plot(runningIterations, runningSimilarityMSELoss)
                self.window.canvas.axes.set_xlabel("Iterations")
                self.window.canvas.axes.set_ylabel("Losses")
                self.window.canvas.axes.legend(["Chamfer Loss", "Height Loss","Manufacturing Constraint Loss", "Similarity Loss"])
                self.window.canvas.axes.set_title("Performance history")
                self.window.canvas.draw()
                # self.window.canvas.axes.show()
                # self.window.canvas.axes.plot([0,1,2,3,4], [5,1,20,3,4])

                # plt.plot(runningIterations, runningHeightLoss)
                # plt.xlabel("Iterations")
                # plt.ylabel("Losses")
                # plt.title("Height loss")
                # plt.show()

                # plt.plot(runningIterations, runningMaxThinning)
                # temp = np.array([[0,maxAllowableThinning],[e,maxAllowableThinning]])
                # plt.plot(temp[:,0], temp[:,1], 'r')
                # plt.xlabel("Iterations")
                # plt.ylabel("Max Thinning")
                # plt.title("Manufacturing performance only history")
                # plt.show()

                # plt.plot(runningIterations, runningTotalBackwardLoss)
                # plt.plot(runningIterations, runningBackwardLoss)
                # plt.plot(runningIterations, runningSimilarityMSELoss)
                # plt.plot(runningIterations, runningLatentLoss)
                # plt.xlabel("Iterations")
                # plt.ylabel("Losses")
                # plt.legend(["Total Backward Loss", "Backward Gradients Loss", "Similarity Loss", "Latent Loss"])
                # plt.title("Gradient histories before d/dz taken")
                # plt.show()

                # #plot rendered image
                # plt.figure(figsize=(10, 10))
                # plt.imshow(imageForGIF[0, ..., :3].cpu().numpy())
                # plt.grid("off")
                # plt.axis("off")
                # plt.show()

                #-------------------------------------------------------------------------------------
                #4.5: Save other things at intervals for plotting
                #-------------------------------------------------------------------------------------
                latentVectorsForPlotting.append((e, torch.clone(latentForOptimization)))
                thinningFieldsForPlotting.append((e, thinningField.detach().cpu().numpy().squeeze()))

                # break

            ######################################################################################
            #Part 5: STORE OUTPUTS OF OPTIMISATION
            ######################################################################################

            #-------------------------------------------------------------------------------------
            #5.0: Store .gif
            #-------------------------------------------------------------------------------------
            if plotGIFhere:

                #thinning GIF
                with imageio.get_writer(thinningGIFNames, mode='I') as writer:
                    for filename in thinningGIFNames:
                        image = imageio.imread(filename)
                        writer.append_data(image)

                #remove files
                for filename in set(thinningGIFNames):
                    os.remove(filename)

            #save for plotting
            outputPath = os.path.join("temp", "Optimisation", "OptimisationOutputs", "bulkhead")
            if not os.path.exists(outputPath):
                os.makedirs(outputPath)

            latentVectorsForPlottingName = os.path.join(outputPath, "LatentVectorsForPlotting.pkl")
            thinningFieldsForPlottingName = os.path.join(outputPath, "ThinningFieldsForPlotting.pkl")

            torch.save(latentVectorsForPlotting, latentVectorsForPlottingName)
            torch.save(thinningFieldsForPlotting, thinningFieldsForPlottingName)

            ######################################################################################
            # Tell Qt it's done
            ######################################################################################
            self.finished.emit('bulkhead')

            print("Done.")

##########################################################################################################################################
# U-bending
##########################################################################################################################################

        if self.component == "u-bending":

            #hyperparameters optimisation
            random_seed = 37
            refLengthDies = 600 #mm
            # normalsAlpha = 50 #beta
            normalsAlpha = 1 #beta

            backwardsLambda = 1 #lambda_0
            latentSimilarityLambda = 0 #lambda_1 if added after backward pass is calculated
            manufacturingConstraint_lambda = 1e4 #1e2 #lambda_2
            latent_Lambda = 0 #lambda_3
            meanThinning_lambda = 0 #lambda_4
            springbackLoss_lambda = 1e3
            learningRate = 1e-5 #eta <------------------- KEY POINT: The LR must be low enough such that gradient updates are not too large, which for the optimisation platform, may result in fast divergance since unrealistic geometries might be generated where NN3 is unable to predict on
            LrReductionInterval = 500

            LR_BHF = 1e-1
            LR_friction = 1e-4
            LR_clearance = 1e-2
            LR_thickness = 1e-2

            #hyperparameters surrogate model
            batch_size = 4
            num_channel_thinning = (np.array([4,8,16,32,64,128,256,512])).astype(np.int64)
            num_channel_displacement = (np.array([4,8,16,32,64,128,256,512])).astype(np.int64)

            #load trained NN1 shape representation model
            hiddenLayerSizes = 128
            latentVectorLength = 64
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(device)

            #load trained model
            # loadedCheckpoint = torch.load(NN1Path, map_location = device)

            networkSettings = {
                "dims" : [hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes],
                "skip_in" : [4],
                "geometric_init" : True,
                "radius_init" : 1,
                "beta" : 100,
            }

            decoder = ImplicitNet(z_dim = latentVectorLength,
                                dims = networkSettings["dims"],
                                skip_in = networkSettings["skip_in"],
                                geometric_init = networkSettings["geometric_init"],
                                radius_init = networkSettings["radius_init"],
                                beta = networkSettings["beta"]).to(device)

            # decoder.load_state_dict(loadedCheckpoint['state_dict'])
            decoder.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/NN1_FinalTrained.pkl", map_location=device))

            decoder.eval()

            #load trained NN3 manufacturing constraints surrogate models
            #load trained model
            thinningModel = manufacturingSurrogateModels_ubending.ResUNet_Thinning(num_channel_thinning,batch_size)
            thinningModel = thinningModel.to(device)
            thinningModel.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/ResSEUNet_512_B4_2000_COS0.0_LRFix0.0002_E4B6D4_NewShape_08Feb23_best.pkl",map_location=device))
            thinningModel.eval()

            displacementModel = manufacturingSurrogateModels_ubending.ResUNet_totalDisplacement(num_channel_displacement,batch_size)
            displacementModel = displacementModel.to(device)
            displacementModel.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/ResSEUNet_512_B4_2000_COS0.0_VECD0.5_LRFix0.0001_E6B6D6_disp_MAE_27Feb23_best.pkl",map_location=device))
            displacementModel.eval()

            #------------------------------------------------------------------------------------------------------------------------------
            #Rasteriser 1: Rasteriser settings for top view orthographic projection 
            #------------------------------------------------------------------------------------------------------------------------------
            refLengthDies = 168 
            additionalDistance = 75

            CAMERA_DISTANCE_ZProjection = additionalDistance
            AZIMUTH = 0
            ELEVATION = 0
            IMAGE_SIZE = 256

            R, T = look_at_view_transform(dist = CAMERA_DISTANCE_ZProjection, elev = ELEVATION, azim = AZIMUTH) 
            # cameras = FoVOrthographicCameras(max_y = 0.47, min_y = 0, max_x = -0.17, min_x = -0.83, device=device, R=R, T=T) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu

            #how to calculate coordinates:
            #y direction is simple
            #x direction: the magnitude of (min_x - max_x should equal half the x side length, which is 168mm for the U-channel)
            cameras = FoVOrthographicCameras(max_y = 83, min_y = 0, max_x = -42, min_x = -126, device=device, R=R, T=T) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu


            # Here we set the output image to be of size 256 x 256
            raster_settings = RasterizationSettings(image_size = (IMAGE_SIZE, int(IMAGE_SIZE*2)), blur_radius = 1e-16, faces_per_pixel = 1) #set a small blur radius to avoid -1 regions in the image (not captured by camera rays). See PyTorch3D documentation
            # raster_settings = RasterizationSettings(image_size = (IMAGE_SIZE, IMAGE_SIZE), blur_radius = 1e-16, faces_per_pixel = 1) #set a small blur radius to avoid -1 regions in the image (not captured by camera rays). See PyTorch3D documentation

            # Initialize rasterizer by using a MeshRasterizer class and store depth map
            rasterizer_ZProjection = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

            #------------------------------------------------------------------------------------------------------------------------------
            #Rasteriser 2: Rasteriser settings for X axis orthographic projection 
            #------------------------------------------------------------------------------------------------------------------------------
            refLengthDies = 168
            additionalDistance = 10

            CAMERA_DISTANCE_XProjection = refLengthDies# + additionalDistance
            AZIMUTH = 90
            ELEVATION = 0
            IMAGE_H = 256
            IMAGE_W = 384

            R, T = look_at_view_transform(dist = CAMERA_DISTANCE_XProjection, elev = ELEVATION, azim = AZIMUTH) 
            # cameras = FoVOrthographicCameras(max_y = 0.47, min_y = 0, max_x = -0.17, min_x = -0.83, device=device, R=R, T=T) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu

            #how to calculate coordinates:
            #here, the min_x and max_x refer to the Z-axis in world space
            #()
            cameras = FoVOrthographicCameras(max_y = 83, min_y = 0, max_x = 22.2665, min_x = -22.2665, device=device, R=R, T=T) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu

            # Here we set the output image to be of size 256 x 256
            raster_settings = RasterizationSettings(image_size = (IMAGE_SIZE, IMAGE_W), blur_radius = 1e-16, faces_per_pixel = 1) #set a small blur radius to avoid -1 regions in the image (not captured by camera rays). See PyTorch3D documentation
            # raster_settings = RasterizationSettings(image_size = (IMAGE_SIZE, IMAGE_SIZE), blur_radius = 1e-16, faces_per_pixel = 1) #set a small blur radius to avoid -1 regions in the image (not captured by camera rays). See PyTorch3D documentation

            # Initialize rasterizer by using a MeshRasterizer class and store depth map
            rasterizer_XProjection = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

            rasterizers = [rasterizer_ZProjection, rasterizer_XProjection]

            #------------------------------------------------------------------------------------------------------------------------------
            #Rasteriser 3: Renderer settings, including rasteriser, for storing .gif of optimised mesh
            #------------------------------------------------------------------------------------------------------------------------------
            CAMERA_DISTANCE = 3
            AZIMUTH = 0
            ELEVATION = -60
            IMAGE_SIZE = 512

            rendererDevice = device

            R, T = look_at_view_transform(dist = CAMERA_DISTANCE, elev = ELEVATION, azim = AZIMUTH) 
            cameras = FoVOrthographicCameras(max_y = 0.4*refLengthDies, min_y = -1.2*refLengthDies, max_x = 0.7*refLengthDies, min_x = -0.7*refLengthDies, device=rendererDevice, R=R, T=T) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu

            # Here we set the output image to be of size 256 x 256
            raster_settings = RasterizationSettings(image_size = IMAGE_SIZE, blur_radius = 1e-16, faces_per_pixel = 1) #set a small blur radius to avoid -1 regions in the image (not captured by camera rays). See PyTorch3D documentation

            # Initialize rasterizer by using a MeshRasterizer class and store depth map
            rasterizer_forGIF = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

            #place point light infront of object.
            lights = PointLights(device=rendererDevice, location=[[2, -1, 5]])
            # lights = PointLights(device=rendererDevice, location=[[4, -1, 5]])

            #creae shader.The textured phong shader will interpolate the texture uv coordinates 
            #for each vertex, sample from a texture image and apply the Phong lighting model
            shader = SoftPhongShader(device=rendererDevice, cameras=cameras, lights=lights)

            materials = Materials(device=rendererDevice, shininess=5.0)

            # Create a phong renderer by composing a rasterizer and a shader. 
            renderer = MeshRenderer(rasterizer=rasterizer_forGIF, shader=shader)

            # Mesh colour
            rgb = [192/255, 192/255, 192/255]

            def process_image(multiChannelImage):
                multiChannelImage = multiChannelImage.squeeze()
                image_out_export = multiChannelImage[..., :3].detach().cpu().numpy().transpose((0, 1, 2)) # [image_size, image_size, RGB]
                alpha_out_export = multiChannelImage[..., 3].detach().cpu().numpy()
                image_out_export = np.concatenate( (image_out_export, alpha_out_export[:,:,np.newaxis]), -1 )

                I = image_out_export

                mn = I.min()
                mx = I.max()

                mx -= mn

                I = ((I - mn)/mx) * 255
                return I.astype(np.uint8)
            
            def adjust_learning_rate(initial_lr, optimizer, iter):
                lr = initial_lr * ((0.25) ** (iter // LrReductionInterval))
                lr = max(lr, 1e-5) #set a minimum LR
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            def displacementsToPositions(totalDisplacementField):
  
                # Store the output from the displacement model
                totalDisplacementField = totalDisplacementField.squeeze().clone()

                # Get the displacement vectors as images
                displacement_x = totalDisplacementField[0]
                displacement_y = totalDisplacementField[1]
                displacement_z = totalDisplacementField[2]

                # Create a meshgrid of x and y coordinates for each pixel in the images
                x_coords = torch.linspace(0, 160, displacement_x.shape[1])
                y_coords = torch.linspace(0, 80, displacement_x.shape[0])
                x_mesh, y_mesh = torch.meshgrid(x_coords, y_coords)

                # Calculate the 3D coordinates of each point
                x_coords_deformedComponent = x_mesh.to(device).T + displacement_x
                y_coords_deformedComponent = y_mesh.to(device).T + displacement_y
                z_coords_deformedComponent = displacement_z

                # Stack the three tensors along a new dimension. Shape: torch.Size([1, 3, 256, 512])
                deformed_component_positions = torch.stack((x_coords_deformedComponent, 
                                                            y_coords_deformedComponent, 
                                                            z_coords_deformedComponent), dim=0).unsqueeze(0)

                return deformed_component_positions
            
            loadedPostStampingDisplacements = torch.tensor(np.load("python/optimisation_funcs/model_confirugrations/u-bending/ModelPreparation/NN2_ManufacturingSurrogate/UBending_models_newgeo/Ubending_reference_blank_geo_before_springback.npy")) #for reference shape, taken after post stamping but before springback
            postStampingTargetDisplacement = loadedPostStampingDisplacements[8].to(device).to(torch.float32)
            postStampingGeometryPositions = displacementsToPositions(postStampingTargetDisplacement)
            
            # Load pre-trained VGG16 model
            vgg = models.vgg16(pretrained=True).features.to(device)

            # Use layers of VGG16 to extract features
            feature_layers = [3, 8, 15, 22] # convert strings to integers
            features = nn.Sequential(*list(vgg)[:max(feature_layers)+1]).eval()
            
            # Define perceptual loss function
            def perceptual_loss(image1, image2, imagesAreNorms = False):

                #image1 and 2 are of size torch.size([3, 256, 512])
                image1 = image1/120
                image2 = image2/120

                if imagesAreNorms:
                    # match the expected shape (batch_size, channels, height, width)
                    image1 = image1.repeat(3, 1, 1)  # Repeat along the channel dimension three times
                    image2 = image2.repeat(3, 1, 1)  # Repeat along the channel dimension three times

                # image1 = image1.unsqueeze(0)  # Add a batch dimension at the beginning
                # image2 = image2.unsqueeze(0)  # Add a batch dimension at the beginning

                loss = 0
                for layer in feature_layers:
                    x1 = features[:layer](image1)
                    x2 = features[:layer](image2)
                    loss += torch.mean((x1 - x2)**2)
                return loss

            # Set variable boundaries
            maxBHF = 59
            maxFriction = 0.199
            maxClearance = 1.49
            maxThickness = 2.99

            minBHF = 5.2
            minFriction = 0.1
            minClearance = 1.1
            minThickness = 0.51

            interval = 500
            lrs = []
            for e in range(2000):
                lr = learningRate * ((0.25) ** (e // interval))
                lr = max(lr, 1e-5) #set a minimum LR
                lrs.append(lr)

            #------------------------------------------------------------------------
            #MAIN OPTIMISATION CODE
            #------------------------------------------------------------------------
            random.seed(random_seed)
            torch.random.manual_seed(random_seed)
            np.random.seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

            # torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False  
            torch.backends.cudnn.enabled   = False

            runningBackwardLoss = []
            runningTotalBackwardLoss = []
            runningSimilarityMSELoss = []
            runningLatentLoss = []
            runningManufacturingConstraintLoss = []
            runningTotalLoss = []
            runningIterations = []
            runningMaxThinning = []
            runningMeanThinningFieldMasked = []
            runningSpringbackLoss = []
            runningMAELoss = []
            runningPerceptualLoss = []

            imagesForGIF = []
            thinningGIFNames = []
            thinningObjectiveGIFNames = []

            latentVectorsForPlotting = []
            thinningFieldsForPlotting = []

            runningBHF = []
            runningFriction = []
            runningClearance = []
            runningThickness = []

            runningConstraint1 = []
            runningConstraint2 = []
            runningConstraint3 = []
            runningConstraint4 = []
            runningConstraint5 = []
            runningConstraint6 = []
            runningConstraint7 = []
            runningConstraint8 = []

            constraint1_lambda = 1e1
            constraint2_lambda = 1e1
            constraint3_lambda = 1e1
            constraint4_lambda = 1e1
            constraint5_lambda = 1e1
            constraint6_lambda = 1e1
            constraint7_lambda = 1e1
            constraint8_lambda = 1e1

            # iterationsToPlot = [0, 174, 500, 800, num_iterations-1]

            plotGIFhere = False

            backwardLossPreviousStep = torch.tensor(1000)

            ######################################################################################
            #PREPARATION BEFORE OPTIMISATION
            ######################################################################################

            sampleNo = 157 #initial design

            loadedInputForDisplacementModelImages_original = np.load("python/optimisation_funcs/model_confirugrations/u-bending/ModelPreparation/NN2_ManufacturingSurrogate/UBending_models_newgeo/InputTestOriginalAndNN_Feb23.npy") #for blank shape 
            loadedInputForDisplacementModelImages = loadedInputForDisplacementModelImages_original[sampleNo].copy()

            surrogateModelInput = torch.tensor(loadedInputForDisplacementModelImages).float().to(device)

            gridOfOnes = torch.ones_like(surrogateModelInput[2])

            BHF = torch.tensor(surrogateModelInput[2].mean(), requires_grad=True)
            friction = torch.tensor(surrogateModelInput[3].mean(), requires_grad=True)
            clearance = torch.tensor(surrogateModelInput[4].mean(), requires_grad=True)
            thickness = torch.tensor(surrogateModelInput[5].mean(), requires_grad=True)

            surrogateModelInput = surrogateModelInput.unsqueeze(0)

            # BHF = 20
            # Friction = 0.15
            # Clearance = 1.1
            # BlankThickness = 1.5

            # surrogateModelInput[0][2] = BHF*surrogateModelInput[0][2]
            # surrogateModelInput[0][3] = Friction*surrogateModelInput[0][3]
            # surrogateModelInput[0][4] = Clearance*surrogateModelInput[0][4]
            # surrogateModelInput[0][5] = BlankThickness*surrogateModelInput[0][5]

            #target displacement magnitude field (later keep as XYZ and optimise to minimise the difference between each)
            # postStampingTargetDisplacement = loadedPostStampingDisplacements[8].to(device).to(torch.float32) #x y z displacements
            # postStampingGeometryPositions = displacementsToPositions(postStampingTargetDisplacement)[..., :-10] #take off last 10 pixels

            #set constraint
            maxAllowableThinning = 0.1
            springbackTolerance = 5 #mm

            #load die data (latent vector)
            # latentPath = testingSetLatentVectorsPath.replace("SHAPENAME", str(sampleNo+1))
            latentPath = "temp/best_latent_vector/Shape_1.pkl"
            # allLatentVectorsPath = os.path.join("temp", "best_latent_vector", "Shape_1.pkl")

            # trainingSetLatentVectors = torch.vstack(torch.load(allLatentVectorsPath)[0]).detach().to(device) # changed
            latentForOptimization = torch.load(latentPath, map_location=device) #this will be updated during optimisation
            latentOfInitialDesign = torch.clone(latentForOptimization.detach()) #latent vector of initial design which will not be updated

            ########################################################################################
            # ----- Uniform grid ------
            ########################################################################################
            # marchingCubesResolution = 90
            # X, Y, Z = np.mgrid[0:1:complex(marchingCubesResolution), 0:0.5:complex(marchingCubesResolution)/2, -0.5:0.5:complex(marchingCubesResolution)]
            # inputPoints = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            # inputPoints = torch.tensor(inputPoints).float().to(device)
            # numGridPoints = inputPoints.shape[0]

            # Define the ranges for x, y, and z
            min_x, max_x = 0, 1
            min_y, max_y = 0, 0.5
            min_z, max_z = -0.5, 0.5

            refinementFactor = 1 #remember during optimisation, this must match what was used to generate the NN2 training input images
            baseResolution = 90
            # Define the number of points in each dimension
            nx, ny, nz = int(baseResolution*(max_x-min_x))*refinementFactor, int(baseResolution*(max_y-min_y))*refinementFactor, int(baseResolution*(max_z-min_z))*refinementFactor

            # Generate a regularly spaced grid of data points
            x = np.linspace(min_x, max_x, nx)
            y = np.linspace(min_y, max_y, ny)
            z = np.linspace(min_z, max_z, nz)
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

            # Combine xx, yy, and zz into a single array
            inputPoints = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
            inputPoints = torch.tensor(inputPoints).float().to(device)
            numGridPoints = inputPoints.shape[0]

            ########################################################################################
            # ----- Uniform grid ------
            ########################################################################################

            #each parameter has its own LR since it was not normalised during training of the model
            optimizer = torch.optim.Adam([{"params": BHF, "lr": LR_BHF}, 
                                        {"params": friction, "lr": LR_friction},
                                        {"params": clearance, "lr": LR_clearance},
                                        {"params": thickness, "lr": LR_thickness},
                                        {"params": latentForOptimization, "lr": learningRate}])

            # optimizer = torch.optim.Adam([latentForOptimization], lr = learningRate) 

            latentVectorFromPreviousIteration = torch.clone(latentForOptimization.detach())

            ########################################################################################
            # ----- Optimisation Loop ------
            ########################################################################################
            for e in range(self.num_iterations):

                ######################################################################################
                # UPDATE PROGRESS IN QT
                ######################################################################################

                self.progress.emit(e + 1)

                if self.cancelled:
                    self.finished.emit("Stopped early!")
                    break

                optimizer.zero_grad() #set all gradients to 0 so they do not accumulate

                ######################################################################################
                #Part 1: FORWARD PASS TO OBTAIN IMAGE OF OF RECONSTRUCTED DIE GEOMETRY
                ######################################################################################

                #-------------------------------------------------------------------------------------
                #1.1: predict SDF on uniform grid vertices using the trained network
                #-------------------------------------------------------------------------------------
                z = []
                for _,pnts in enumerate(torch.split(inputPoints,100000,dim=0)):

                    latentInputs = latentForOptimization.expand(pnts.shape[0], -1)
                    predictedSDF = decoder(latentInputs, pnts)
                    predictedSDF = predictedSDF.detach().cpu().numpy().squeeze()
                    z.append(predictedSDF)

                z = np.concatenate(z,axis=0)
                z = z.astype(np.float64)
                z = z.reshape(nx, ny, nz)
                # z = z.reshape(marchingCubesResolution, int(marchingCubesResolution/2), marchingCubesResolution)

                #-------------------------------------------------------------------------------------
                #1.2: run marching cubes to extract the mesh
                #-------------------------------------------------------------------------------------
                verts, faces, _, _ = measure.marching_cubes(z, level=0) #the lewiner one is better
                verts_copy = np.copy(verts)

                #-------------------------------------------------------------------------------------
                #1.3: Transform the output of MC to match the training data. I.e., the configuration that the network was trained on
                #-------------------------------------------------------------------------------------

                #rotate
                # verts = rotateAboutZAxis(90, verts)

                #translate
                verts[:,0] = verts[:,0] - verts[:,0].min()
                verts[:,1] = verts[:,1] - verts[:,1].min()
                verts[:,2] = verts[:,2] - verts[:,2].max()
                verts[:,2] = verts[:,2] - verts[:,2].min()/2 #ensure mid height of 0. This is needed for the x-projections

                #undo scaling due to marching cubes
                # NOTE: its important to have the correct dimentions here, since this will ensure the image projections are also correct
                verts = verts/(verts[:,0].max() - verts[:,0].min()) #ensure side length of 1
                verts = verts * refLengthDies

                #-------------------------------------------------------------------------------------
                #1.4: Prepare for rasteriser
                #-------------------------------------------------------------------------------------
                xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype = torch.float32, device = device) 
                faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device =device) 
                mesh = Meshes(verts=[xyz_upstream], faces=[faces_upstream])

                #-------------------------------------------------------------------------------------
                #1.5: Project to image
                #-------------------------------------------------------------------------------------
                for count, rasterizer_projection in enumerate(rasterizers):
                    fragments = rasterizer_projection(mesh)
                    depthMap = fragments.zbuf.squeeze() #no need for a renderer just use the rasterizer to obtain the depth map
                    depthMap = depthMap.to(device)

                    if count == 1:
                        depthMap = -1 * (depthMap - CAMERA_DISTANCE_XProjection + 10) #added 10 here because this would match the X-projections to the ones given for the training data. In the X-projections generating script, the projection was done using an old set of codes which were overwriten
                        edge1 = torch.repeat_interleave(depthMap[:, -1].unsqueeze(1), repeats = 64, dim = 1)
                        edge2 = torch.repeat_interleave(depthMap[:, 0].unsqueeze(1), repeats = 64, dim = 1)
                        depthMap = torch.cat((edge2, depthMap, edge1), dim = 1)
                    else:
                        depthMap = (depthMap - CAMERA_DISTANCE_ZProjection)

                    # # for plotting only
                    # depthMapForPlotting = depthMap.detach().cpu().numpy() 
                    # # plt.figure(figsize=(100,50))
                    # plt.imshow(depthMapForPlotting, cmap='jet')
                    # plt.colorbar()
                    # plt.axis("off")
                    # plt.show()

                    surrogateModelInput[:, count, :, :] = depthMap

                surrogateModelInput[:, 2, :, :] = BHF * gridOfOnes #BHF
                surrogateModelInput[:, 3, :, :] = friction * gridOfOnes #friction
                surrogateModelInput[:, 4, :, :] = clearance * gridOfOnes #clearance
                surrogateModelInput[:, 5, :, :] = thickness * gridOfOnes #thickness

                ######################################################################################
                #Part 2: ASSEMBLE LOSS FUNCTION AND BACKWARD PASS
                ######################################################################################

                #-------------------------------------------------------------------------------------
                #2.1: Using NN3, calculate manufacturing performance
                #-------------------------------------------------------------------------------------
                thinningField = thinningModel(surrogateModelInput)[..., :-10]
                totalDisplacementField = displacementModel(surrogateModelInput).squeeze() #total displacement
                postStampingAndSpringbackGeometryPositions = displacementsToPositions(totalDisplacementField)[..., :-10] #deformed positions

                #-------------------------------------------------------------------------------------
                #2.2: Compute loss forward loss
                #-------------------------------------------------------------------------------------

                #-----------------------------
                #Manufacturing constraint loss
                #-----------------------------
                maxThinning = thinningField.max()
                runningMaxThinning.append(maxThinning.detach().cpu().numpy())
                manufacturingConstraintLoss = manufacturingConstraint_lambda * F.relu(maxThinning - maxAllowableThinning)

                # #-----------------------------
                # #Mean thinning loss
                # #-----------------------------
                # thinningFieldMask = torch.clone(thinningField).clamp(min=maxAllowableThinning) #generate differentiable mask 
                # meanThinningFieldMasked = meanThinning_lambda * torch.mean(thinningFieldMask)
                # runningMeanThinningFieldMasked.append(meanThinningFieldMasked.detach().cpu().numpy())

                #-----------------------------
                #Springback loss
                #-----------------------------
                # springbackLoss = torch.mean(torch.abs(totalDisplacementMagnitudeField - postStampingTargetDisplacementNorm)) 
                # print(postStampingGeometryPositions.shape)
                MAELoss = 5e2*torch.abs(postStampingAndSpringbackGeometryPositions - postStampingGeometryPositions[:,:,:,:502]).mean()
                PerceptualLoss = 5e2*perceptual_loss(postStampingAndSpringbackGeometryPositions, postStampingGeometryPositions[:,:,:,:502]) #<------------- modify here to take each 3 channel separetely, not norms
                springbackLoss = MAELoss + PerceptualLoss
                springbackLoss = springbackLoss_lambda * springbackLoss

                # MAELoss = PerceptualLoss = springbackLoss = 0

                runningMAELoss.append(MAELoss.item())
                runningPerceptualLoss.append(PerceptualLoss.item())
                runningSpringbackLoss.append(springbackLoss.item())

                # runningMAELoss.append(MAELoss)
                # runningPerceptualLoss.append(PerceptualLoss)
                # runningSpringbackLoss.append(springbackLoss)

                #-----------------------------
                #Process param bounds constraints
                #-----------------------------
                #penalty on the lagrangian if variable value falls outside of bounds
                constraint1 = constraint1_lambda * F.relu(BHF/maxBHF - 1) 
                constraint2 = constraint2_lambda * F.relu(1 - BHF/minBHF)

                constraint3 = constraint3_lambda * F.relu(friction/maxFriction - 1) 
                constraint4 = constraint4_lambda * F.relu(1 - friction/minFriction)

                constraint5 = constraint5_lambda * F.relu(clearance/maxClearance - 1) 
                constraint6 = constraint6_lambda * F.relu(1 - clearance/minClearance)

                constraint7 = constraint7_lambda * F.relu(thickness/maxThickness - 1) 
                constraint8 = constraint8_lambda * F.relu(1 - thickness/minThickness)

                runningConstraint1.append(constraint1.detach().cpu().numpy())
                runningConstraint2.append(constraint2.detach().cpu().numpy())
                runningConstraint3.append(constraint3.detach().cpu().numpy())
                runningConstraint4.append(constraint4.detach().cpu().numpy())
                runningConstraint5.append(constraint5.detach().cpu().numpy())
                runningConstraint6.append(constraint6.detach().cpu().numpy())
                runningConstraint7.append(constraint7.detach().cpu().numpy())
                runningConstraint8.append(constraint8.detach().cpu().numpy())

                #----------
                #Total loss
                #----------
                Loss = manufacturingConstraintLoss + springbackLoss + constraint1 + constraint2 + constraint3 + constraint4 + constraint5 + constraint6 + constraint7 + constraint8 # + meanThinningFieldMasked 

                #-------------------------------------------------------------------------------------
                #2.3: Store upstream gradients
                #-------------------------------------------------------------------------------------
                Loss.backward(retain_graph=True)
                dL_dv_i = xyz_upstream.grad #note: gradients are only calculated with respect to leaf nodes in the computational graph
                
                #-------------------------------------------------------------------------------------
                #2.4: Take care of weird stuff possibly happening
                #-------------------------------------------------------------------------------------
                dL_dv_i[torch.isnan(dL_dv_i)] = 0 
                # percentiles = torch.tensor(np.percentile(dL_dv_i.cpu().numpy(), 99.9, axis=0)).to(device)
                # dL_dv_i = torch.clamp(dL_dv_i, min = -percentiles, max = percentiles) #clip gradient components at percentiles 

                #-------------------------------------------------------------------------------------
                #2.5: Use vertices to compute full backward pass
                #-------------------------------------------------------------------------------------
                optimizer.zero_grad()
                xyz = torch.tensor(verts.astype(float), requires_grad = True, dtype = torch.float32, device = device) #predicted surface points
                latentInputs = latentForOptimization.expand(xyz.shape[0], -1)

                #-------------------------------------------------------------------------------------
                #2.6: First compute surface normals
                #-------------------------------------------------------------------------------------
                predictedSDF = decoder(latentInputs, xyz) #<==================== IMPORTANT: ensure scale of xyz is in line with the training data for the network
                loss_normals = torch.sum(predictedSDF) #some gradient invarient function, e.g., sum or mean, since autograd requires single scalar to calculate gradients of
                loss_normals.backward(retain_graph = True)
                normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1) #xyz.grad[i] is the partial derivatives of the SDF with respect to each x y and z coordinate of point i. Its normalised to get the unit normal vector, since SDF prediction is not perfect.

                # #weighting normals (give more priority to non-flat areas such as radii)
                # nonVerticalNormalsIdx = ((normals[:,2] <= 0.99) + 0) #give a large weight to normals on the radius and on the sidewall
                # verticalNormalsIdx = ((normals[:,2] > 0.99) + 0) #give a low weight to normals on the flat surfaces
                # normalsWeights = verticalNormalsIdx + normalsAlpha*nonVerticalNormalsIdx
                # normals = normals * normalsWeights.unsqueeze(-1)

                #-------------------------------------------------------------------------------------
                #2.7: Now assemble inflow derivative
                #-------------------------------------------------------------------------------------
                optimizer.zero_grad()
                dL_ds_i = -torch.matmul(dL_dv_i.to(device).unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1) #dot product, normals are 100% correct because we visualised them

                #-------------------------------------------------------------------------------------
                #2.8: Finally assemble full backward pass
                #-------------------------------------------------------------------------------------
                similarityLoss = latentSimilarityLambda * torch.mean((latentOfInitialDesign - latentForOptimization).pow(2))

                # loss_latent = latent_Lambda * torch.norm(latentForOptimization)  #old

                # #Here computing addional latent regulariation term. This ensures shapes are not too far away from ones learnt during training.
                # #Further info section 6.8.4 from DeepMesh paper. 
                # dist = torch.norm(trainingSetLatentVectors - latentVectorFromPreviousIteration, dim=1, p=None) #find distance from current latent vec to all vecs in training set
                # knn = dist.topk(10, largest=False) #pick the top 10 highest
                # nearestTrainingSetLatentVectors = trainingSetLatentVectors[knn.indices]
                # additionalLatentReg = torch.mean(torch.norm((nearestTrainingSetLatentVectors - latentForOptimization).pow(2), dim=1))

                # latentVectorFromPreviousIteration = torch.clone(latentForOptimization.detach()) #store for next iteration

                # if maxThinning.detach() < maxAllowableThinning:
                #   loss_latent = 0
                # else:
                #   loss_latent = latent_Lambda * additionalLatentReg

                # loss_latent = latent_Lambda * additionalLatentReg

                loss_customBackwardPass = backwardsLambda * torch.mean(dL_ds_i * predictedSDF) #dot product

                l2_reg = 0.001 * torch.norm(latentForOptimization, p=2)

                loss_backward = loss_customBackwardPass  + similarityLoss + l2_reg# + loss_latent #<====================================================== SIMILARITY LOSS ADDED HERE #############################################
                loss_backward.backward() #calculates ds_dz (using predictedSDF) and finally dL_dz by the above chain rule

                #----------
                #Log
                #----------
                runningSimilarityMSELoss.append(similarityLoss.detach().cpu().numpy())
                runningManufacturingConstraintLoss.append(manufacturingConstraintLoss.detach().cpu().numpy())
                runningIterations.append(e)
                runningTotalBackwardLoss.append(loss_backward.detach().cpu().numpy())
                runningBackwardLoss.append(loss_customBackwardPass.detach().cpu().numpy())
                # runningLatentLoss.append(loss_latent.detach().cpu().numpy())

                #log variables
                runningBHF.append(BHF.detach().cpu().numpy() / maxBHF)
                runningFriction.append(friction.detach().cpu().numpy() / maxFriction)
                runningClearance.append(clearance.detach().cpu().numpy() / maxClearance)
                runningThickness.append(thickness.detach().cpu().numpy() / maxThickness)

                #-------------------------------------------------------------------------------------
                #2.10: Stopping criteria
                #------------------------------------------------------------------------------------- 

                # if (torch.abs(backwardLossPreviousStep - Loss.detach().cpu()) <= 0.01*(1+torch.abs(backwardLossPreviousStep))) and manufacturingConstraintLoss == 0 :
                #   break

                # backwardLossPreviousStep = Loss.detach().cpu()
                
                #-------------------------------------------------------------------------------------
                #2.11: Update parameters using the above custom back propagation and update LR if desired
                #-------------------------------------------------------------------------------------
                optimizer.step()

                # adjust_learning_rate(learningRate, optimizer, e)

                ######################################################################################
                #Part 3: VISULISE 
                ######################################################################################

                if (e == 0) or plotGIFhere:

                    trimeshMesh = trimesh.Trimesh(vertices=verts, faces=faces)
                    trimesh.repair.fix_inversion(trimeshMesh) #need this to flip the normals to get a good rendered image

                    verts = trimeshMesh.vertices
                    faces = trimeshMesh.faces

                    #-------------------------------------------------------------------------------------
                    #3.1: Rotate to match camera view (trial and error) and define tensors for plotting
                    #-------------------------------------------------------------------------------------

                    verts_dr = rotateAboutZAxis(-90-45, verts)

                    verts_dr = torch.tensor(verts_dr[None, :, :].copy(), dtype=torch.float32, requires_grad = False).to(device)  # (num_vertices, XYZ) -> (batch_size=1, num_vertices, XYZ)
                    faces_dr = torch.tensor(faces[None, :, :].copy()).to(device)

                    #-------------------------------------------------------------------------------------
                    # 3.2: Hand crafted color map (vertex textures)
                    #-------------------------------------------------------------------------------------
                    
                    textures_dr = torch.tensor(rgb).to(device)*torch.ones(verts_dr.shape[1], 3, dtype=torch.float32).unsqueeze(0).to(device)
                    meshTextures = TexturesVertex(verts_features=textures_dr).to(device)

                    #-------------------------------------------------------------------------------------
                    #3.3: Define mesh and render it to an image, then plot and save
                    #-------------------------------------------------------------------------------------

                    meshForRendering = Meshes(verts = [verts_dr.squeeze()], faces = [faces_dr.squeeze()], textures = meshTextures)
                    imageForGIF = renderer(meshForRendering, materials=materials)

                    imagesForGIF.append(process_image(imageForGIF)) #forgot to store the alpha value
                
                #-------------------------------------------------------------------------------------
                #2.9: Plot optimisation progress
                #-------------------------------------------------------------------------------------
                self.window.canvas.axes.plot(runningIterations, runningBHF)
                self.window.canvas.axes.plot(runningIterations, runningFriction)
                self.window.canvas.axes.plot(runningIterations, runningClearance)
                self.window.canvas.axes.plot(runningIterations, runningThickness)
                self.window.canvas.axes.set_xlabel("Iterations")
                self.window.canvas.axes.set_ylabel("Normalised parameter value")
                self.window.canvas.axes.legend(["BHF", "Friction", "Clearance", "Thickness"])
                # self.window.canvas.axes.set_title("Performance history")
                self.window.canvas.draw()

                if (e == 0) or (int(e+1) % 10 == 0):

                    # if (e == 0):
                    #     print("Initial shape")
                    #     plt.figure(figsize=(10, 10))
                    #     plt.imshow(imageForGIF[0, ..., :3].cpu().numpy())
                    #     plt.grid("off")
                    #     plt.axis("off")
                    #     plt.show()

                    # maxValuesList = []
                    # for kk in range(3):
                    #     maxValue = torch.abs(postStampingAndSpringbackGeometryPositions - postStampingGeometryPositions).squeeze()[kk].detach().cpu().numpy().max()
                    #     maxValuesList.append(maxValue)

                    print("Process Param Values:", [BHF.item(), friction.item(), clearance.item(), thickness.item()])
                    print("Process Param Gradients:", [BHF.grad.item(), friction.grad.item(), clearance.grad.item(), thickness.grad.item()])

                    # plt.plot(runningIterations, runningBHF)
                    # plt.plot(runningIterations, runningFriction)
                    # plt.plot(runningIterations, runningClearance)
                    # plt.plot(runningIterations, runningThickness)
                    # plt.xlabel("Iterations")
                    # plt.ylabel("Normalised parameter value")
                    # plt.legend(["BHF", "Friction", "Clearance", "Thickness"])
                    # plt.show()

                    # plt.plot(runningIterations, runningConstraint1)
                    # plt.plot(runningIterations, runningConstraint2)
                    # plt.plot(runningIterations, runningConstraint3)
                    # plt.plot(runningIterations, runningConstraint4)
                    # plt.plot(runningIterations, runningConstraint5)
                    # plt.plot(runningIterations, runningConstraint6)
                    # plt.plot(runningIterations, runningConstraint7)
                    # plt.plot(runningIterations, runningConstraint8)
                    # plt.xlabel("Iterations")
                    # plt.ylabel("Normalised parameter value")
                    # plt.legend(["maxBHF", 
                    #             "minBHF", 
                    #             "maxFriction", 
                    #             "minFriction",
                    #             "maxClearance",
                    #             "minClearance",
                    #             "maxThickness",
                    #             "minThickness"])
                    # plt.show()


                    # #plot reconstucted image
                    # print("Depth map of reconstructed shape")
                    # for i in range(surrogateModelInput.squeeze().shape[0]):
                    #     depthMapForPlotting = surrogateModelInput.squeeze()[i].detach().cpu().numpy()
                    #     plt.imshow(depthMapForPlotting, cmap='jet')
                    #     cbar = plt.colorbar(orientation="vertical")
                    #     cbar.ax.tick_params(labelsize=20)
                    #     plt.axis("off")
                    #     plt.show()

                    #     # if i == 1:
                    #     #   continue

                    # #for thinning field prediction
                    # print("Predicted thinning field")
                    # thinningFieldForPlotting = thinningField.squeeze().detach().cpu().numpy()
                    # plt.imshow(thinningFieldForPlotting, cmap='jet', vmax=maxAllowableThinning, vmin=0)
                    # # plt.colorbar(fraction=0.046, pad=0.04)
                    # cbar = plt.colorbar(fraction=0.036, pad=0.03, aspect=13)
                    # cbar.ax.tick_params(labelsize=15)
                    # plt.axis("off")
                    # fileName = os.path.join(outputPath, "thinning_" + str(e) + ".png")
                    # thinningGIFNames.append(fileName)
                    # # plt.savefig(fileName)
                    # plt.show()

                    # for coord in range(postStampingAndSpringbackGeometryPositions.squeeze().shape[0]):

                    #     print("Target displacement field")
                    #     plt.imshow(postStampingGeometryPositions.squeeze()[coord].detach().cpu().numpy(), cmap='jet')
                    #     # plt.colorbar(fraction=0.046, pad=0.04)
                    #     cbar = plt.colorbar(fraction=0.036, pad=0.03, aspect=13)
                    #     cbar.ax.tick_params(labelsize=15)
                    #     plt.axis("off")
                    #     fileName = os.path.join(outputPath, "targetDispField_" + str(e) + ".png")
                    #     thinningGIFNames.append(fileName)
                    #     # plt.savefig(fileName)
                    #     plt.show()

                    #     print("Predicted displacement field")
                    #     plt.imshow(postStampingAndSpringbackGeometryPositions.squeeze()[coord].detach().cpu().numpy(), cmap='jet')
                    #     # plt.colorbar(fraction=0.046, pad=0.04)
                    #     cbar = plt.colorbar(fraction=0.036, pad=0.03, aspect=13)
                    #     cbar.ax.tick_params(labelsize=15)
                    #     plt.axis("off")
                    #     fileName = os.path.join(outputPath, "totalDispField_" + str(e) + ".png")
                    #     thinningGIFNames.append(fileName)
                    #     # plt.savefig(fileName)
                    #     plt.show()

                    #     print("Difference")
                    #     springbackDispFieldForPlotting = torch.abs(postStampingAndSpringbackGeometryPositions - postStampingGeometryPositions).squeeze()[coord].detach().cpu().numpy()
                    #     plt.imshow(springbackDispFieldForPlotting, cmap='jet', vmax=max(maxValuesList), vmin = 0)
                    #     # plt.colorbar(fraction=0.046, pad=0.04)
                    #     cbar = plt.colorbar(fraction=0.036, pad=0.03, aspect=13)
                    #     cbar.ax.tick_params(labelsize=15)
                    #     plt.axis("off")
                    #     fileName = os.path.join(outputPath, "differenceField_" + str(e) + ".png")
                    #     thinningGIFNames.append(fileName)
                    #     # plt.savefig(fileName)
                    #     plt.show()

                    # # #for thinning field prediction
                    # # thinningFieldForPlotting = torch.where(thinningFieldMask > maxAllowableThinning, 1, 0)
                    # # thinningFieldForPlotting = thinningFieldForPlotting.item().squeeze()
                    # # plt.imshow(thinningFieldForPlotting, cmap='gray')
                    # # cbar = plt.colorbar(orientation="vertical")
                    # # cbar.ax.tick_params(labelsize=20)
                    # # plt.axis("off")
                    # # plt.show()

                    # #for dL_dv_i gradient distributions
                    # plt.figure(figsize=(20,3))
                    # plt.subplot(1,3,1)
                    # plt.hist(dL_dv_i[:,0].detach().cpu(), bins=200)
                    # plt.xlabel("X component of dL_dv_i")
                    # plt.ylabel("Count")

                    # plt.subplot(1,3,2)
                    # plt.hist(dL_dv_i[:,1].detach().cpu(), bins=200)
                    # plt.xlabel("Y component of dL_dv_i")
                    # plt.ylabel("Count")

                    # plt.subplot(1,3,3)
                    # plt.hist(dL_dv_i[:,2].detach().cpu(), bins=200)
                    # plt.xlabel("Z component of dL_dv_i")
                    # plt.ylabel("Count")
                    # plt.show()


                    # #plot losses
                    # plt.plot(runningIterations, runningManufacturingConstraintLoss)
                    # plt.plot(runningIterations, runningSpringbackLoss)
                    # # plt.plot(runningIterations, runningSimilarityMSELoss)
                    # # plt.plot(runningIterations, runningMeanThinningFieldMasked)
                    # plt.xlabel("Iterations")
                    # plt.ylabel("Log10(Losses)")
                    # plt.legend(["Manufacturing Constraint Loss", "Geometric loss"])
                    # plt.show()

                    # # plt.plot(runningIterations, runningSimilarityMSELoss)
                    # # plt.xlabel("Iterations")
                    # # plt.ylabel("Latent vector similarity loss")
                    # # plt.show()

                    # # plt.plot(runningIterations, np.log10(runningSpringbackLoss))
                    # plt.plot(runningIterations, runningMAELoss)
                    # plt.plot(runningIterations, runningPerceptualLoss)
                    # plt.xlabel("Iterations")
                    # plt.ylabel("Log10(Geometry loss components)")
                    # plt.legend(["MAE loss", "Perceptual Loss"])
                    # plt.show()

                    # plt.plot(runningIterations, runningMaxThinning)
                    # temp = np.array([[0,maxAllowableThinning],[e,maxAllowableThinning]])
                    # plt.plot(temp[:,0], temp[:,1], 'r')
                    # plt.xlabel("Iterations")
                    # plt.ylabel("Max Thinning")
                    # plt.show()

                    # # plt.plot(runningIterations, runningMeanThinningFieldMasked)
                    # # plt.xlabel("Iterations")
                    # # plt.ylabel("Mean Thinning Masked")
                    # # plt.title("Mean of thinning greater than allowable")
                    # # plt.show() 

                    # plt.plot(runningIterations, runningTotalBackwardLoss)
                    # plt.plot(runningIterations, runningSimilarityMSELoss)
                    # plt.plot(runningIterations, runningBackwardLoss)
                    # # plt.plot(runningIterations, runningLatentLoss)
                    # plt.xlabel("Iterations")
                    # plt.ylabel("Losses")
                    # plt.legend(["Total Backward Loss", "Similarity Loss", "Backward Gradients Loss", "Latent Loss"])
                    # plt.show()

                    # if plotGIFhere:
                    #     #plot rendered image
                    #     plt.figure(figsize=(10, 10))
                    #     plt.imshow(imageForGIF[0, ..., :3].cpu().numpy())
                    #     plt.grid("off")
                    #     plt.axis("off")
                    #     plt.show()
                
                    #-------------------------------------------------------------------------------------
                    print("Learning rate:", optimizer.param_groups[-1]["lr"])
                    print("DONE Iteration:", e)
                    print("==============================================================================================================================")

                #-------------------------------------------------------------------------------------
                #2.10: Save other things at intervals for plotting
                #-------------------------------------------------------------------------------------
                # if e in iterationsToPlot:
                latentVectorsForPlotting.append((e, torch.clone(latentForOptimization)))
                thinningFieldsForPlotting.append((e, thinningField.detach().cpu().numpy().squeeze()))

            # break

            # ######################################################################################
            # #Part 4: STORE OUTPUTS OF OPTIMISATION
            # ######################################################################################

            # #-------------------------------------------------------------------------------------
            # #4.0: Store .gif
            # #-------------------------------------------------------------------------------------
            # print("Optimization completed, storing GIFs...")

            # if plotGIFhere:
            #   #shape optimisation GIF
            #   imageio.mimsave(shapeGifName, imagesForGIF)

            #   #thinning GIF
            #   with imageio.get_writer(thinningGifName, mode='I') as writer:
            #     for filename in thinningGIFNames:
            #       image = imageio.imread(filename)
            #       writer.append_data(image)

            #   #remove files
            #   for filename in set(thinningGIFNames):
            #       os.remove(filename)

            # #save for plotting
            outputPath = os.path.join("temp", "Optimisation", "OptimisationOutputs", "u-bending")
            if not os.path.exists(outputPath):
                os.makedirs(outputPath)
            latentVectorsForPlottingName = os.path.join(outputPath, "LatentVectorsForPlotting.pkl")
            thinningFieldsForPlottingName = os.path.join(outputPath, "ThinningFieldsForPlotting.pkl")
            torch.save(latentVectorsForPlotting, latentVectorsForPlottingName)
            torch.save(thinningFieldsForPlotting, thinningFieldsForPlottingName)

            # lossHistoryDictionary = {"runningIterations" : runningIterations,
            #                     "runningTotalBackwardLoss" : runningTotalBackwardLoss,
            #                     "runningSimilarityMSELoss" : runningSimilarityMSELoss,
            #                     "runningBackwardLoss" : runningBackwardLoss,
            #                     "runningLatentLoss" : runningLatentLoss}

            # performanceHistoryDictionary = {"runningIterations" : runningIterations,
            #                                 "runningMaxThinning" : runningMaxThinning,
            #                                 "maxAllowableThinning" : np.array([[0,maxAllowableThinning],[len(runningIterations),maxAllowableThinning]]),
            #                                 "runningSimilarityMSELoss" : runningSimilarityMSELoss}

            # torch.save(lossHistoryDictionary, lossHistoriesFileName)
            # torch.save(performanceHistoryDictionary, performanceHistoriesFileName)

            ######################################################################################
            # Tell Qt it's done
            ######################################################################################
            self.finished.emit("u-bending")

            print("Done.")

            # record the working case details - find one case that also has high thinning
            # get process params working - possibly by setting them as a torch parameter or something?

            # match the exact same codes as before for process param optimisaion



            # print("u-bending thinning model loaded")
            # decoder.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/NN1_FinalTrained.pkl", map_location=device))
            # batch_size = 4
            # ratio = 1
            # num_channel = np.array([4,8,16,32,64,128,256,512])*ratio
            # thinningModel = manufacturingSurrogateModels_ubending.ResUNet_Thinning(num_channel,batch_size)
            # thinningModel.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/ResSEUNet_512_B4_2000_COS0.0_LRFix0.0002_E4B6D4_NewShape_08Feb23_best.pkl", map_location=device))
