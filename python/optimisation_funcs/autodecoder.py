import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure

from python.optimisation_funcs.implicitShapeNetworkModels import *

selected_component = "bulkhead"

def get_active_component ():
    global selected_component
    return selected_component

def get_latent_vector(surfacePoints_all, surfaceNormals_all, offSurfacePoints_all, worker, component):
    global selected_component
    selected_component = component
    # --------------------------------------------------------------------------
    # INFERENCE OPTIMISATION
    # Decoder weights are fixed and an optimal1 latent vector is estimaeted
    # based on minimising the IGR loss for each shape.
    # Repeated for all shapes in training, testing and select sets.
    # --------------------------------------------------------------------------

    # Hyper parameters
    random_seed = 37
    learningRate_latent = 1e-3  # actually used
    # learningRate_latent = 0.00001 #to get nice figure for paper
    LrReductionInterval = 200
    # num_iterations= 300 #actually used
    num_iterations= 10
    # num_iterations = 500  # to get nice figure for paper

    if component.lower() == "bulkhead":
        latentVectorLength = 256
        hiddenLayerSizes = 512
    if component.lower() == "u-bending":
        latentVectorLength = 64
        hiddenLayerSizes = 128

    # convergencePatients = 50
    # convergenceTolerance = 0.001

    normalsAlpha = 8
    lambda_grad = 0.9
    lambda_normals = 0.7
    lambda_latent = 1
    lambda_surface = 3

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(4)
    # torch.get_num_threads()

    # Set random seeds
    random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)

    numShapes = surfacePoints_all.shape[0]

    latentPath = os.path.join("temp", "best_latent_vector", "Shape_1.pkl")

    folderName = latentPath.rsplit("/", 1)[0]
    # folderName = "/".join(latentPath.split("/")[:-1])

    print(folderName)
    if not os.path.exists(folderName):
        Path(folderName).mkdir(parents=True, exist_ok=True) #create correct folder structure if not exists already

    # load decoder
    networkSettings = {
        "dims": [hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes, hiddenLayerSizes],
        "skip_in": [4],
        "geometric_init": True,
        "radius_init": 1,
        "beta": 100,
    }
    decoder = ImplicitNet(z_dim=latentVectorLength,
                        dims=networkSettings["dims"],
                        skip_in=networkSettings["skip_in"],
                        geometric_init=networkSettings["geometric_init"],
                        radius_init=networkSettings["radius_init"],
                        beta=networkSettings["beta"]).to(device)
    
    if component.lower() == "bulkhead":
        decoder.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/bulkhead/NN1_ImplicitRepresentationDies_FinalTrained_NN1.pkl", map_location=device))
    if component.lower() == "u-bending":
        decoder.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/NN1_FinalTrained.pkl", map_location=device))


    def computeGradient_IGR(inputs, outputs):
        d_points = torch.ones_like(
            outputs, requires_grad=False, device=outputs.device)
        points_grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0][:, -3:]
        return points_grad
    

    surfacePoints_i = torch.from_numpy(surfacePoints_all).float().to(device) 
    surfaceNormals_i = torch.from_numpy(surfaceNormals_all).float().to(device) 
    offSurfacePoints_i = torch.from_numpy(offSurfacePoints_all).float().to(device) 

    surfacePoints_i.requires_grad = True
    surfaceNormals_i.requires_grad = False
    offSurfacePoints_i.requires_grad = True

    latentVector = torch.ones(1, latentVectorLength).normal_(mean = 0, std = 0.0001).to(device)
    latentVector.requires_grad = True

    optimizer = torch.optim.Adam([latentVector], lr = learningRate_latent)

    # # initialize the early_stopping object
    # early_stopping = EarlyStopping(patience = convergencePatients, verbose = True, delta = convergenceTolerance)

    runningInferenceLoss = []
    runningGradLoss = []
    runningSDFLoss = []
    runningNormalsLoss = []
    runningLatentLoss = []
    inferenceIterations = []
    numPoints_surface = surfacePoints_i.shape[0]
    numPoints_offSurface = offSurfacePoints_i.shape[0]

    bestLoss = 100

    latentInputs_surface = latentVector.expand(numPoints_surface, -1)
    latentInputs_offSurface = latentVector.expand(numPoints_offSurface, -1)

    debug = True

    for e in range(num_iterations):

        ######################################################################################
        # UPDATE PROGRESS IN QT
        ######################################################################################

        if hasattr(worker, "progress"):
            worker.progress.emit(100 * (e + 1)/num_iterations)
            
        #forward pass
        SDFPredicted_surface =  decoder(latentInputs_surface, surfacePoints_i)
        SDFPredicted_OffSurface =  decoder(latentInputs_offSurface, offSurfacePoints_i)

        gradient_surface = computeGradient_IGR(surfacePoints_i, SDFPredicted_surface) #calculate partial gradients with respect to x y z positions
        gradient_OffSurface = computeGradient_IGR(offSurfacePoints_i, SDFPredicted_OffSurface) #calculate partial gradients with respect to x y z positions

        #surface points loss
        surfacePoints_loss = (SDFPredicted_surface.abs()).mean()

        #Eikonal loss
        grad_loss = ((gradient_OffSurface.norm(2, dim=-1) - 1) ** 2).mean()
        Loss = lambda_surface*surfacePoints_loss + lambda_grad*grad_loss

        #normals loss
        nonVerticalNormalsIdx = ((surfaceNormals_i[:,2] <= 0.99) + 0) #give a large weight to normals on the radius and on the sidewall
        verticalNormalsIdx = ((surfaceNormals_i[:,2] > 0.99) + 0) #give a low weight to normals on the flat surfaces
        normalsWeights = verticalNormalsIdx + normalsAlpha*nonVerticalNormalsIdx

        normals_loss = (((gradient_surface - surfaceNormals_i).abs()).norm(2, dim=1) * normalsWeights).mean()
        Loss = Loss + lambda_normals*normals_loss
        LossWithoutLatentReg = Loss

        #regularisation with z vector
        sumL2LatentVectors = torch.sum(torch.norm(latentInputs_surface, dim = 1)) + torch.sum(torch.norm(latentInputs_offSurface, dim = 1))
        latent_loss =  (min(1, e / 100) * sumL2LatentVectors) / (numPoints_surface + numPoints_offSurface)
        Loss = Loss + lambda_latent*latent_loss

        #updates
        # adjust_learning_rate(learningRate_latent, optimizer, e)

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        runningInferenceLoss.append(Loss.detach().cpu().numpy())
        runningGradLoss.append(lambda_grad * grad_loss.detach().cpu().numpy())
        runningSDFLoss.append(lambda_surface * surfacePoints_loss.detach().cpu().numpy())
        runningNormalsLoss.append(lambda_normals * normals_loss.detach().cpu().numpy())
        runningLatentLoss.append(lambda_latent * latent_loss.detach().cpu().numpy())
        inferenceIterations.append(e)

        # window.progressbar.setValue(100 * e/num_iterations) # update window progress bar

        # if e % 10 == 0: # every 10th iteration
        #     window.progress_label.setText("Geometry load progress {} / {}".format(e, num_iterations)) # update progress text

        if e % 100 == 0: #every 100th iteration
            print("Shape: {} | Iteration: {} / {} | Loss: {} | LR: {}".format(numShapes, e, num_iterations, Loss.detach(), optimizer.state_dict()["param_groups"][0]["lr"]))

            # print('--> Shape: ', i+1, ' done')
            # plt.plot(inferenceIterations, runningInferenceLoss)
            # plt.plot(inferenceIterations, runningGradLoss)
            # plt.plot(inferenceIterations, runningSDFLoss)
            # plt.plot(inferenceIterations, runningNormalsLoss)
            # plt.plot(inferenceIterations, runningLatentLoss)
            # plt.xlabel("Inference Iterations")
            # plt.ylabel("Loss Components")
            # plt.legend(['Total', 'Grad Loss', 'SDF Loss', 'Normals Loss', 'Latent Loss'])
            # plt.show()

        if LossWithoutLatentReg.detach().cpu().numpy() < bestLoss:
            #note that the dip in the loss is due to the regularisation term min(1, e/100); e/100 increases between iterations 1 to 100.
            bestLoss = LossWithoutLatentReg.detach().cpu().numpy()
            bestLatentVector = latentVector
            print("### -> New best loss at iteration:", e)

    # if not early_stopping.early_stop:
    # print('--> Shape: ', i+1, ' done')
    # plt.plot(inferenceIterations, runningInferenceLoss)
    # plt.plot(inferenceIterations, runningGradLoss)
    # plt.plot(inferenceIterations, runningSDFLoss)
    # plt.plot(inferenceIterations, runningNormalsLoss)
    # plt.plot(inferenceIterations, runningLatentLoss)
    # plt.xlabel("Inference Iterations")
    # plt.ylabel("Loss Components")
    # plt.legend(['Total', 'Grad Loss', 'SDF Loss', 'Normals Loss', 'Latent Loss'])
    # plt.show()

    # --------------------------------------------------------------------------
    # SAVE FINAL STATE - SHAPE i - INFERENCE OPTIMISATION 
    # --------------------------------------------------------------------------
    torch.save(bestLatentVector, latentPath)

    #   if i == 2:
    #     break
    # break

    print("### BEST LATENT VECTOR SAVED ###")

    ######################################################################################
    # Tell Qt it's done
    ######################################################################################
    if hasattr(worker, "finished"):
        worker.finished.emit()

    return bestLatentVector


def get_verts_faces (latentForOptimization, window, component):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if component.lower() == "bulkhead":
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

        decoder.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/bulkhead/NN1_ImplicitRepresentationDies_FinalTrained_NN1.pkl", map_location=device))
       
        #assemble uniform grid
        marchingCubesResolution = 90
        X, Y, Z = np.mgrid[0:1:complex(marchingCubesResolution), 0:1:complex(marchingCubesResolution), -0.5:0.5:complex(marchingCubesResolution)]
        inputPoints = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        inputPoints = torch.tensor(inputPoints).float().to(device) 
        numGridPoints = inputPoints.shape[0]

        z = []

        for i,pnts in enumerate(torch.split(inputPoints,100000,dim=0)):
        # for i,pnts in enumerate(torch.split(inputPoints,1,dim=0)):

            latentInputs = latentForOptimization.expand(pnts.shape[0], -1)
            # latentInputs = latentForOptimization
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
        # verts, faces, _, _ = measure.marching_cubes(volume=z1,level=0.1) # changed

        return verts, faces
    
    if component.lower() == "u-bending":
        print("decoder for u-bending")
        hiddenLayerSizes = 128
        latentVectorLength = 64
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
        decoder.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/u-bending/NN1_FinalTrained.pkl", map_location=device))

        # Define the ranges for x, y, and z
        min_x, max_x = 0, 1
        min_y, max_y = 0, 0.5
        min_z, max_z = -0.5, 0.5
        
        refinementFactor = 1 #remember during optimisation, this must match what was used to generate the NN2 training input images
        baseResolution = 90
        nx, ny, nz = int(baseResolution*(max_x-min_x))*refinementFactor, int(baseResolution*(max_y-min_y))*refinementFactor, int(baseResolution*(max_z-min_z))*refinementFactor
        
        # Generate a regularly spaced grid of data points
        x = np.linspace(min_x, max_x, nx)
        y = np.linspace(min_y, max_y, ny)
        z = np.linspace(min_z, max_z, nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        # Combine xx, yy, and zz into a single array
        inputPoints = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
        inputPoints = torch.tensor(inputPoints).float().to(device)

        z = []
        for _,pnts in enumerate(torch.split(inputPoints,100000,dim=0)):

            latentInputs = latentForOptimization.expand(pnts.shape[0], -1)
            predictedSDF = decoder(latentInputs, pnts)
            predictedSDF = predictedSDF.detach().cpu().numpy().squeeze()
            z.append(predictedSDF)

        z = np.concatenate(z,axis=0)
        z = z.astype(np.float64)
        z = z.reshape(nx, ny, nz)
        # z = z.reshape(marchingCubesResolution, marchingCubesResolution, marchingCubesResolution)

        #-------------------------------------------------------------------------------------
        #1.2: Run marching cubes to extract the mesh
        #-------------------------------------------------------------------------------------
        verts, faces, _, _ = measure.marching_cubes(volume=z,level=0)
        # verts, faces, _, _ = measure.marching_cubes(volume=z1,level=0.1) # changed

        return verts, faces