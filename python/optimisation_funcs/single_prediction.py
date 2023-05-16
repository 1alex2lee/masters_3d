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



# def gradient_changed (verts, faces, ax, gw, data_colours, window):
#     add_items(verts, faces, ax, gw, data_colours, window)

def rotateAboutZAxis(angleDeg, points): #rotates point cloud about Y-axis
    theta = np.radians(angleDeg)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -1*s, 0], [s, c, 0], [0, 0, 1]])
    
    points = points.T
    rotatedPoints = np.dot(Rz, points)
    return rotatedPoints.T

def add_items (verts, faces, ax, gw, data_colours, window):
    gw.sigGradientChangeFinished.connect(lambda: add_items(verts, faces, ax, gw, data_colours, window))
    cmap = gw.colorMap()
    verts_colours = cmap.mapToFloat(data_colours)

    meshdata = MeshData(vertexes=np.asarray(verts), faces=np.asarray(faces), vertexColors=verts_colours)
    meshitem = GLMeshItem(meshdata=meshdata, drawFaces=True, drawEdges=False)

    window.main_view.clear()
    window.main_view.addItem(meshitem)
    window.GraphicsLayoutWidget.clear()
    window.GraphicsLayoutWidget.addItem(ax)
    window.GraphicsLayoutWidget.addItem(gw)
    print("mesh and colourbar added")

def bulkhead_thinning (verts, faces, window):

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

    refLengthDies = 600 #mm
    normalisedHeight = verts[:,2].max() - verts[:,2].min()
    
    batch_size = 20
    num_channel = (np.array([16,32,64,128,256,512])).astype(np.int64)
    thinningModel = manufacturingSurrogateModels_bulkhead.ResUNet(num_channel,batch_size)
    thinningModel = thinningModel.to(device)
    thinningModel.load_state_dict(torch.load("python/optimisation_funcs/model_confirugrations/bulkhead/NN2_ManufacturingSurrogateModel_BestModel.pkl", map_location=device))
    
    thinningModel.eval()

    CAMERA_DISTANCE = 1
    AZIMUTH = 0
    ELEVATION = 0
    R, T = look_at_view_transform(dist = CAMERA_DISTANCE, elev = ELEVATION, azim = AZIMUTH) 
    cameras = FoVOrthographicCameras(max_y = 1, min_y = 0, max_x = 0, min_x = -1, device=device, R=R, T=T) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu

    # Here we set the output image to be of size 256 x 256
    IMAGE_SIZE = 256
    raster_settings = RasterizationSettings(image_size = IMAGE_SIZE, blur_radius = 1e-16, faces_per_pixel = 1) #set a small blur radius to avoid -1 regions in the image (not captured by camera rays). See PyTorch3D documentation

    rasterizer_forImageLoss = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    # xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype = torch.float32, device = device)  #replace the above with this
    # faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device = device) 
    xyz_upstream = torch.tensor(verts.astype(float), dtype = torch.float32)  #replace the above with this
    faces_upstream = torch.tensor(faces.astype(float), dtype = torch.float32) 
    mesh = Meshes(verts=[xyz_upstream], faces=[faces_upstream]) #pytorch3d "Meshes" object
    # mesh = Meshes(verts=verts, faces=faces) #pytorch3d "Meshes" object
    #-------------------------------------------------------------------------------------
    #3.1: Project to image
    #-------------------------------------------------------------------------------------
    fragments = rasterizer_forImageLoss(mesh)
    # print(fragments)
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

    #-------------------------------------------------------------------------------------
    #3.2: Using NN3, calculate manufacturing performance loss
    #-------------------------------------------------------------------------------------
    thinningField = thinningModel(depthMap.unsqueeze(0).unsqueeze(0)).detach().numpy()
    # plt.imsave("temp/bulkhead_thinningfield.png",thinningField[0,0,:,:])
    # thinning_max = torch.amax(thinningField)
    # thinning_min = torch.amin(thinningField)
    thinning_max = thinningField.max()
    thinning_min = thinningField.min()
    # o3dpoints = np.asarray(o3dmesh.vertices)
    data_colours = []
    # o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]
    print(thinningField.shape)
    verts_max = [np.max(verts[:,0]), np.max(verts[:,1])]
    for point in verts:
        x = round(point[0]/verts_max[0] * 255)
        y = round(point[1]/verts_max[1] * 255)
        # if x == 255:
        #     for i in range(8):
        #         data_colours.append(0)
        # elif y == 255:
        #     for i in range(20):
        #         data_colours.append(0)
        # else:
            # data_colours.append((thinningField[0, 0, x, y] - thinning_min)/(thinning_max - thinning_min))
        data_colours.append((thinningField[0, 0, x, y] - thinning_min)/thinning_max)

    ax = AxisItem("left")
    ax.setLabel(text="Thinning")
    ax.setRange(thinning_min, thinning_max)

    gw = GradientEditorItem(orientation="right")
    GradientMode = {'ticks': [(0, (0,255,0,255)), (0.5, (0,0,255,255)), (1, (255,0,0,255))], 'mode': 'rgb'}
    gw.restoreState(GradientMode)
    add_items(verts, faces, ax, gw, data_colours, window)

#####################################################################################################################################################

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

def ubending_thinning (verts, faces, window):
    #hyperparameters surrogate model
    batch_size = 4
    num_channel_thinning = (np.array([4,8,16,32,64,128,256,512])).astype(np.int64)
    num_channel_displacement = (np.array([4,8,16,32,64,128,256,512])).astype(np.int64)

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

        surrogateModelInput[:, count, :, :] = depthMap

    surrogateModelInput[:, 2, :, :] = BHF * gridOfOnes #BHF
    surrogateModelInput[:, 3, :, :] = friction * gridOfOnes #friction
    surrogateModelInput[:, 4, :, :] = clearance * gridOfOnes #clearance
    surrogateModelInput[:, 5, :, :] = thickness * gridOfOnes #thickness

    #-------------------------------------------------------------------------------------
    #2.1: Using NN3, calculate manufacturing performance
    #-------------------------------------------------------------------------------------
    thinningField = thinningModel(surrogateModelInput)[..., :-10].detach().numpy()
    totalDisplacementField = displacementModel(surrogateModelInput).squeeze() #total displacement
    postStampingAndSpringbackGeometryPositions = displacementsToPositions(totalDisplacementField)[..., :-10] #deformed positions

    # geometryPositionsImageTo3DPoints(postStampingAndSpringbackGeometryPositions, thinningField, vmax = 0.2, vmin=0)

    # postStampingAndSpringbackGeometryPositions = torch.reshape(postStampingAndSpringbackGeometryPositions, (128512,3)).detach().numpy()
    postStampingAndSpringbackGeometryPositions = torch.squeeze(postStampingAndSpringbackGeometryPositions)
    # print(postStampingAndSpringbackGeometryPositions.shape)
    postStampingAndSpringbackGeometryPositions = postStampingAndSpringbackGeometryPositions.flatten(1,2)
    # print(postStampingAndSpringbackGeometryPositions.shape)
    postStampingAndSpringbackGeometryPositions = torch.transpose(postStampingAndSpringbackGeometryPositions, 0, 1).detach().numpy()
    # print(postStampingAndSpringbackGeometryPositions.shape)

    # Use Open3D to plot smooth surface
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(postStampingAndSpringbackGeometryPositions)
    # Create and estimate normals
    pcd.normals =  o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd.estimate_normals()
    o3dmesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    o3dverts = np.asarray(o3dmesh.vertices)

    thinning_max = thinningField.max()
    thinning_min = thinningField.min()
    # o3dpoints = np.asarray(o3dmesh.vertices)
    data_colours = []
    # o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]

    verts_max = [np.max(o3dverts[:,0]), np.max(o3dverts[:,1])]
    for point in o3dverts:
        y = round(point[0]/verts_max[0] * 501)
        x = round(point[1]/verts_max[1] * 255)
        data_colours.append((thinningField[0, 0, x, y] - thinning_min)/thinning_max)

    ax = AxisItem("left")
    ax.setLabel(text="Thinning")
    ax.setRange(thinning_min, thinning_max)

    gw = GradientEditorItem(orientation="right")
    GradientMode = {'ticks': [(0, (0,255,0,255)), (-1, (0,0,255,255)), (1, (255,0,0,255))], 'mode': 'rgb'}
    gw.restoreState(GradientMode)
    add_items(o3dmesh.vertices, o3dmesh.triangles, ax, gw, data_colours, window)

#####################################################################################################################################################

def ubending_displacement (verts, faces, window):
    direction = window.direction_dropdown.currentText()

    #hyperparameters surrogate model
    batch_size = 4
    num_channel_thinning = (np.array([4,8,16,32,64,128,256,512])).astype(np.int64)
    num_channel_displacement = (np.array([4,8,16,32,64,128,256,512])).astype(np.int64)

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

        surrogateModelInput[:, count, :, :] = depthMap

    surrogateModelInput[:, 2, :, :] = BHF * gridOfOnes #BHF
    surrogateModelInput[:, 3, :, :] = friction * gridOfOnes #friction
    surrogateModelInput[:, 4, :, :] = clearance * gridOfOnes #clearance
    surrogateModelInput[:, 5, :, :] = thickness * gridOfOnes #thickness

    #-------------------------------------------------------------------------------------
    #2.1: Using NN3, calculate manufacturing performance
    #-------------------------------------------------------------------------------------
    thinningField = thinningModel(surrogateModelInput)[..., :-10].detach().numpy()
    totalDisplacementField = displacementModel(surrogateModelInput).squeeze() #total displacement
    postStampingAndSpringbackGeometryPositions = displacementsToPositions(totalDisplacementField)[..., :-10] #deformed positions
    # print("thinning", thinningField.shape)
    # print("displacement", totalDisplacementField.shape)
    # geometryPositionsImageTo3DPoints(postStampingAndSpringbackGeometryPositions, thinningField, vmax = 0.2, vmin=0)

    # postStampingAndSpringbackGeometryPositions = torch.reshape(postStampingAndSpringbackGeometryPositions, (128512,3)).detach().numpy()
    postStampingAndSpringbackGeometryPositions = torch.squeeze(postStampingAndSpringbackGeometryPositions)
    # print(postStampingAndSpringbackGeometryPositions.shape)
    postStampingAndSpringbackGeometryPositions = postStampingAndSpringbackGeometryPositions.flatten(1,2)
    # print(postStampingAndSpringbackGeometryPositions.shape)
    postStampingAndSpringbackGeometryPositions = torch.transpose(postStampingAndSpringbackGeometryPositions, 0, 1).detach().numpy()
    # print(postStampingAndSpringbackGeometryPositions.shape)

    # Use Open3D to plot smooth surface
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(postStampingAndSpringbackGeometryPositions)
    # Create and estimate normals
    pcd.normals =  o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd.estimate_normals()
    o3dmesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    verts = np.asarray(o3dmesh.vertices)

    # Get the displacement vectors as images
    displacement_x = totalDisplacementField[0].detach().numpy()
    displacement_y = totalDisplacementField[1].detach().numpy()
    displacement_z = totalDisplacementField[2].detach().numpy()
    displacement_total = displacement_x + displacement_y + displacement_z

    if direction == "X":
        displacement_max = displacement_x.max()
        displacement_min = displacement_x.min()
        data_colours = []
        verts_max = [np.max(verts[:,0]), np.max(verts[:,1])]
        for point in verts:
            y = round(point[0]/verts_max[0] * 511)
            x = round(point[1]/verts_max[1] * 255)
            data_colours.append((displacement_x[x, y] - displacement_min)/displacement_max)
        ax = AxisItem("left")
        ax.setLabel(text="X Displacement")
        ax.setRange(displacement_min, displacement_max)

    if direction == "Y":
        displacement_max = displacement_y.max()
        displacement_min = displacement_y.min()
        data_colours = []
        verts_max = [np.max(verts[:,0]), np.max(verts[:,1])]
        for point in verts:
            y = round(point[0]/verts_max[0] * 511)
            x = round(point[1]/verts_max[1] * 255)
            data_colours.append((displacement_y[x, y] - displacement_min)/displacement_max)
        ax = AxisItem("left")
        ax.setLabel(text="Y Displacement")
        ax.setRange(displacement_min, displacement_max)

    if direction == "Z":
        displacement_max = displacement_z.max()
        displacement_min = displacement_z.min()
        data_colours = []
        verts_max = [np.max(verts[:,0]), np.max(verts[:,1])]
        for point in verts:
            y = round(point[0]/verts_max[0] * 511)
            x = round(point[1]/verts_max[1] * 255)
            data_colours.append((displacement_z[x, y] - displacement_min)/displacement_max)
        ax = AxisItem("left")
        ax.setLabel(text="Z Displacement")
        ax.setRange(displacement_min, displacement_max)

    if direction == "Total":
        displacement_max = displacement_total.max()
        displacement_min = displacement_total.min()
        data_colours = []
        verts_max = [np.max(verts[:,0]), np.max(verts[:,1])]
        for point in verts:
            y = round(point[0]/verts_max[0] * 511)
            x = round(point[1]/verts_max[1] * 255)
            data_colours.append((displacement_total[x, y].sum() - displacement_min)/displacement_max)
        ax = AxisItem("left")
        ax.setLabel(text="Total Displacement")
        ax.setRange(displacement_min, displacement_max)

    gw = GradientEditorItem(orientation="right")
    GradientMode = {'ticks': [(0, (0,255,0,255)), (-1, (0,0,255,255)), (1, (255,0,0,255))], 'mode': 'rgb'}
    gw.restoreState(GradientMode)
    add_items(o3dmesh.vertices, o3dmesh.triangles, ax, gw, data_colours, window)
