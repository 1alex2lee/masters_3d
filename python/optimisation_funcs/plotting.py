import torch
# from torch import nn
# import time
# import random
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.spatial import cKDTree as KDTree
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d import ops
import open3d
import plotly.graph_objs as go
# import plotly.figure_factory as ff
# import plotly.offline as offline

def rotateAboutZAxis(angleDeg, points): #rotates point cloud about Y-axis
    theta = np.radians(angleDeg)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -1*s, 0], [s, c, 0], [0, 0, 1]])
    
    points = points.T
    rotatedPoints = np.dot(Rz, points)
    return rotatedPoints.T

def get_threed_scatter_trace(points,caption = None,colorscale = None,color = "orange"):

    if (type(points) == list):
        trace = [go.Scatter3d(
            x=p[0][:, 0],
            y=p[0][:, 1],
            z=p[0][:, 2],
            mode='markers',
            name=p[1],
            marker=dict(
                size=2,
                # line=dict(width=2),
                line = None,
                opacity=0.9,
                colorscale=colorscale,
                showscale=True,
                color=color,
            ), text=caption) for p in points]

    else:

        trace = [go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            name='projection',
            marker=dict(
                size=2,
                # line=dict(width=2),
                line = None,
                opacity=0.9,
                colorscale=colorscale,
                showscale=False,
                color=color,
            ), text=caption)]

    return trace

def get_threed_quiver_trace(surfacePoints, surfaceNormals, step):
  trace = [go.Cone(
    x=surfacePoints[::step,0],
    y=surfacePoints[::step,1],
    z=surfacePoints[::step,2],
    u=surfaceNormals[::step,0],
    v=surfaceNormals[::step,1],
    w=surfaceNormals[::step,2],
    colorscale=[[0, "rgb(201, 18,18)"], [1, "rgb(140, 201,18)"]],
    # colorscale = "jet",
    showscale=False,
    sizemode="scaled",
    sizeref=20,
    anchor="tail")]

  return trace

def get_surface_trace(decoder,latentVector,marchingCubesResolution,mc_value,inputPoints,verbose,save_ply, surfaceColour="blue", opacity=1):

    trace = []
    meshexport = None
    z = []

    #-------------------------------------------------------------------------------------
    #predict SDF on uniform grid vertices using the trained network and run MC
    #-------------------------------------------------------------------------------------
    for i,pnts in enumerate(torch.split(inputPoints,100000,dim=0)):
        if (verbose):
            print ('{0}'.format(i/(inputPoints.shape[0] // 100000) * 100))

        latentInputs = latentVector.expand(pnts.shape[0], -1)
        predictedSDF = decoder(latentInputs, pnts)
        predictedSDF = predictedSDF.detach().cpu().numpy().squeeze()
        z.append(predictedSDF)

    z = np.concatenate(z,axis=0)
    z = z.astype(np.float64)
    z = z.reshape(marchingCubesResolution, marchingCubesResolution, marchingCubesResolution)

    verts, faces, normals, values = measure.marching_cubes_lewiner(volume=z,level=mc_value)

    #-------------------------------------------------------------------------------------
    #Transform the output of MC to match the training data. I.e., the configuration that the network was trained on
    #-------------------------------------------------------------------------------------
    #rotate
    # verts = rotateAboutYAxis(90 + 180, verts)
    # verts = rotateAboutZAxis(90 + 180, verts)

    #translate
    verts[:,0] = verts[:,0] - verts[:,0].min()
    verts[:,1] = verts[:,1] - verts[:,1].min()
    verts[:,2] = verts[:,2] - verts[:,2].max()
    verts[:,2] = verts[:,2] + (verts[:,2].max() - verts[:,2].min()) / 2 #offset Z dimention to be exactly in the centre of a unit cube

    #scale
    verts = verts/(verts[:,0].max() - verts[:,0].min()) #ensure side length of 1

    if (save_ply):
        meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

    def tri_indices(simplices):
        return ([triplet[c] for triplet in simplices] for c in range(3))

    pl_mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']

    I, J, K = tri_indices(faces)
    trace.append(go.Mesh3d(x=verts[:, 0], 
                           y=verts[:, 1], 
                           z=verts[:, 2],
                           colorscale=pl_mygrey, 
                          #  intensity= z, 
                           i=I, 
                           j=J, 
                           k=K, 
                           name='', 
                           color=surfaceColour, 
                           opacity=opacity,
                           flatshading=True,
                           showscale=False)
    )

    return {"mesh_trace":trace,
            "mesh_export":meshexport}

def plot_surface(decoder, 
                 marchingCubesResolution,
                 verbose,
                 gridPoints=None, 
                 additionalPoints = None, 
                 showPredictedSurface = True,
                 with_points=False, 
                 with_normals = False,
                 with_gtSurface = False, 
                 with_mesh = False,
                 surfacePoints=None, 
                 surfaceNormals=None, 
                 step=None,
                 GTMesh=None, 
                 latentVector=None,
                 predictedSurfaceOpacity = 1):

    if with_points:
        trace_pnts = get_threed_scatter_trace(points=additionalPoints)

    if with_normals:
        trace_normals = get_threed_quiver_trace(surfacePoints, surfaceNormals, step)

    if with_gtSurface:
        trace_GTMesh = get_GT_surface_trace(mesh=GTMesh)

    surface = get_surface_trace(decoder=decoder, 
                                latentVector=latentVector, 
                                marchingCubesResolution=marchingCubesResolution, 
                                mc_value=0, 
                                inputPoints=gridPoints, 
                                verbose=verbose, 
                                save_ply=False,
                                opacity=predictedSurfaceOpacity)

    trace_surface = surface["mesh_trace"]

    layout = go.Layout(
                       width=1200, 
                       height=1200, 
                       scene=dict(xaxis=dict(range=[0, 1], autorange=False),
                          yaxis=dict(range=[0, 1], autorange=False),
                          zaxis=dict(range=[-0.5, 0.5], autorange=False),
                          aspectratio=dict(x=1, y=1, z=1)),
                       )
    
    fig1 = go.Figure(layout=layout)
    if showPredictedSurface:
        fig1.add_trace(trace_surface[0])

    if with_points:
        fig1.add_trace(trace_pnts[0])

    if with_normals:
        fig1.add_trace(trace_normals[0])

    if with_gtSurface:
        fig1.add_trace(trace_GTMesh[0])

    trace_surface[0].update(cmin=(trace_surface[0]["z"].min() + 0.1),# atrick to get a nice plot (z.min()=-3.31909)
                            lighting=dict(ambient=0.18,
                                          diffuse=1,
                                          fresnel=0.1,
                                          specular=1,
                                          roughness=0.05,
                                          facenormalsepsilon=1e-15,
                                          vertexnormalsepsilon=1e-15),
                            lightposition=dict(x=1.2,
                                                y=1.2,
                                                z=0.5
                                              )
                           )
    
    if with_mesh:
        #add triangles
        Xe = []
        Ye = []
        Ze = []
        triangles = np.array([trace_surface[0]["i"], trace_surface[0]["j"], trace_surface[0]["k"]]).T
        vertices = np.array([trace_surface[0]["x"], trace_surface[0]["y"], trace_surface[0]["z"]]).T

        tri_points = vertices[triangles] 
        for T in tri_points:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])
              
        #define the trace for triangle sides
        lines = go.Scatter3d(
                          x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          name='',
                          line=dict(color= 'rgb(70,70,70)', width=5))  
        
        fig1.add_trace(lines)

    fig1.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
                                camera_eye=dict(x=1.2, y=1.2, z=0.6),
                                xaxis = dict(visible=False),
                                yaxis = dict(visible=False),
                                zaxis = dict(visible=False)),
                      margin_l=1,
                      margin_r=1,
                      margin_t=1,
                      margin_b=1)
    
    fig1.layout.scene.camera.projection.type = "orthographic"
    # fig1.layout.scene.camera.projection.type = "perspective"

    fig1.show()

    return surface['mesh_export']

def generateMeshOrImageFromSDF(device, decoder, latentVector, inputPoints, marchingCubesResolution, rasterizer, exportMeshOnly=False, verbose=True):

  """Using a trained decoder and inferred latent vector, this function first predicts a 
  reconstructed surface via running marhcing cubes on the predicted implicit field.
  Then, projects the surface onto a 2D image using a differentiable rasteriser
  
  Arguments
  ---------
  Decoder : Trained decoder

  latentVector : torch.tensor
    Latent vector previously inferred

  inputPoints : torch.tensor
    3D grid XYZ points

  marchingCubesResolution : int or float

  rasterizer : pytorch3d.renderer.mesh.rasterizer.MeshRasterizer
    Differentiable rasterizer used to project 3D mesh onto image

  exportMeshOnly : bool
    if true, exports the extracted mesh witout projection onto image


  Returns
  -------
  if exportMeshOnly: 
    trimesh.base.Trimesh mesh object
  else:
    depth map image
  """

  #-------------------------------------------------------------------------------------
  #Part 1: predict SDF on uniform grid vertices using the trained network and run MC
  #-------------------------------------------------------------------------------------
  z = []
  for i,pnts in enumerate(torch.split(inputPoints,100000,dim=0)):
      if (verbose):
          print ('{0}'.format(i/(inputPoints.shape[0] // 100000) * 100))

      latentInputs = latentVector.expand(pnts.shape[0], -1)
      predictedSDF = decoder(latentInputs, pnts)
      predictedSDF = predictedSDF.detach().cpu().numpy().squeeze()
      z.append(predictedSDF)

  z = np.concatenate(z,axis=0)
  z = z.astype(np.float64)
  z = z.reshape(marchingCubesResolution, marchingCubesResolution, marchingCubesResolution)

  verts, faces, _, _ = measure.marching_cubes_lewiner(volume=z,level=0)

  #-------------------------------------------------------------------------------------
  #Part 2: Transform
  #-------------------------------------------------------------------------------------

  if not exportMeshOnly:
    #rotate only if image is output
    # verts = rotateAboutYAxis(90 + 180, verts)
    verts = rotateAboutZAxis(90 + 180, verts)

  #translate
  verts[:,0] = verts[:,0] - verts[:,0].min()
  verts[:,1] = verts[:,1] - verts[:,1].min()
  verts[:,2] = verts[:,2] - verts[:,2].max()
  verts[:,2] = verts[:,2] + (verts[:,2].max() - verts[:,2].min()) / 2 #offset Z dimention to be exactly in the centre of a unit cube

  #scale
  verts = verts/(verts[:,0].max() - verts[:,0].min()) #ensure side length of 1
  normalisedHeight = verts[:,2].max() - verts[:,2].min()

  if exportMeshOnly:
    mesh = trimesh.Trimesh(verts, faces)
    return mesh

  #-------------------------------------------------------------------------------------
  #Part 3: Prepare for rasteriser
  #-------------------------------------------------------------------------------------
  #reduce mesh to increase speed of rasterizer
  temporaryMesh = trimesh.Trimesh(verts, faces)
  temporaryMesh = temporaryMesh.simplify_quadratic_decimation(5000) #first time runing this takes a while since it imports open3d library
  verts = temporaryMesh.vertices
  faces = temporaryMesh.faces

  verts = torch.tensor(verts.astype(float), requires_grad = True, dtype = torch.float32, device = device) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu
  faces = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device = device) #<============= DEVICE SET AS CPU FOR NOW SINCE MeshRasteriser not working properly on GPu
  mesh = Meshes(verts=[verts], faces=[faces])

  #-------------------------------------------------------------------------------------
  #Part 4: Project to image
  #-------------------------------------------------------------------------------------
  fragments = rasterizer(mesh)
  depthMap = fragments.zbuf.squeeze() #no need for a renderer just use the rasterizer to obtain the depth map
  depthMap = depthMap.to('cpu')

  # #correct the scaling due to camera view
  depthMap = depthMap - depthMap.min() #ensure min value of 0
  depthMap = depthMap / depthMap.max() #scale between 1 and 0
  depthMap = depthMap * normalisedHeight #correct the hight
  depthMap = depthMap - normalisedHeight/2 #offset Z direction so that Z=0 is exactly at the mid height - matches training SDF data
  depthMap = torch.rot90(depthMap, 0)

  return depthMap
