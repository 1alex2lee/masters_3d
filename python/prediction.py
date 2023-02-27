
import random, os, threading
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata

from python import model_control, load


def load_mesh (file):
    print(file)
    stl = mesh.Mesh.from_file(file)
    points = stl.points.reshape(-1,3)
    # print(points.shape)

    x_max, x_min = np.max(points[:,0]), np.min(points[:,0])
    y_max, y_min = np.max(points[:,1]), np.min(points[:,1])
    z_max, z_min = np.max(points[:,2]), np.min(points[:,2])

    np.save(os.path.join("temp","points.npy"),points)
    
    x_edge = 376
    x_resolution = 384
    x_zoom_start, x_zoom_end = 0.6583, 0.9083
    y_edge = 236
    y_resolution = 256
    y_zoom_start, y_zoom_end = 0.425, 0.175
    z_scalar = 208

    input = np.zeros((5,y_resolution,x_resolution))

    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, x_edge), np.linspace(y_max, y_min, y_edge))
    input[0,:y_edge,:x_edge]=input[2,:y_edge,:x_edge] = griddata(points[:, 0:2], z_scalar/(z_max-z_min)*(points[:, 2]-z_min), (x_grid, y_grid), method='linear', fill_value=0)

    x_grid, y_grid = np.meshgrid(np.linspace(x_min+(x_max-x_min)*x_zoom_start, x_min+(x_max-x_min)*x_zoom_end, x_resolution), 
                        np.linspace(y_min+(y_max-y_min)*y_zoom_start, y_min+(y_max-y_min)*y_zoom_end, y_resolution))
    input[1,:,:]=input[3,:,:] = griddata(points[:, 0:2], z_scalar/(z_max-z_min)*(points[:, 2]-z_min), (x_grid, y_grid), method='linear', fill_value=0)

    input[4,:,:] = 1

    # print("x ", x_max," ", x_min)
    # print("y ", y_max," ", y_min)
    # print("z ", z_max," ", z_min)

    output = model_control.predict(input)

    # np.save(os.path.join("temp","input.npy"),input)
    # np.save(os.path.join("temp","output.npy"),output)

    points = np.zeros((256*384,3))
    points[:,0] = np.tile(np.arange(256),384)

    thinning = np.multiply(input[0],1-output[0])

    for i in range(384):
        points[256*i:256*(i+1),1] = i

    for i in range(384):
        for j in range(256):
            points[(i*256)+j,2] = thinning[j,i]

    faces = np.arange(points.shape[0]).reshape(-1, 3)

    return points, faces


def parameters (model):
    model = str(model).lower()
    print("parameters for ",model," requested")
    

def metrics (model):
    model = str(model).lower()
    print("metrics for ",model," requested")
    

def update (metric):
    metric = str(metric).lower()
    print("result for ",metric," requested")
    return str(round(100*random.random())) + " %"


def receive_inputvalue (value, input, qml):

    if input == "Handle Groove Depth (H)":

        ub = load.process_input_upperbound(input)
        scalar = value/ub

        input = np.load(os.path.join(os.getcwd(),"temp","input.npy"))
        input[2,:,:] = scalar * input[0,:,:]
        input[3,:,:] = scalar * input[1,:,:]
        input[0,:,:] = scalar * input[0,:,:]
        input[1,:,:] = scalar * input[1,:,:]

        pred = model_control.predict(input)

        plt.imshow(pred[0,:,:])
        plt.colorbar()
        plt.savefig(os.path.join(os.getcwd(),"temp","outputfield1.png"))
        plt.savefig(os.path.join(os.getcwd(),"temp","outputfield2.png"))
        plt.clf()
        # plt.imsave(os.path.join(os.getcwd(),"temp","outputfield1.png"),pred[0,:,:])
        # plt.imsave(os.path.join(os.getcwd(),"temp","outputfield2.png"),pred[0,:,:])

        qml.pred_updated.emit()
