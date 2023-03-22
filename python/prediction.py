
import random, os, threading, math
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
from pyqtgraph import ColorMap, ColorBarItem, mkColor, AxisItem, GradientEditorItem
from pyqtgraph.opengl import MeshData, GLScatterPlotItem, GLMeshItem
from scipy.spatial import Delaunay
import open3d as o3d
import cadquery as cq

from python import model_control, load

x_edge = 376
x_resolution = 384
x_zoom_start, x_zoom_end = 0.6583, 0.9083
y_edge = 236
y_resolution = 256
y_zoom_start, y_zoom_end = 0.425, 0.175
disp_scalar = 208
file = o3dmesh = ""
die_shape = die_shape_zoom = disp = data_colours = []

def load_mesh (file_path, window):
    global file, o3dmesh, die_shape, die_shape_zoom, disp, data_colours
    file = file_path
    file_ext = file.split(".")[-1].lower()
    # load displacement model
    model_control.load_model(window.process_dropdown.currentText(), window.material_dropdown.currentText(), 
                            "Displacement")

    print(file_ext)

    if file_ext == "stl":
        stl = mesh.Mesh.from_file(file)
    elif file_ext == "step":
        step_file = cq.importers.importStep(file)
        cq.exporters.export(step_file,"temp/STL_from_STEP.stl")
        stl = mesh.Mesh.from_file("temp/STL_from_STEP.stl")

    # predict displacement
    points = stl.points.reshape(-1,3)
    x_max, x_min = np.max(points[:,0]), np.min(points[:,0])
    y_max, y_min = np.max(points[:,1]), np.min(points[:,1])
    disp_max, disp_min = np.max(points[:,2]), np.min(points[:,2])
    input = np.zeros((3,y_resolution,x_resolution))
    # input channels 1 (die shape)
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, x_edge), np.linspace(y_max, y_min, y_edge))
    # die_shape = griddata(points[:, 0:2], disp_scalar/(disp_max-disp_min)*(points[:, 2]-disp_min),
    #                     (x_grid, y_grid), method='linear', fill_value=0)
    die_shape = griddata(points[:, 0:2], disp_max-points[:, 2],
                        (x_grid, y_grid), method='linear', fill_value=0)
    input[0,:y_edge,:x_edge] = die_shape
    # input channels 2 (zoomed die shape)
    x_grid, y_grid = np.meshgrid(np.linspace(x_min+(x_max-x_min)*x_zoom_start, x_min+(x_max-x_min)*x_zoom_end, 
                                             x_resolution), 
                        np.linspace(y_min+(y_max-y_min)*y_zoom_start, y_min+(y_max-y_min)*y_zoom_end, y_resolution))
    # die_shape_zoom = griddata(points[:, 0:2], disp_scalar/(disp_max-disp_min)*(points[:, 2]-disp_min), (x_grid, y_grid), 
    #                     method='linear', fill_value=0)
    die_shape_zoom = griddata(points[:, 0:2], disp_max-points[:, 2], (x_grid, y_grid), 
                        method='linear', fill_value=0)
    input[1,:,:] = die_shape_zoom
    # input channel 3 (blank shape)
    input[2,:,:] = 1
    disp = model_control.predict(input)
    disp_min, disp_max = np.amin(disp), np.amax(disp)
    x, y = np.meshgrid(np.arange(disp[0].shape[1]), np.arange(disp[0].shape[0]))
    x = x/np.max(x) * 1200  # convert to mm
    y = y/np.max(y) * 800  # convert to mm
    x = x + disp[0]  # apply displacement
    y = y + disp[1]  # apply displacement
    z = -disp[2]  # z height (displacement only)
    points = np.stack((x, y, z), axis=-1).reshape(-1,3)
    # Use Open3D to plot smooth surface
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Create and estimate normals
    pcd.normals =  o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd.estimate_normals()
    o3dmesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)

    # load selected model
    selected_model = window.model_dropdown.currentText()
    
    # predict value and apply colour map
    if selected_model == "Displacement":
        plot_displacement(window)

    elif selected_model == "Thinning":
        plot_thinning(window)


def change_model (window):
    global o3dmesh, die_shape, die_shape_zoom, disp, data_colours
    # load selected model
    selected_model = window.model_dropdown.currentText()
    model_control.load_model(window.process_dropdown.currentText(), window.material_dropdown.currentText(), 
                            selected_model)
    if die_shape != []:
        if selected_model == "Thinning":
            plot_thinning(window)
        elif selected_model == "Displacement":
            plot_displacement(window)
            

def gradient_changed (self, ax, window):
    add_items(ax, self, window)


def add_items (ax, gw, window):
    global data_colours
    gw.sigGradientChangeFinished.connect(lambda: gradient_changed(gw, ax, window))
    cmap = gw.colorMap()
    o3dcolours = cmap.mapToFloat(data_colours)
    meshdata = MeshData(vertexes=np.asarray(o3dmesh.vertices), faces=np.asarray(o3dmesh.triangles), vertexColors=o3dcolours)
    meshitem = GLMeshItem(meshdata=meshdata, drawFaces=True, drawEdges=True)
    window.main_view.clear()
    window.main_view.addItem(meshitem)
    window.GraphicsLayoutWidget.clear()
    window.GraphicsLayoutWidget.addItem(ax)
    window.GraphicsLayoutWidget.addItem(gw)
    print("mesh and colourbar added")


def plot_displacement (window):
    global o3dmesh, die_shape, disp, data_colours
    selected_direction = window.direction_dropdown.currentText()
    if die_shape != []:
        if selected_direction == "X":
            disp_min, disp_max = np.amin(disp[0]), np.amax(disp[0])
            o3dpoints = np.asarray(o3dmesh.vertices)
            data_colours = []
            o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]
            for point in o3dpoints:
                y = round(point[0]/o3dpointx_max[0] * 383)
                x = round(point[1]/o3dpointx_max[1] * 255)
                data_colours.append((disp[0, x, y] - disp_min)/(disp_max - disp_min))
            ax = AxisItem("left")
            ax.setLabel(text="Displacement (X)", units="mm")
            ax.setRange(disp_min, disp_max)

        elif selected_direction == "Y":
            disp_min, disp_max = np.amin(disp[1]), np.amax(disp[1])
            o3dpoints = np.asarray(o3dmesh.vertices)
            data_colours = []
            o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]
            for point in o3dpoints:
                y = round(point[0]/o3dpointx_max[0] * 383)
                x = round(point[1]/o3dpointx_max[1] * 255)
                data_colours.append((disp[1, x, y] - disp_min)/(disp_max - disp_min))
            ax = AxisItem("left")
            ax.setLabel(text="Displacement (Y)", units="mm")
            ax.setRange(disp_min, disp_max)

        elif selected_direction == "Z":
            disp_min, disp_max = np.amin(disp[2]), np.amax(disp[2])
            o3dpoints = np.asarray(o3dmesh.vertices)
            data_colours = []
            o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]
            for point in o3dpoints:
                y = round(point[0]/o3dpointx_max[0] * 383)
                x = round(point[1]/o3dpointx_max[1] * 255)
                data_colours.append((disp[2, x, y] - disp_min)/(disp_max - disp_min))
            ax = AxisItem("left")
            ax.setLabel(text="Displacement (Z)", units="mm")
            ax.setRange(disp_min, disp_max)

        else:
            disp_total = np.sum(disp, axis=0)
            disp_min, disp_max = np.amin(disp_total), np.amax(disp_total)
            o3dpoints = np.asarray(o3dmesh.vertices)
            data_colours = []
            o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]
            for point in o3dpoints:
                y = round(point[0]/o3dpointx_max[0] * 383)
                x = round(point[1]/o3dpointx_max[1] * 255)
                data_colours.append((disp_total[x, y] - disp_min)/(disp_max - disp_min))
            ax = AxisItem("left")
            ax.setLabel(text="Displacement (Total)", units="mm")
            ax.setRange(disp_min, disp_max)

        gw = GradientEditorItem(orientation="right")
        GradientMode = {'ticks': [(0, (0,255,0,255)), (0.5, (0,0,255,255)), (1, (255,0,0,255))], 'mode': 'rgb'}
        gw.restoreState(GradientMode)
        add_items(ax, gw, window)


def plot_thinning (window):
    global o3dmesh, die_shape, die_shape_zoom, disp, data_colours
    model_control.load_model(window.process_dropdown.currentText(), window.material_dropdown.currentText(), 
                                "Thinning")
        
    input = np.zeros((5,y_resolution,x_resolution))
    # input channels 1 and 3 (die and punch shape)
    input[0,:y_edge,:x_edge] = input[2,:y_edge,:x_edge] = die_shape
    # input channels 2 and 4 (zoomed die and punch shape)
    input[1,:,:] = input[3,:,:] = die_shape_zoom
    # input channel 5 (blank shape)
    input[4,:,:] = 1
    # predict using model
    pred = model_control.predict(input)
    thinning = pred[0,:]

    np.save("temp/input.npy", input)
    np.save("temp/thinning.npy", pred)

    thinning_max = np.amax(thinning)
    thinning_min = np.amin(thinning)
    o3dpoints = np.asarray(o3dmesh.vertices)
    data_colours = []
    o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]
    for point in o3dpoints:
        y = round(point[0]/o3dpointx_max[0] * 383)
        x = round(point[1]/o3dpointx_max[1] * 255)
        if x == 383:
            for i in range(8):
                data_colours.append(0)
        elif y == 255:
            for i in range(20):
                data_colours.append(0)
        else:
            data_colours.append((thinning[x, y] - thinning_min)/(thinning_max - thinning_min))
    ax = AxisItem("left")
    ax.setLabel(text="Thinning")
    ax.setRange(thinning_min, thinning_max)
    
    gw = GradientEditorItem(orientation="right")
    GradientMode = {'ticks': [(0, (0,255,0,255)), (0.5, (0,0,255,255)), (1, (255,0,0,255))], 'mode': 'rgb'}
    gw.restoreState(GradientMode)
    add_items(ax, gw, window)