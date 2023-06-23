
import random, os, threading, math
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
from pyqtgraph import ColorMap, ColorBarItem, mkColor, AxisItem, GradientEditorItem
from pyqtgraph.opengl import MeshData, GLScatterPlotItem, GLMeshItem, GLAxisItem
from PyQt6 import QtGui
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
die_shape = die_shape_zoom = disp = data_colours = blank_shape = []

def load_mesh (file_path, window, L1, L2):
    global file, o3dmesh, die_shape, die_shape_zoom, disp, data_colours, blank_shape
    file = file_path
    file_ext = file.split(".")[-1].lower()
    # load displacement model
    model_control.load_model(window.component_dropdown.currentText(), window.process_dropdown.currentText(), window.material_dropdown.currentText(), "Displacement (mm)")

    # print(file_ext)

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
    L1 = L1/800 * y_resolution
    L2 = L2/800 * y_resolution
    # Calculate the slope of the line
    slope = ((L1 - L2)/x_resolution)
    # Iterate over each row and interpolate the values
    input[2,:,:] = 1
    for x in range(x_resolution - 1):
        edge_length = int(x * slope + L2)
        input[2, edge_length:, x] = 0
    blank_shape = input[2,...]
    # make prediction
    disp = model_control.predict(input)
    disp_min, disp_max = np.amin(disp), np.amax(disp)
    x, y = np.meshgrid(np.arange(disp[0].shape[1]), np.arange(disp[0].shape[0]))
    x = x/np.max(x) * 1200  # convert to mm
    y = y/np.max(y) * 800  # convert to mm
    x = np.add(x, disp[0])  # apply displacement
    y = np.add(y, disp[1])  # apply displacement
    z = -disp[2]  # z height (displacement only)
    points = np.stack((x, y, z), axis=-1).reshape(-1,3)
    # Use Open3D to plot smooth surface
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Create and estimate normals
    pcd.normals =  o3d.utility.Vector3dVector(np.zeros((1,3)))
    pcd.estimate_normals()
    o3dmesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # load selected model
    selected_indicator = window.indicator_dropdown.currentText()
    
    # predict value and apply colour map
    if "displacement" in selected_indicator.lower():
        plot_displacement(window)

    elif "thinning" in selected_indicator.lower():
        plot_thinning(window)


def change_model (window):
    global o3dmesh, die_shape, die_shape_zoom, disp, data_colours
    # load selected model
    selected_indicator = window.indicator_dropdown.currentText()
    model_control.load_model(window.component_dropdown.currentText(), window.process_dropdown.currentText(), window.material_dropdown.currentText(), selected_indicator)
    if die_shape != []:
        if "thinning" in selected_indicator.lower():
            plot_thinning(window)
        elif "displacement" in selected_indicator.lower():
            plot_displacement(window)
            

def gradient_changed (self, ax, window):
    add_items(ax, self, window)


def add_items (ax, gw, window):
    global data_colours
    gw.sigGradientChangeFinished.connect(lambda: gradient_changed(gw, ax, window))
    cmap = gw.colorMap()
    o3dcolours = cmap.mapToFloat(data_colours)
    meshdata = MeshData(vertexes=np.asarray(o3dmesh.vertices), faces=np.asarray(o3dmesh.triangles), vertexColors=o3dcolours)
    meshitem = GLMeshItem(meshdata=meshdata, drawFaces=True, drawEdges=False)
    window.main_view.clear()
    axis = GLAxisItem(size=QtGui.QVector3D(2000,2000,2000), glOptions="opaque")
    window.main_view.addItem(axis)
    window.main_view.addItem(meshitem)
    window.GraphicsLayoutWidget.clear()
    window.GraphicsLayoutWidget.addItem(ax)
    window.GraphicsLayoutWidget.addItem(gw)
    print("mesh and colourbar added")


def plot_displacement (window):
    global o3dmesh, die_shape, disp, data_colours

    def get_displacement_data_colours (disp_field, dir):
        disp_field = np.transpose(disp_field)
        plt.imsave("temp/cardoorpanel_disp.png", disp_field)
        disp_min, disp_max = np.amin(disp_field), np.amax(disp_field)
        o3dpoints = np.asarray(o3dmesh.vertices)
        x_max, x_min = np.amax(o3dpoints[:,0]), np.amin(o3dpoints[:,0])
        y_max, y_min = np.amax(o3dpoints[:,1]), np.amin(o3dpoints[:,1])
        field_x, field_y = disp_field.shape[0], disp_field.shape[1]
        data_colours = []
        # o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]
        for idx in range(o3dpoints.shape[0]):
            x = math.floor((o3dpoints[idx, 0] - x_min)/(x_max-x_min) * (field_x-1))
            y = math.floor((o3dpoints[idx, 1] - y_min)/(y_max-y_min) * (field_y-1))
            data_colours.append(disp_field[x, y])
        data_colours = np.array(data_colours)
        data_colours -= np.min(data_colours)
        data_colours /= np.max(data_colours)
        ax = AxisItem("left")
        ax.setLabel(text=f"{dir} Displacement", units="mm")
        ax.setRange(disp_min, disp_max)
        return data_colours, ax
    
    selected_direction = window.direction_dropdown.currentText()
    if die_shape != []:
        if selected_direction == "X":
            data_colours, ax = get_displacement_data_colours(disp[0,:,:], selected_direction)
        if selected_direction == "Y":
            data_colours, ax = get_displacement_data_colours(disp[1,:,:], selected_direction)
        if selected_direction == "Z":
            data_colours, ax = get_displacement_data_colours(disp[2,:,:], selected_direction)
        if selected_direction == "Total":
            data_colours, ax = get_displacement_data_colours(np.sum(disp, axis=0), selected_direction)

        gw = GradientEditorItem(orientation="right")
        GradientMode = {'ticks': [(0, (0,0,255,255)), (0.5, (0,255,0,255)), (1, (255,0,0,255))], 'mode': 'rgb'}
        gw.restoreState(GradientMode)
        add_items(ax, gw, window)


def plot_thinning (window):
    global o3dmesh, die_shape, die_shape_zoom, disp, data_colours
    model_control.load_model(window.component_dropdown.currentText(), window.process_dropdown.currentText(), window.material_dropdown.currentText(), "Thinning (%)")
        
    input = np.zeros((5,y_resolution,x_resolution))
    # input channels 1 and 3 (die and punch shape)
    input[0,:y_edge,:x_edge] = input[2,:y_edge,:x_edge] = die_shape
    # input channels 2 and 4 (zoomed die and punch shape)
    input[1,:,:] = input[3,:,:] = die_shape_zoom
    # input channel 5 (blank shape)
    input[4,:,:] = blank_shape
    # predict using model
    pred = model_control.predict(input)
    thinning_field = pred[0,:] * 100
    # thinning -= thinning.min()

    thinning_field = np.transpose(thinning_field)
    thinning_min, thinning_max = np.amin(thinning_field), np.amax(thinning_field)
    o3dpoints = np.asarray(o3dmesh.vertices)
    x_max, x_min = np.amax(o3dpoints[:,0]), np.amin(o3dpoints[:,0])
    y_max, y_min = np.amax(o3dpoints[:,1]), np.amin(o3dpoints[:,1])
    field_x, field_y = thinning_field.shape[0], thinning_field.shape[1]
    data_colours = []
    # o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]
    for idx in range(o3dpoints.shape[0]):
        x = math.floor((o3dpoints[idx, 0] - x_min)/(x_max-x_min) * (field_x-1))
        y = math.floor((o3dpoints[idx, 1] - y_min)/(y_max-y_min) * (field_y-1))
        data_colours.append(thinning_field[x, y])
    data_colours = np.array(data_colours)
    data_colours -= np.min(data_colours)
    data_colours /= np.max(data_colours)
    ax = AxisItem("left")
    ax.setLabel(text="Thinning", units="%")
    ax.setRange(thinning_min, thinning_max)

    gw = GradientEditorItem(orientation="right")
    GradientMode = {'ticks': [(0, (0,0,255,255)), (0.5, (0,255,0,255)), (1, (255,0,0,255))], 'mode': 'rgb'}
    gw.restoreState(GradientMode)
    add_items(ax, gw, window)
    
    # np.save("temp/input.npy", input)
    # np.save("temp/thinning.npy", pred)

    # thinning_max = np.amax(thinning)
    # thinning_min = np.amin(thinning)
    # data_colours = []

    # o3dpoints = np.asarray(o3dmesh.vertices)
    # o3dpoints[:,0] -= np.min(o3dpoints[:,0])
    # o3dpoints[:,1] -= np.min(o3dpoints[:,1])
    # o3dpointx_max = [np.max(o3dpoints[:,0]), np.max(o3dpoints[:,1])]

    # for point in o3dpoints:
    #     x = math.floor(point[0]/o3dpointx_max[0] * (thinning.shape[0] - 1))
    #     y = math.floor(point[1]/o3dpointx_max[1] * (thinning.shape[1] - 1))
    #     # if x == 383:
    #     #     for i in range(8):
    #     #         data_colours.append(0)
    #     # elif y == 255:
    #     #     for i in range(20):
    #     #         data_colours.append(0)
    #     # else:
    #         # data_colours.append((thinning[x, y] - thinning_min)/(thinning_max - thinning_min))
    #     data_colours.append(thinning[x, y])
    # data_colours = np.array(data_colours)
    # data_colours -+ np.amin(data_colours)
    # ax = AxisItem("left")
    # ax.setLabel(text="Thinning (%)")
    # ax.setRange(np.amin(data_colours), np.amax(data_colours))
    