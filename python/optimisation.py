
import threading
import time
import random
import os
import json
import pandas as pd
import numpy as np
import torch
from pyqtgraph.opengl import MeshData, GLScatterPlotItem, GLMeshItem
import open3d as o3d
from pyqtgraph import AxisItem, GradientEditorItem

from PyQt6.QtCore import QObject, pyqtSignal
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Property, Slot

from python import model_control
from python.optimisation_funcs import surface_points_normals, autodecoder, single_prediction


class load_mesh_worker(QObject):
    def __init__(self, file, window, selected_component):
        super().__init__()
        self.file = file
        self.window = window
        self.component = selected_component
        # self.cancelled = False
        # self.window.cancel.connect(self.stop)

    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def stop (self):
        self.cancelled = True

    def run (self):
        points, normals, offsurface_points = surface_points_normals.generate(self.file, self)
        best_latent_vector = autodecoder.get_latent_vector(points, normals, offsurface_points, self, self.component)
        print("best latent vector: ", best_latent_vector)
        

def load_mesh(file_path, window):
    file = file_path
    is_file, file_ext = os.path.isfile(file), file.rsplit('.', 1)[1].lower()

    if is_file and file_ext == 'stl':  # if the mesh files exist and they are .STL extension
        points, normals, offsurface_points = surface_points_normals.generate(file, window)
        best_latent_vector = autodecoder.get_latent_vector(points, normals, offsurface_points, window)
        print("best latent vector: ", best_latent_vector)
    else:
        print("error with file!")

# def begin (num_iterations, window):
#     optimisation_mainscript.begin(num_iterations, window)


def load_result(window):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    file_path = window.file
    latentVectorsForPlotting = torch.load(file_path, map_location=device)
    component = file_path.split("/")[-2]
    # variables_path = os.path.join(os.path.split(file_path)[0], "variables")

    # runningBHF = torch.load(os.path.join(variables_path, "BHF"), map_location=device)
    # runningFriction = torch.load(os.path.join(variables_path, "friction"), map_location=device)
    # runningClearance = torch.load(os.path.join(variables_path, "clearance"), map_location=device)
    # runningThickness = torch.load(os.path.join(variables_path, "thickness"), map_location=device)
    
    num_designs = len(latentVectorsForPlotting)
    
    window.num_designs = num_designs
    window.num_design_slider.setMaximum(num_designs)
    num = window.num_design_slider.value()

    window.num_design_label.setText(f"Showing design iteration {num} of {num_designs}")

    latentForOptimization = latentVectorsForPlotting[num - 1][-1]
    # window.bhf_value.setText(str(runningBHF[num -1]))
    # window.friction_value.setText(str(runningFriction[num -1]))
    # window.clearance_value.setText(str(runningClearance[num -1]))
    # window.thickness_value.setText(str(runningThickness[num -1]))
    # window.bhf_value.setText(str(round(runningBHF[-1], 2)) + " kN")
    # window.friction_value.setText(str(round(runningFriction[-1], 2)))
    # window.clearance_value.setText(str(round(runningClearance[-1], 2)) + " %")
    # window.thickness_value.setText(str(round(runningThickness[-1], 2)) + " mm")

    verts, faces = autodecoder.get_verts_faces(latentForOptimization, window, component)
    
    # meshdata = MeshData(vertexes=verts, faces=faces)
    # meshitem = GLMeshItem(meshdata=meshdata, drawFaces=True, drawEdges=False)
    # window.main_view.clear()
    # window.main_view.addItem(meshitem)

    if "bulkhead" in component.lower():
        single_prediction.bulkhead_thinning(verts, faces, window)
    if "u-bending" in component.lower():
        indicator = window.indicator_dropdown.currentText().lower()
        if "thinning" in indicator.lower():
            window.direction_label.setEnabled(False)
            window.direction_dropdown.setEnabled(False)
            single_prediction.ubending_thinning(verts, faces, window)
        if "displacement" in indicator.lower():
            window.direction_label.setEnabled(True)
            window.direction_dropdown.setEnabled(True)
            single_prediction.ubending_displacement(verts, faces, window)


    # window.GraphicsLayoutWidget.clear()
    # window.GraphicsLayoutWidget.addItem(ax)
    # window.GraphicsLayoutWidget.addItem(gw)

    # print("mesh added")




#-------------------------------------------------------------------------------------
# LEGACY
#-------------------------------------------------------------------------------------

# with open('info.json') as f:
#     info = json.load(f)

#     all_vars = []
#     for process in info["processes"]:
#         for input in process["inputs"]:
#             if input["name"] not in all_vars:
#                 all_vars.append(input["name"])
#         for output in process["outputs"]:
#             if output["name"] not in all_vars:
#                 all_vars.append(output["name"])

# # all_vars = ["Thinning","Springback","Strain","Force","Velocity","Blank Thickness","Temperature"]
# picked_vars = ["thinning input 1", "thinning input 2"]
# units_library = ["%", "%", "%", "kN", "mm/s", "mm", "°C"]
# units = []
# bounds = {}
# options = []
# progress = 0
# runsno = 0
# best_output = 0
# current_output = 0
# all_runs = {}
# bestrun = []
# stop_requested = False


# def load_varopts():
#     global all_vars

#     print("variable options for requested")
#     return all_vars


# def enable_vars(var_name, process_type):
#     with open('info.json') as f:
#         info = json.load(f)

#         for process in info["processes"]:
#             if process["name"] == process_type:
#                 for input in process["inputs"]:
#                     if input["name"] == var_name:
#                         return True
#                 for output in process["outputs"]:
#                     if output["name"] == var_name:
#                         return True
#         return False


# def pick_option(option):
#     global picked_vars

#     # option = str(option).lower()
#     picked_vars.append(option)
#     print(option + " picked")


# def unpick_option(option):
#     global picked_vars

#     # option = str(option).lower()
#     if option in picked_vars:
#         picked_vars.remove(option)
#         print(option + " unpicked")


# def set_options(material, process, model, goal, aim, setto, search, runsno):
#     global options

#     options = [goal, aim, setto, search, runsno, model]
#     print("options set: ", material, process,
#           model, goal, aim, setto, search, runsno)

#     model_control.selectMaterialandProcess(material, process)
#     model_control.load(model)


# def picked_optivars():
#     global picked_vars

#     return picked_vars


# def get_var_property(var, property):
#     global options

#     model_type = options[5]

#     with open('info.json') as f:
#         info = json.load(f)

#         for model in info["models"]:
#             if model["name"] == model_type:
#                 for input in model["inputs"]:
#                     if input["name"] == var:
#                         return input[property]

#                 for output in model["outputs"]:
#                     if output["name"] == var:
#                         return output[property]


# def change_name(idx, old, new):
#     global all_vars

#     all_vars[idx] = new

#     with open('info.json') as f:
#         info = json.load(f)

#     for model in info["models"]:
#         for input in model["inputs"]:
#             if input["name"] == old:
#                 input["name"] = new
#         for output in model["outputs"]:
#             if output["name"] == old:
#                 output["name"] = new

#     with open('info.json', "w+") as f:
#         f.write(json.dumps(info, indent=2))

#     print("name change from ", old, " to ", all_vars[idx])


# def set_bounds(var, lower, upper, unit):
#     global bounds

#     print(var, lower, upper, unit)
#     bounds[var] = [lower, upper, unit]


# def start(qml):
#     global bounds, options

#     threading.Thread(target=optimise, args=(bounds, options, qml)).start()


# def stop():
#     global stop_requested

#     stop_requested = True


# def optimise(bounds, options, qml):
#     global progress, runsno, picked_vars, bestrun, best_output, current_output, all_runs, stop_requested

#     print("optimisation started")
#     goal, aim, setto, search, runsno = options[0], options[1], options[2], options[3], options[4]

#     goal_column = goal+"_"+aim+"_"+str(setto)

#     for var in picked_vars:
#         all_runs[var] = list()

#     all_runs[goal_column] = list()

#     while progress < runsno:

#         if stop_requested:
#             all_runs_df = pd.DataFrame(all_runs)
#             all_runs_df.to_csv(os.path.join("temp", "optimisation_result.csv"))
#             print("optimisation stopped")
#             break

#         progress += 1
#         current_output = random.random()

#         for var in picked_vars:
#            all_runs[var].append(random.randint(bounds[var][0], bounds[var][1]))

#         #    var_list = all_runs[var]
#         #    var_list.append(random.randint(bounds[var][0],bounds[var][1]))
#         #    all_runs[var] = var_list

#         goal_list = all_runs[goal_column]
#         goal_list.append(current_output)
#         all_runs[goal_column] = goal_list

#         # print(all_runs)

#         if current_output > best_output:
#             bestrun = []
#             best_output = current_output

#             for var in all_runs:
#                 bestrun.append([var, all_runs[var][-1]])

#         qml.opti_result_updated.emit(progress, runsno, current_output)

#         time.sleep(2)

#     all_runs_df = pd.DataFrame(all_runs)
#     all_runs_df.to_csv(os.path.join("temp", "optimisation_result.csv"))


# def update_graph():
#     return random.randint(1, 20), random.randint(1, 20), random.random()


# def getbestrun():
#     global picked_vars, bestrun

#     return bestrun


# def getbestrun_final():
#     all_runs = pd.read_csv(os.path.join("temp", "optimisation_result.csv"))
#     aim = str(all_runs.columns[-1]).split('_')[-2]
#     setto = int(str(all_runs.columns[-1]).split('_')[-1])

#     if aim == "minimise":
#         idx = all_runs[all_runs.columns[-1]].idxmin()
#     if aim == "maximise":
#         idx = all_runs[all_runs.columns[-1]].idxmax()
#     if aim == "setto":
#         idx = all_runs.sub(setto).abs().idxmin()

#     best_run = []

#     for i, column in enumerate(all_runs.columns):

#         if i == 0:
#             best_run.append(["Run #", float(all_runs[column][idx])])

#         elif i == len(all_runs.columns)-1:
#             this_aim = str(column).split('_')[-2]

#             if this_aim != "setto":
#                 best_run.append(
#                     ["_".join(str(column).split('_')[:-1]), float(all_runs[column][idx])])

#         else:
#             best_run.append([column, float(all_runs[column][idx])])

#     return best_run


# def get_final_vars():
#     all_runs = pd.read_csv(os.path.join("temp", "optimisation_result.csv"))
#     vars = ["Run #"]

#     for i, column in enumerate(all_runs.columns[1:]):

#         if i == len(all_runs.columns)-1:
#             this_aim = str(column).split('_')[-2]

#             if this_aim != "setto":
#                 vars.append("_".join(str(column).split('_')[:-1]))

#         else:
#             vars.append(column)

#     return vars


# def get_final_graph():
#     all_runs = np.genfromtxt(os.path.join(
#         "temp", "optimisation_result.csv"), delimiter=',')

#     final = [[i, float(run[-1])] for i, run in enumerate(all_runs)]
#     return final


# class BestRunTable(QAbstractTableModel):
#     DtypeRole = Qt.UserRole + 1000
#     ValueRole = Qt.UserRole + 1001

#     def __init__(self, df=pd.DataFrame(), parent=None):
#         super(BestRunTable, self).__init__(parent)
#         df = df.rename(columns={"Unnamed: 0": "Run #"})
#         self._dataframe = df.round(decimals=2)

#     def setDataFrame(self, dataframe):
#         self.beginResetModel()
#         self._dataframe = dataframe.copy()
#         self.endResetModel()

#     def dataFrame(self):
#         return self._dataframe

#     dataFrame = Property(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

#     def rowCount(self, parent=QModelIndex()):
#         if parent.isValid():
#             return 0
#         return len(self._dataframe.index)

#     def columnCount(self, parent=QModelIndex()):
#         if parent.isValid():
#             return 0
#         return self._dataframe.columns.size

#     def data(self, index, role=Qt.DisplayRole):
#         if not index.isValid() or not (
#             0 <= index.row() < self.rowCount()
#             and 0 <= index.column() < self.columnCount()
#         ):
#             return
#         row = self._dataframe.index[index.row()]
#         col = self._dataframe.columns[index.column()]

#         if row == 0:
#             val = col
#         else:
#             val = self._dataframe.iloc[row-1][col]
#         if role == Qt.DisplayRole:
#             return str(val)

#     def sort(self, var):
#         return "sort by "+var
