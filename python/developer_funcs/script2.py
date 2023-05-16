import torch
import numpy as np
import matplotlib as mpl
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import os 
import re #for splitting strings
from scipy.interpolate import griddata
import meshio

from PyQt6.QtCore import QAbstractTableModel, Qt, QObject, QThread, pyqtSignal

from python.optimisation_funcs import surface_points_normals, autodecoder, single_prediction

### HELPER FUNCTIONS ###
#sort files in alphanumeric order
def sortFiles(dirName,extention): #this is really important to get the order of the files correct for the dataset
    numbers = []
    for file in os.listdir(dirName):
        if extention not in file: #only consider '.pc' files wihtin this folder
            continue
        
        baseName, sampleNo, frame, frame_number, fileExt = file.replace('.','_').split("_")
        numbers.append(int(sampleNo))

    numbers.sort()
    
    sortedNames = []
    for i,number in enumerate(numbers):
        currentName = baseName + "_" + str(number) + "_" + frame + "_" + frame_number + "." + fileExt
        sortedNames.append(currentName)
    return sortedNames

def sortedgeFiles(dirName,extention): #this is really important to get the order of the files correct for the dataset
    numbers = []
    for file in os.listdir(dirName):
        if extention not in file: #only consider '.pc' files wihtin this folder
            continue
        
        baseName, sampleNo, item, fileExt = file.replace('.','_').split("_")
        numbers.append(int(sampleNo))

    numbers.sort()
    tmp = numbers[1:].copy()
    tmp.append(numbers[0])
    
    tmp = [tmp[i] - numbers[i] for i in range(len(numbers))]
    
    print(np.where(np.array(tmp) > 1))
    
    sortedNames = []
    for i,number in enumerate(numbers):
        currentName = baseName + "_" + str(number) + "_" + item + "." + fileExt
        sortedNames.append(currentName)
    return sortedNames

#sort folders in alphanumeric order
def sortFolders(dirName): #this is really important to get the order of the frames correct for the dataset
    numbers = []
    for folder in os.listdir(dirName):
        
        if "Frame" not in folder: #only consider 'FrameX' folders
            continue
        
        baseName, number, ext = re.split('(\d+)',folder)
        numbers.append(int(number))

    numbers.sort()

    sortedNames = []
    for i,number in enumerate(numbers):
        currentName = baseName + str(number)
        sortedNames.append(currentName)
    return sortedNames

### Whether a point is in a polygon ###
def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[[], []], [[], []], [[], []], [[], []], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, cweld in enumerate(poly):

        x1, y1 = cweld[0]
        x2, y2 = cweld[1]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in

class worker(QObject):
    def __init__(self, component, input_dir, window):
        super().__init__()
    # def __init__ (self, num_iterations=100, *args, **kwargs):
        self.input_dir = input_dir
        self.component = component.lower()
        self.window = window
        self.cancelled = False

        window.stop.connect(self.stop_requested)

    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def stop_requested (self):
        print("stop requested")
        self.cancelled = True

    def run (self):
        #PATHS 
        #These are where the text files exist from Script 1
        simulationResultsImages_dir = 'temp/SimulationResultsImages/'
        elementCentre_dir = simulationResultsImages_dir + 'ElementCentres/'
        deformedFEMesh_dir = simulationResultsImages_dir + 'DeformedFENodes/'
        undeformedFEMesh_dir = simulationResultsImages_dir + 'UndeformedFENodes/'
        Thinning_dir = simulationResultsImages_dir + 'Thinning/'
        MajorStrain_dir = simulationResultsImages_dir + 'MajorStrain/'
        MinorStrain_dir = simulationResultsImages_dir + 'MinorStrain/'
        MisesStress_dir = simulationResultsImages_dir + 'MisesStress/'
        nodalDisplacements_dir = simulationResultsImages_dir + 'NodalDisplacements/'
        Edges_dir = self.input_dir + '/Edge/'
        export_dir = "temp/final_target_images/"

        print(os.path.exists(elementCentre_dir), os.path.exists(deformedFEMesh_dir), os.path.exists(Thinning_dir), os.path.exists(MajorStrain_dir), os.path.exists(MinorStrain_dir), os.path.exists(MisesStress_dir), os.path.exists(nodalDisplacements_dir))
        
        #fixed parameters
        imageResolution_H = 512
        imageResolution_W = 256
        baseLength_H = 160 #in mm
        baseLength_W = 80 #in mm
        # nFrames = 1
        nFields = 4 #thinning, major strain, minor strain, 
        nDisps = 3 #X disp, Y disp, Z disp

        #empty grid
        X = np.linspace(0,baseLength_H,imageResolution_H)
        Y = np.linspace(2,baseLength_W+2,imageResolution_W)
        gridX, gridY = np.meshgrid(X,Y)

        #get names of frame folders in a string vector sorted by number
        frameFolders = sortFolders(elementCentre_dir)

        #get names of the edge files in a string vector sorted by number
        edgeFiles = sortedgeFiles(Edges_dir,'Blank.nas')

        current_prog = 0

        #loop over all frames
        for i, frameFolder in enumerate(frameFolders):
            
            #================================================================================================================
            # PART 1: SORT OUT FILE NAMES OF .TXT FILES IN ORDER WITHIN EACH FRAME
            #================================================================================================================
            
            #----------------------------------------------------------------------------------------------------------------
            # 1.1: SORT ELEMENT CENTRE COORDINATES FILES
            #------------------------------------------------------renyan----------------------------------------------------------
            
            elementCentre_dir_Frame_i = elementCentre_dir + frameFolder 
            elementCentre_files_Frame_i = sortFiles(elementCentre_dir_Frame_i, ".txt")
            nSamples = len(elementCentre_files_Frame_i)

            #----------------------------------------------------------------------------------------------------------------
            # 1.2: SORT THINNING FILES
            #----------------------------------------------------------------------------------------------------------------
            
            Thinning_dir_Frame_i = Thinning_dir + frameFolder 
            Thinning_files_Frame_i = sortFiles(Thinning_dir_Frame_i, ".txt")
            
            #----------------------------------------------------------------------------------------------------------------
            # 1.3: SORT MAJOR STRAIN FILES
            #----------------------------------------------------------------------------------------------------------------
            
            MajorStrain_dir_Frame_i = MajorStrain_dir + frameFolder 
            MajorStrain_files_Frame_i = sortFiles(MajorStrain_dir_Frame_i, ".txt")
            
            #----------------------------------------------------------------------------------------------------------------
            # 1.4: SORT MINOR STRAIN FILES
            #----------------------------------------------------------------------------------------------------------------
            
            MinorStrain_dir_Frame_i = MinorStrain_dir + frameFolder 
            MinorStrain_files_Frame_i = sortFiles(MinorStrain_dir_Frame_i, ".txt")
            
            #----------------------------------------------------------------------------------------------------------------
            # 1.5: SORT NODAL DISPLACEMENT FILES
            #----------------------------------------------------------------------------------------------------------------
            
            nodalDisplacements_dir_Frame_i = nodalDisplacements_dir + frameFolder 
            nodalDisplacements_files_Frame_i = sortFiles(nodalDisplacements_dir_Frame_i, ".txt")
            
            #----------------------------------------------------------------------------------------------------------------
            # 1.6: SORT NODAL DISPLACEMENT FILES
            #----------------------------------------------------------------------------------------------------------------
            
            undeformedFENodes_dir_Frame_i = undeformedFEMesh_dir + frameFolder 
            undeformedFENodes_files_Frame_i = sortFiles(undeformedFENodes_dir_Frame_i, ".txt")
            
            #----------------------------------------------------------------------------------------------------------------
            # 1.7: INITIALISE IMAGES ARRAY FOR FRAME i
            #----------------------------------------------------------------------------------------------------------------
            
            targetsImage_Fields_Frame_i = np.zeros((nSamples, nFields, imageResolution_W, imageResolution_H))
            targetsImage_Displacements_Frame_i = np.zeros((nSamples, nDisps, imageResolution_W, imageResolution_H))

            #----------------------------------------------------------------------------------------------------------------
            # 1.8: SORT VON MISES STRESS FILES
            #----------------------------------------------------------------------------------------------------------------
            
            MisesStress_dir_Frame_i = MisesStress_dir + frameFolder 
            MisesStress_files_Frame_i = sortFiles(MisesStress_dir_Frame_i, ".txt")
            
            #================================================================================================================
            # PART 2: LOAD EACH SAMPLE POINT CLOUD, CLIP EXTREME VALUES AND INTERPOLATE TO IMAGE
            #================================================================================================================

            for j in range(len(elementCentre_files_Frame_i)): #iterate through all samples

                ######################################################################################
                # QT stuff
                num_iterations = len(frameFolders) * len(elementCentre_files_Frame_i)
                current_prog += 1
                self.progress.emit(100 * current_prog/num_iterations)
                if self.cancelled:
                    break
                ###################################################################################### 
                
                #----------------------------------------------------------------------------------------------------------------
                # 2.1: LOAD DATA FROM SAMPLE j IN FRAME i
                #----------------------------------------------------------------------------------------------------------------
                elementCentreFilePathInFrameFolder = elementCentre_dir_Frame_i + "/" + elementCentre_files_Frame_i[j]
                jth_elementCentre_Frame_i = np.loadtxt(elementCentreFilePathInFrameFolder,unpack=True).T[:,0:2]
            
                thinningFilePathInFrameFolder = Thinning_dir_Frame_i + "/" + Thinning_files_Frame_i[j]
                jth_Thinning_Frame_i = np.loadtxt(thinningFilePathInFrameFolder,unpack=True).T
                
                majorStrainFilePathInFrameFolder = MajorStrain_dir_Frame_i + "/" + MajorStrain_files_Frame_i[j]
                jth_MajorStrain_Frame_i = np.loadtxt(majorStrainFilePathInFrameFolder,unpack=True).T
                
                minorStrainFilePathInFrameFolder = MinorStrain_dir_Frame_i + "/" + MinorStrain_files_Frame_i[j]
                jth_MinorStrain_Frame_i = np.loadtxt(minorStrainFilePathInFrameFolder,unpack=True).T
                
                misesStressFilePathInFrameFolder = MisesStress_dir_Frame_i + "/" + MisesStress_files_Frame_i[j]
                jth_MisesStress_Frame_i = np.loadtxt(misesStressFilePathInFrameFolder,unpack=True).T
                
                displacementsFilePathInFrameFolder = nodalDisplacements_dir_Frame_i + "/" + nodalDisplacements_files_Frame_i[j]
                jth_Displacements_Frame_i = np.loadtxt(displacementsFilePathInFrameFolder,unpack=True).T
                
                undeformedFENodesFilePathInFrameFolder = undeformedFENodes_dir_Frame_i + "/" + undeformedFENodes_files_Frame_i[j]
                jth_undeformedFENodes_Frame_i = np.loadtxt(undeformedFENodesFilePathInFrameFolder,unpack=True).T[:,0:2]
                
                jth_XDisplacement_Frame_i = jth_Displacements_Frame_i[:,0]
                jth_YDisplacement_Frame_i = jth_Displacements_Frame_i[:,1]
                jth_ZDisplacement_Frame_i = jth_Displacements_Frame_i[:,2]
                
                #----------------------------------------------------------------------------------------------------------------
                # 2.2: CLIP EXTREME COMPRESSIVE STRAIN FIELD VALUES TO BE AT 99.5TH PERCENTILE OF ENTIRE DISTRIBUTION
                #----------------------------------------------------------------------------------------------------------------
                percentile_995_thinning = np.percentile(jth_Thinning_Frame_i,0.5) #0.5 is (1-99.5) since we are looking for less than
                idx = np.where(jth_Thinning_Frame_i < percentile_995_thinning)
                jth_Thinning_Frame_i[idx] = percentile_995_thinning
                
                percentile_995_minorStrain = np.percentile(jth_MinorStrain_Frame_i,0.5)
                idx = np.where(jth_MinorStrain_Frame_i < percentile_995_minorStrain)
                jth_MinorStrain_Frame_i[idx] = percentile_995_minorStrain
                
                #----------------------------------------------------------------------------------------------------------------
                # 2.3: INTERPOLATE DATA FROM SAMPLE j IN FRAME i TO IMAGE
                #----------------------------------------------------------------------------------------------------------------
                
                gridThinning = griddata(jth_elementCentre_Frame_i, jth_Thinning_Frame_i, (gridX, gridY), method='linear',fill_value=0)
                gridMajorStrain = griddata(jth_elementCentre_Frame_i, jth_MajorStrain_Frame_i, (gridX, gridY), method='linear',fill_value=0)
                gridMinorStrain = griddata(jth_elementCentre_Frame_i, jth_MinorStrain_Frame_i, (gridX, gridY), method='linear',fill_value=0)
                gridMisesStress = griddata(jth_elementCentre_Frame_i, jth_MisesStress_Frame_i, (gridX, gridY), method='linear',fill_value=0)
                gridDisplacementX = griddata(jth_undeformedFENodes_Frame_i, jth_XDisplacement_Frame_i, (gridX, gridY), method='linear',fill_value=0)
                gridDisplacementY = griddata(jth_undeformedFENodes_Frame_i, jth_YDisplacement_Frame_i, (gridX, gridY), method='linear',fill_value=0)
                gridDisplacementZ = griddata(jth_undeformedFENodes_Frame_i, jth_ZDisplacement_Frame_i, (gridX, gridY), method='linear',fill_value=0)
                
                #----------------------------------------------------------------------------------------------------------------
                # 2.4: Assign 0 to regions without materials
                #----------------------------------------------------------------------------------------------------------------
                
                meshFilePath = Edges_dir + edgeFiles[j]
                
                nodalIDs = []
                nodalCoordinates = []
                elementIDs = []
                elementConnectivity = []
                nodeConnectivity_elementIndicies = []
                
                with open(meshFilePath) as f: #open file j

                    # Iterate through lines
                    for line in f.readlines():

                        # Find the keyword
                        nodalIndex = line.find('GRID')
                        elementIndex = line.find('CROD')

                        # If the keyword is at the beginning of the line (index == 0)
                        if nodalIndex == 0:

                            nodalIDs.append(int(line[4:16]))
                            
                        if elementIndex == 0:
                            
                            temp = line.replace('\n', ' ') #replace \n with white space
                            temp = re.split(' +',line) #split the string at all occurances of white space                    
                            elementIDs.append(int(temp[1])) #store into element ID list
                            temp = re.split(' +',line) #split the string at all occurances of white space
                            temp = temp[2:] #only keep columns containing strings of element ID and connectivity
                            temp = np.int_(temp) #convert the strings into ints
                            elementConnectivity.append(temp) #store connectivity in a list
                            
                nodalCoordinates = meshio.read(meshFilePath).points
                            
                #convert into np arrays
                nodalIDs = np.array(nodalIDs)
                
                #elementConnectivity was populated with nodal IDs, here we populate elementConnectivity_nodalIndex with corresponding nodel index values
                #Note, this np.searchsorted() function works only if nodalIDs is sorted, which it will be every time, since the FE solver outputs node IDs in sorted order
                #If it is not sorted, please assign a valid value to the augment 'sort'
                poly = []
                for e in range(len(elementConnectivity)):
                    ID = np.searchsorted(nodalIDs,elementConnectivity[e])
                    poly.append([[nodalCoordinates[ID[0],0], nodalCoordinates[ID[0],1]], [nodalCoordinates[ID[1],0], nodalCoordinates[ID[1],1]]])
                    
                for p in range(len(X)):
                    for q in range(len(Y)):
                        if not is_in_poly([X[p], Y[q]], poly):
                            gridThinning[q, p] = 0
                            gridMajorStrain[q, p] = 0
                            gridMinorStrain[q, p] = 0
                            gridMisesStress[q, p] = 0
                            
                            gridDisplacementX[q, p] = 0
                            gridDisplacementY[q, p] = 0
                            gridDisplacementZ[q, p] = 0
                
                #assign images to correct channels in the targets images tensor
                #----------------------------------------------------------------------------------------------------------------
                # 2.5: ASSIGN IMAGES TO CORRECT CHANNEL IN THE targetsImageData_AllFrames TENSOR
                #----------------------------------------------------------------------------------------------------------------
                
                targetsImage_Fields_Frame_i[j, 0, :, :] = gridThinning
                targetsImage_Fields_Frame_i[j, 1, :, :] = gridMajorStrain
                targetsImage_Fields_Frame_i[j, 2, :, :] = gridMinorStrain
                targetsImage_Fields_Frame_i[j, 3, :, :] = gridMisesStress
                
                targetsImage_Displacements_Frame_i[j, 0, :, :] = gridDisplacementX
                targetsImage_Displacements_Frame_i[j, 1, :, :] = gridDisplacementY
                targetsImage_Displacements_Frame_i[j, 2, :, :] = gridDisplacementZ 
                
                print("--- SAMPLE", j, "IN FRAME", i,"DONE ---")
                
            #save each frame
            if not os.path.exists(export_dir):
                os.mkdir(export_dir)
            savedName_Strains = export_dir + "REDONE_SAMPLEStargetImages_Strains_Frame_" + "00" + str(i) + ".npy"
            savedName_Disps = export_dir + "REDONE_SAMPLEStargetImages_Displacements_Frame_" + "00" + str(i) + ".npy"

            np.save(file= savedName_Strains, arr=targetsImage_Fields_Frame_i)
            np.save(file= savedName_Disps, arr=targetsImage_Displacements_Frame_i)
            
            print("--- NEXT FRAME ---")

        print("script 2 done")
        ######################################################################################
        # QT stuff
        self.finished.emit()
        ######################################################################################    