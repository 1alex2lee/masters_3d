import os
import numpy as np
import re #for splitting strings

from PyQt6.QtCore import QAbstractTableModel, Qt, QObject, QThread, pyqtSignal

from python.optimisation_funcs import surface_points_normals, autodecoder, single_prediction

### HELPER FUNCTIONS ###

#sort files in alphanumeric order
def sortFiles(dirName,extention):
    numbers = []
    for file in os.listdir(dirName):
        if extention not in file:
            continue
        baseName, number, fileExt = file.replace('.','_').split('_')
        numbers.append(int(number))

    numbers.sort()

    sortedNames = []
    for i,number in enumerate(numbers):
        currentName = baseName + "_" + str(number) + "." + fileExt
        sortedNames.append(currentName)
    return sortedNames

#sort folders in alphanumeric order
def sortFolders(dirName):
    numbers = []
    for folder in os.listdir(dirName):
        
        if "Frame" not in folder:
            continue
        
        baseName, number, ext = re.split('(\d+)',folder)
        numbers.append(int(number))

    numbers.sort()

    sortedNames = []
    for i,number in enumerate(numbers):
        currentName = baseName + str(number)
        sortedNames.append(currentName)
    return sortedNames

class worker(QObject):
    def __init__(self, component, target_dir, window):
        super().__init__()
    # def __init__ (self, num_iterations=100, *args, **kwargs):
        self.target_dir = target_dir
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
    # def get_result_images (component, input_dir):
        target_dir = self.target_dir
        #input directories
        extractedResults_dir = target_dir + "/"
        deformedMesh_dir = extractedResults_dir + 'DeformedMeshes/DeformedMeshes4/'
        thinningResults_dir = extractedResults_dir + 'Thinning/Thinning4/'
        majorStrainResults_dir = extractedResults_dir + 'MajorStrain/MajorStrain4/'
        minorStrainResults_dir = extractedResults_dir + 'MinorStrain/MinorStrain4/'
        misesStressResults_dir = extractedResults_dir + 'MisesStress/MisesStress4/'
        displacementsResults_dir = extractedResults_dir + 'Displacements/'

        print(os.path.exists(deformedMesh_dir), os.path.exists(thinningResults_dir), os.path.exists(majorStrainResults_dir), os.path.exists(minorStrainResults_dir), os.path.exists(displacementsResults_dir))

        #output directories
        simulationResultsImages_dir = 'temp/SimulationResultsImages/'
        undeformedFEMesh_dir = simulationResultsImages_dir + 'UndeformedFENodes/'
        deformedFEMesh_dir = simulationResultsImages_dir + 'DeformedFENodes/'
        ElementCentre_dir = simulationResultsImages_dir + 'ElementCentres/'
        Thinning_dir = simulationResultsImages_dir + 'Thinning/'
        MajorStrain_dir = simulationResultsImages_dir + 'MajorStrain/'
        MinorStrain_dir = simulationResultsImages_dir + 'MinorStrain/'
        MisesStress_dir = simulationResultsImages_dir + 'MisesStress/'
        nodalDisplacements_dir = simulationResultsImages_dir + 'NodalDisplacements/'

        folders = [undeformedFEMesh_dir, deformedFEMesh_dir, ElementCentre_dir, Thinning_dir, MajorStrain_dir, MinorStrain_dir, MisesStress_dir, nodalDisplacements_dir]

        #create the output folders if they do not exist
        nFrames = 1
        for i, folder in enumerate(folders):
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            for j in range(nFrames):
                frameFolderName = folder + "Frame_" + str(j+1) + "/"
                if not os.path.exists(frameFolderName):
                    os.makedirs(frameFolderName)

        #================================================================================================================
        # PART 1: EXTRACT DATA FROM DEFORMED MESHES
        #================================================================================================================
        frameFolder = "Frame_1"

        #sorting the file names in alphanumeric order. i.e. file1, file100, file 10, file 2... turns into file1, file2, file 10, file 100 etc
        filesInFrameFolder_1 = sortFiles(deformedMesh_dir,'.pc') #consider only '.pc' files. These are mesh files exported by PAM
        filesInFrameFolder_2 = sortFiles(displacementsResults_dir + 'Disp4/','.asc')

        allNodalCoordinates_frame_i = []

        joffset = 0

        num_iterations = len(filesInFrameFolder_1) * 4
        current_prog = 0

        #the loop for all files (samples) in frame_i
        for j, (fileInFrameFolder_1, fileInFrameFolder_2) in enumerate(zip(filesInFrameFolder_1, filesInFrameFolder_2)): #for each mesh file within the frame folder

            ######################################################################################
            # QT stuff
            current_prog += 1
            self.progress.emit(100 * current_prog/num_iterations)
            if self.cancelled:
                break
            ######################################################################################

            meshFilePath = deformedMesh_dir + fileInFrameFolder_1

            nodalIDs = []
            nodalCoordinates = []
            elementIDs = []
            elementConnectivity = []
            nodeConnectivity_elementIndicies = []

            with open(meshFilePath) as f: #open file j

                # Iterate through lines
                for line in f.readlines():

                    # Find the keyword
                    nodalIndex = line.find('NODE')
                    elementIndex = line.find('SHELL')

                    # If the keyword is at the beginning of the line (index == 0)
                    if nodalIndex == 0:
                        temp = re.split(' +',line) #split the string at all occurances of white space
                        temp = temp[2:-1] #only keep 4 columns containing strings of node ID and node x y z position
                        temp = np.float_(temp) #convert the strings into floats
                        nodalIDs.append(int(temp[0]))
                        nodalCoordinates.append(temp[1:])

                    if elementIndex == 0:
                        temp = line.replace('\n', ' ') #replace \n with white space
                        temp = re.split(' +',temp) #split the string at all occurances of white space
                        temp = temp[2:-1] #only keep columns containing strings of element ID and connectivity
                        temp = np.float_(temp) #convert the strings into floats
                        elementIDs.append(int(temp[0])) #store into element ID list
                        elementConnectivity.append(temp[2:]) #store connectivity in a list

            #convert into np arrays
            nodalIDs = np.array(nodalIDs)
            nodalCoordinates = np.array(nodalCoordinates)
            
            #save to *.txt files
            deformedNodalCoordinatesName = deformedFEMesh_dir + frameFolder + "/" + "deformedNodalCoordinates_" + str(j+1+joffset) + "_" + frameFolder + ".txt"
            np.savetxt(deformedNodalCoordinatesName, nodalCoordinates, delimiter=' ')
            
            print('-> Connectivities and deformed nodal coordinates extracted:', fileInFrameFolder_1, frameFolder)

            TotalDisp = []
            
            for k in range(2,4):
                
            # nodalDisplacementsFilePath = displacementsResults_dir + frameFolder + "/" + fileInFrameFolder
                nodalDisplacementsFilePath = displacementsResults_dir + "Disp" +str(k+1) + "/" + fileInFrameFolder_2
            
                #override these every new file in frame i
                nodalDisplacementData = []

                # open the displacements file and extract data
                with open(nodalDisplacementsFilePath) as f:

                    # Iterate through lines and grab the thinning data
                    for line in f.readlines()[9:-1]: #skip the first 9 lines and the last one, these are not needed

                        temp = line.replace('\n', ' ')
                        temp = re.split(' +',temp) #split the string at all occurances of white space
                        temp = temp[1:-1] #only keep 4 columns containing strings of node ID and node x y z position
                        temp = np.float_(temp) #convert the strings into floats
                        nodalDisplacementData.append(temp[1:]) #these displacements are at nodes already
                        
                TotalDisp.append(np.array(nodalDisplacementData))
                
            TotalDisplacement = sum(TotalDisp)

            #save to text file
            nodalDisplacementsName = nodalDisplacements_dir + frameFolder + "/" + "nodalDisplacements_" + str(j+1+joffset) + "_" + frameFolder + ".txt"
            np.savetxt(nodalDisplacementsName, TotalDisplacement, delimiter=' ')

            #compute undeformed mesh
            undeformedNodes = np.array(nodalCoordinates) - np.array(TotalDisplacement)

            #save to text file
            undeformedName = undeformedFEMesh_dir + frameFolder + "/" + "undeformedNodes_" + str(j+1+joffset) + "_" + frameFolder + ".txt"
            np.savetxt(undeformedName, undeformedNodes, delimiter=' ')

            print('-> Displacements and undeformed nodes extracted:', fileInFrameFolder_2, frameFolder)
            
            #elementConnectivity was populated with nodal IDs, here we populate elementConnectivity_nodalIndex with corresponding nodel index values
            #Note, this np.searchsorted() function works only if nodalIDs is sorted, which it will be every time, since the FE solver outputs node IDs in sorted order
            #If it is not sorted, please assign a valid value to the augment 'sort'
            elementConnectivity_nodalIndex = []
            for e in range(len(elementConnectivity)):
                elementConnectivity_nodalIndex.append(np.searchsorted(nodalIDs,elementConnectivity[e]))
                
            #calculate the coordinates of the central point of each element
            element_centre = []
            for e in range(len(elementConnectivity_nodalIndex)):
                corner = undeformedNodes[elementConnectivity_nodalIndex[e]]
                centre = corner.mean(axis=0)
                element_centre.append(centre)
            element_centre = np.array(element_centre)
            
            #save to text file
            ElementCentreName = ElementCentre_dir + frameFolder + "/" + "elementcentre_" + str(j+1+joffset) + "_" + frameFolder + ".txt"
            np.savetxt(ElementCentreName, element_centre, delimiter=' ')
            
            print('-> Element centres extracted:', fileInFrameFolder_2, frameFolder)

        #================================================================================================================
        # PART 2: EXTRACT DATA FROM THINNING FIELDS
        #================================================================================================================

        #sorting the file names in alphanumeric order. i.e. file1, file100, file 10, file 2... turns into file1, file2, file 10, file 100 etc
        filesInFrameFolder = sortFiles(thinningResults_dir,'.asc') #consider only '.pc' files

        for j, fileInFrameFolder in enumerate(filesInFrameFolder): #for each mesh file within the frame folder

            ######################################################################################
            # QT stuff
            current_prog += 1
            self.progress.emit(100 * current_prog/num_iterations)
            if self.cancelled:
                break
            ######################################################################################

            thinningFilePath = thinningResults_dir + "/" + fileInFrameFolder


            #override these every new file in frame i
            elementThinningIDs = []
            elementThinningData = []

            #open the thinning file and extract data
            with open(thinningFilePath) as f:

                # Iterate through lines and grab the thinning data
                for line in f.readlines()[9:-1]: #skip the first 9 lines and the last one, these are not needed

                    temp = line.replace('\n', ' ')
                    temp = re.split(' +',temp) #split the string at all occurances of white space
                    temp = temp[1:-1] #only keep 4 columns containing strings of node ID and node x y z position
                    temp = np.float_(temp) #convert the strings into floats
                    elementThinningIDs.append(int(temp[0]))
                    elementThinningData.append(temp[1]) 

            #save to text file
            ThinningName = Thinning_dir + frameFolder + "/" + "Thinning_" + str(j+1+joffset) + "_" + frameFolder + ".txt"
            np.savetxt(ThinningName, elementThinningData, delimiter=' ')

            print('-> Thinning Data extracted:', fileInFrameFolder, frameFolder)

        #================================================================================================================
        # PART 3: EXTRACT DATA FROM MAJOR STRAIN FIELDS AND AVERAGE TO THE CONNECTED NODES
        #================================================================================================================

        #sorting the file names in alphanumeric order. i.e. file1, file100, file 10, file 2... turns into file1, file2, file 10, file 100 etc
        # frameFolder_path = majorStrainResults_dir + frameFolder
        filesInFrameFolder = sortFiles(majorStrainResults_dir,'.asc') #consider only '.pc' files

        for j, fileInFrameFolder in enumerate(filesInFrameFolder): #for each mesh file within the frame folder

            ######################################################################################
            # QT stuff
            current_prog += 1
            self.progress.emit(100 * current_prog/num_iterations)
            if self.cancelled:
                break
            ######################################################################################
            
            # !!! # this is the number of a single sample which was redone due to some discovered numerical error, and now it needs its point cloud data to be extracted
        #     if fileInFrameFolder != "MajorStrain_252.asc": 
        #         continue

        #     majorStrainFilePath = majorStrainResults_dir + "/" + frameFolder + "/" + fileInFrameFolder
            majorStrainFilePath = majorStrainResults_dir + "/" + fileInFrameFolder

            #override these every new file in frame i
            elementMajorStrainIDs = []
            elementMajorStrainData = []

            #open the file and extract data
            with open(majorStrainFilePath) as f:

                # Iterate through lines and grab the major strain data
                for line in f.readlines()[9:-1]: #skip the first 9 lines and the last one, these are not needed

                    temp = line.replace('\n', ' ')
                    temp = re.split(' +',temp) #split the string at all occurances of white space
                    temp = temp[1:-1] #only keep 4 columns containing strings of node ID and node x y z position
                    temp = np.float_(temp) #convert the strings into floats
                    elementMajorStrainIDs.append(int(temp[0]))
                    elementMajorStrainData.append(temp[1]) 
        
            #save to text file
            MajorStrainName = MajorStrain_dir + frameFolder + "/" + "Major_" + str(j+1+joffset) + "_" + frameFolder + ".txt"
            np.savetxt(MajorStrainName, elementMajorStrainData, delimiter=' ')
            print('-> Major Strain Data:', fileInFrameFolder, frameFolder) 

        #================================================================================================================
        # PART 4: EXTRACT DATA FROM MINOR STRAIN FIELDS AND AVERAGE TO THE CONNECTED NODES
        #================================================================================================================

        #sorting the file names in alphanumeric order. i.e. file1, file100, file 10, file 2... turns into file1, file2, file 10, file 100 etc
        # frameFolder_path = minorStrainResults_dir + frameFolder
        filesInFrameFolder = sortFiles(minorStrainResults_dir,'.asc') #consider only '.pc' files

        for j, fileInFrameFolder in enumerate(filesInFrameFolder): #for each mesh file within the frame folder

            ######################################################################################
            # QT stuff
            current_prog += 1
            self.progress.emit(100 * current_prog/num_iterations)
            if self.cancelled:
                break
            ######################################################################################
            
            # !!! # this is the number of a single sample which was redone due to some discovered numerical error, and now it needs its point cloud data to be extracted
        #     if fileInFrameFolder != "MajorStrain_252.asc": 
        #         continue

        #     minorStrainFilePath = minorStrainResults_dir + "/" + frameFolder + "/" + fileInFrameFolder
            minorStrainFilePath = minorStrainResults_dir + "/" + fileInFrameFolder

            #override these every new file in frame i
            elementMinorStrainIDs = []
            elementMinorStrainData = []

            #open the file and extract data
            with open(minorStrainFilePath) as f:

                # Iterate through lines and grab the minor strain data
                for line in f.readlines()[9:-1]: #skip the first 9 lines and the last one, these are not needed

                    temp = line.replace('\n', ' ')
                    temp = re.split(' +',temp) #split the string at all occurances of white space
                    temp = temp[1:-1] #only keep 4 columns containing strings of node ID and node x y z position
                    temp = np.float_(temp) #convert the strings into floats
                    elementMinorStrainIDs.append(int(temp[0]))
                    elementMinorStrainData.append(temp[1]) 

            #save to text file
            MinorStrainName = MinorStrain_dir + frameFolder + "/" + "Minor_" + str(j+1+joffset) + "_" + frameFolder + ".txt"
            np.savetxt(MinorStrainName, elementMinorStrainData, delimiter=' ')
            print('-> Minor Strain Data:', fileInFrameFolder, frameFolder) 
            
        #================================================================================================================
        # PART 5: EXTRACT DATA FROM VON MISES STRESS
        #================================================================================================================

        #sorting the file names in alphanumeric order. i.e. file1, file100, file 10, file 2... turns into file1, file2, file 10, file 100 etc
        # frameFolder_path = minorStrainResults_dir + frameFolder
        filesInFrameFolder = sortFiles(misesStressResults_dir,'.asc') #consider only '.pc' files

        for j, fileInFrameFolder in enumerate(filesInFrameFolder): #for each mesh file within the frame folder

            ######################################################################################
            # QT stuff
            current_prog += 1
            self.progress.emit(100 * current_prog/num_iterations)
            if self.cancelled:
                break
            ######################################################################################
            
            # !!! # this is the number of a single sample which was redone due to some discovered numerical error, and now it needs its point cloud data to be extracted
        #     if fileInFrameFolder != "MajorStrain_252.asc": 
        #         continue

        #     minorStrainFilePath = minorStrainResults_dir + "/" + frameFolder + "/" + fileInFrameFolder
            misesStressFilePath = misesStressResults_dir + "/" + fileInFrameFolder

            #override these every new file in frame i
            elementMisesStressIDs = []
            elementMisesStressData = []

            #open the file and extract data
            with open(misesStressFilePath) as f:

                # Iterate through lines and grab the minor strain data
                for line in f.readlines()[9:-1]: #skip the first 9 lines and the last one, these are not needed

                    temp = line.replace('\n', ' ')
                    temp = re.split(' +',temp) #split the string at all occurances of white space
                    temp = temp[1:-1] #only keep 4 columns containing strings of node ID and node x y z position
                    temp = np.float_(temp) #convert the strings into floats
                    elementMisesStressIDs.append(int(temp[0]))
                    elementMisesStressData.append(temp[1]) 

            #save to text file
            MisesStressName = MisesStress_dir + frameFolder + "/" + "Mises_" + str(j+1+joffset) + "_" + frameFolder + ".txt"
            np.savetxt(MisesStressName, elementMisesStressData, delimiter=' ')
            print('-> von Mises Stress Data:', fileInFrameFolder, frameFolder) 


        print('Done')    

        ######################################################################################
        # QT stuff
        self.finished.emit()
        ######################################################################################    

        # for file in [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f[0] != "."]:
        #     points, normals, offsurface_points = surface_points_normals.generate(file)
        #     best_latent_vector = autodecoder.get_latent_vector(points, normals, offsurface_points, None, component)
        #     verts, faces = autodecoder.get_verts_faces(best_latent_vector, None, component)

            # if component == "bulkhead":
                
            # if component == "u-bending":
