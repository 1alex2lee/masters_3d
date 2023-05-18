import os
import torch.nn as nn
import matplotlib as mpl
import torch.utils.data as data
import matplotlib.pyplot as plt
#package import
import time
import numpy as np
import torch
import random
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
# from visdom import Visdom
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

from PyQt6.QtCore import QObject, pyqtSignal

from python.developer_funcs import manufacturingSurrogateModel

class worker(QObject):
    def __init__(self, component, process, material, indicator, epochs_num, batch_size, window):
        super().__init__()
    # def __init__ (self, num_iterations=100, *args, **kwargs):
        self.component = component
        self.process = process
        self.material = material
        self.indicator = indicator
        self.epochs_num = epochs_num
        self.batch_size = batch_size
        self.window = window
        self.cancelled = False

        window.stop.connect(self.stop_requested)

    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def stop_requested (self):
        print("stop requested")
        self.cancelled = True

    def run (self):
       # Hyper parameters
        random_seed = 1
        learning_rate = 0.0004
        batch_size = self.batch_size
        num_epochs= self.epochs_num
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # num_features = 2500
        ratio = 1
        num_channel = np.array([4,8,16,32,64,128,256,512])*ratio
        test_ratio = 0.85
        torch.set_num_threads(8)

        # load data (expensive operation)
        # path_node_die = 'dieImages01thin.npy'
        # path_node_BHF = 'BHFImages01thin.npy'
        # path_node_Friction = 'FrictionImages01thin.npy'
        # path_node_Clearance = 'ClearanceImages01thin.npy'
        # path_node_Thickness = 'ThicknessImages01thin.npy'
        path_node_inputs = 'temp/dieImagesZ8.npy'
        if self.indicator.lower() == "displacement":
            path_node_target = 'temp/final_target_images/REDONE_SAMPLEStargetImages_Displacements_Frame_000.npy'
        else:
            path_node_target = 'temp/final_target_images/REDONE_SAMPLEStargetImages_Strains_Frame_000.npy'
        # samplenum = 603

        # Inputs = np.zeros((samplenum,5,256,512))
        # Inputs[:,0,:] = np.load(path_node_die)[0:samplenum,:,:]
        # Inputs[:,1,:] = np.load(path_node_BHF)[0:samplenum,:,:]
        # Inputs[:,2,:] = np.load(path_node_Friction)[0:samplenum,:,:]
        # Inputs[:,3,:] = np.load(path_node_Clearance)[0:samplenum,:,:]
        # Inputs[:,4,:] = np.load(path_node_Thickness)[0:samplenum,:,:]
        # Inputs = np.delete(Inputs, [601,602,603,604,605,606,607,608], 0)
        # Inputs = np.load(path_node_inputs)[0:samplenum,:,:]
        # Targets = np.load(path_node_target)[0:samplenum,0,:] #Targets: thinning, major strain, minor strain, von mises stress
        Inputs = np.load(path_node_inputs)
        Targets = np.load(path_node_target) #Targets: thinning, major strain, minor strain, von mises stress

        # Data
        random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        np.random.seed(random_seed)
        index = [i for i in range(len(Targets))]
        random.shuffle(index)

        Inputs = Inputs[index]
        Targets = Targets[index]

        Inputs_train=Inputs[:int(len(Targets)*test_ratio),:,:]
        Inputs_test=Inputs[int(len(Targets)*test_ratio):,:,:]

        Targets_train=Targets[:int(len(Targets)*test_ratio),0,:,:] #only 1st channel kept (thinning image)
        Targets_test=Targets[int(len(Targets)*test_ratio):,0,:,:]

        class FormDataset(data.Dataset):
            def __init__(self,Inputs,Targets):
                self.Inputs=Inputs
                self.Targets=Targets

            def __getitem__(self,index):
                Input, Target=self.Inputs[index,:,:], self.Targets[index,:,:] # changed

                Input=torch.from_numpy(Input).float().to(device)
                Target=torch.from_numpy(Target).float().to(device)

                a = torch.zeros([1,256,512])
                a[:]=Input[:,:] # changed
                b = torch.zeros([1,256,512])
                b[0] = Target
                return a, b

            def __len__(self):
                return self.Inputs.shape[0]
            
        print('Inputs training dimensions:', Inputs_train.shape)
        print('Inputs testing dimensions:', Inputs_test.shape)
        print('Targets training dimensions:', Targets_train.shape)
        print('Targets testing dimensions:', Targets_test.shape)

        #Input and target is same
        Formdataset_train=FormDataset(Inputs_train,Targets_train)
        Formdataset_test=FormDataset(Inputs_test,Targets_test)

        train_loader=torch.utils.data.DataLoader(dataset=Formdataset_train,
                                                batch_size=batch_size,
                                                shuffle=True)

        test_loader=torch.utils.data.DataLoader(dataset=Formdataset_test,
                                            batch_size=batch_size,
                                            shuffle=False)
        
        torch.manual_seed(random_seed)
        model = manufacturingSurrogateModel.ResUNet(num_channel,batch_size)
        model = model.to(device)
        model.train()

        def my_loss(decoded, target):
            loss = (decoded * target).sum() / ((decoded * decoded).sum().sqrt() * (target * target).sum().sqrt())
            return loss.mean()

        loss = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # torchsummary.summary(model,(4,512,512))

        # #TEST IMAGE
        # randomImage = torch.rand((1,4,512,512))
        # model = ResUNet(num_channel,batch_size)
        # testOutput = model(randomImage)

        # print('Input image dimentions:', randomImage.shape)
        # print('Output image dimensions:', testOutput.shape)

        # vis = Visdom(env='Res-SE-U-Net-1_ref_256', port=8097)
        # assert vis.check_connection()
        # x, y = 0, 0
        # file = open('Epoches_of_Res-SE-U_SP_512_B4_2000_MSECOS0.2_LRFix0.0004_Seed1_E6B6D6_filtered_09Aug22.txt', 'w')
        # file2 = open('Loss_train_Res-SE-U_SP_512_B4_2000_MSECOS0.2_LRFix0.0004_Seed1_E6B6D6_filtered_09Aug22.txt', 'w')
        # file3 = open('Loss_test_Res-SE-U_SP_512_B4_2000_MSECOS0.2_LRFix0.0004_Seed1_E6B6D6_filtered_09Aug22.txt', 'w')

        # start_time = time.time()
        # cost_test_all = -0.19*np.ones(2010)
        # start_time = time.time()
        runningIterations = []
        runningCost = []
        self.window.canvas.axes.clear()

        for epoch in range(num_epochs):
            ######################################################################################
            # QT stuff
            self.progress.emit(100 * epoch/num_epochs)
            if self.cancelled:
                break
            ######################################################################################
            for batch_idx, (features, targets) in enumerate(train_loader):

                features = features.to(device) # don't need labels, only the images (features)
                ### FORWARD AND BACK PROP
                decoded = model(features)
                cost = loss(decoded, targets.to(device)) - 0.2 * my_loss(decoded, targets.to(device))
                optimizer.zero_grad()
                del features
                torch.cuda.empty_cache()
                del targets
                torch.cuda.empty_cache()
                del decoded
                torch.cuda.empty_cache()

                cost.backward()

                ### UPDATE MODEL PARAMETERS
                optimizer.step()

                ### LOGGING
                # if not batch_idx % 100: # record loss every 50 batches, or every epoch if the total batches less than 50
                #     file.write ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.8f'
                #         %(epoch+1, num_epochs, batch_idx,
                #             len(train_loader), cost))
                #     file.write('\r\n')
                #     file2.write ('%.8f' %(cost))
                #     file2.write('\r\n')

            runningIterations.append(epoch)
            runningCost.append(cost.detach().numpy())
            self.window.canvas.axes.plot(runningIterations, runningCost)
            self.window.canvas.axes.set_xlabel("Iterations")
            self.window.canvas.axes.set_ylabel("Loss")
            self.window.canvas.axes.set_title(f"Performance history for {self.indicator} model")
            self.window.canvas.draw()

            for batch_idx, (features_test, targets_test) in enumerate(test_loader):

                features_test = features_test.to(device) # don't need labels, only the images (features)
                ### FORWARD AND BACK PROP
                decoded_test = model(features_test)
                cost_test = loss(decoded_test, targets_test.to(device)) - 0.2 * my_loss(decoded_test, targets_test.to(device))
                #cost_test = my_loss(decoded_test, targets_test.to(device))

                ### LOGGING
                # if not batch_idx % 100:
                    # file.write ('Epoch: %03d/%03d | Batch %03d/%03d | Cost Test: %.8f'
                    #     %(epoch+1, num_epochs, batch_idx,
                    #         len(test_loader), cost_test))
                    # file.write('\r\n')
                    # file3.write ('%.8f' %(cost_test))
                    # file3.write('\r\n')
                    # cost_test_all[epoch] = cost_test
                    # if cost_test <= min(cost_test_all):
                    #     torch.save(model.state_dict(), 'Res-SE-U-Net_512_0.85train_B4_2000_COS0.2_LRFix0.0004_Seed1_E6B6D6_filtered_09Aug22_best.pkl')
                    #     print('Saved model: Epoch = %03d | Cost_test = %.8f' % (epoch+1, cost_test))

        #     file.write ('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        #     file.write('\r\n')

        # file.write ('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

        # file.close()
        # file2.close()
        # file3.close()

        model_save_dir = os.path.join("components", self.component, self.process, self.material)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(model.state_dict(), os.path.join(model_save_dir, self.indicator))
        print("training done")
        ######################################################################################
        # QT stuff
        self.finished.emit()
        ###################################################################################### 
