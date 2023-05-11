import numpy as np
import torch.nn as nn
import torch

class FC1(nn.Module):

    def __init__(self, z_dim=256, positional_encoding = False, fourier_degree = 5): #datashape is number of training shapes
        super(FC1, self).__init__()

        if positional_encoding is True:
            inputDimentions = z_dim + 2*fourier_degree*3
        else:
            inputDimentions = z_dim + 3
        
        self.decoder_stage1 = nn.Sequential(
            nn.Linear(inputDimentions, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512 - inputDimentions))
        
        self.decoder_stage2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Tanh())
        
        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree

    def fourier_transform(self, x, L=5):
        cosines = torch.cat([torch.cos((2**l)*3.1415*x) for l in range(L)], -1)
        sines = torch.cat([torch.sin((2**l)*3.1415*x) for l in range(L)], -1)
        transformed_x = torch.cat((cosines,sines),-1)
        return transformed_x
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints): #instead of giving it a ind, give it a latent vector

        #Transform xyz points (i.e., mapping from R into a higher dimentional space R^2L) before passing them into the network to:
        #1) Capture higher frequency details (such as sharp corners)
        #2) Simplifies learning of being translation invariant - according to Artem Lukoianov
        if self.positional_encoding:
            batchSDFxyzPoints = self.fourier_transform(batchSDFxyzPoints, self.fourier_degree)

        input1 = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        decoder_stage1_out = self.decoder_stage1(input1) #feed that contatonated latent vector into decoder stage 1
        input2 = torch.cat((decoder_stage1_out, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)
        decoder_stage2_out = self.decoder_stage2(input2)
        return decoder_stage2_out

class FC2(nn.Module):

    def __init__(self, z_dim=128, hidden_dim = 256, positional_encoding = False, fourier_degree = 5): #datashape is number of training shapes
        super(FC2, self).__init__()

        if positional_encoding is True:
            inputDimentions = z_dim + 2*fourier_degree*3
        else:
            inputDimentions = z_dim + 3
        
        self.decoder_stage1 = nn.Sequential(
            nn.Linear(inputDimentions, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim - inputDimentions))
        
        self.decoder_stage2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh())
        
        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree

    def fourier_transform(self, x, L=5):
        cosines = torch.cat([torch.cos((2**l)*3.1415*x) for l in range(L)], -1)
        sines = torch.cat([torch.sin((2**l)*3.1415*x) for l in range(L)], -1)
        transformed_x = torch.cat((cosines,sines),-1)
        return transformed_x
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints): #instead of giving it a ind, give it a latent vector

        #Transform xyz points (i.e., mapping from R into a higher dimentional space R^2L) before passing them into the network to:
        #1) Capture higher frequency details (such as sharp corners)
        #2) Simplifies learning of being translation invariant - according to Artem Lukoianov
        if self.positional_encoding:
            batchSDFxyzPoints = self.fourier_transform(batchSDFxyzPoints, self.fourier_degree)

        input1 = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        decoder_stage1_out = self.decoder_stage1(input1) #feed that contatonated latent vector into decoder stage 1
        input2 = torch.cat((decoder_stage1_out, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)
        decoder_stage2_out = self.decoder_stage2(input2)
        return decoder_stage2_out

class FC6(nn.Module):

    def __init__(self, z_dim=128, hidden_dim = 256, positional_encoding = False, fourier_degree = 5): #datashape is number of training shapes
        super(FC6, self).__init__()

        if positional_encoding is True:
            inputDimentions = z_dim + 2*fourier_degree*3
        else:
            inputDimentions = z_dim + 3
        
        self.fc1 = nn.Linear(inputDimentions, hidden_dim)
        self.ReLU = nn.ReLU(True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim - inputDimentions) #concatinative skip connection attaches here
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        
        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree

    def fourier_transform(self, x, L=5):
        cosines = torch.cat([torch.cos((2**l)*3.1415*x) for l in range(L)], -1)
        sines = torch.cat([torch.sin((2**l)*3.1415*x) for l in range(L)], -1)
        transformed_x = torch.cat((cosines,sines),-1)
        return transformed_x
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints): #instead of giving it a ind, give it a latent vector

        #Transform xyz points (i.e., mapping from R into a higher dimentional space R^2L) before passing them into the network to:
        #1) Capture higher frequency details (such as sharp corners)
        #2) Simplifies learning of being translation invariant - according to Artem Lukoianov
        if self.positional_encoding:
            batchSDFxyzPoints = self.fourier_transform(batchSDFxyzPoints, self.fourier_degree)

        input1 = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        output1 = self.ReLU(self.fc1(input1))
        output2 = self.ReLU(self.fc2(output1))
        output3 = self.ReLU(self.fc3(output2))
        output3 = output3 + output1 #residual connection
        output4 = self.ReLU(self.fc4(output3))
        input2 = torch.cat((output4, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output5 = self.ReLU(self.fc5(input2))
        output6 = self.ReLU(self.fc6(output5))
        output7 = self.ReLU(self.fc7(output6))
        output7 = output7 + output5 #residual connection
        output8 = self.Tanh(self.fc8(output7))
 
        return output8

class FC7(nn.Module):

    def __init__(self, z_dim=128, hidden_dim = 256, positional_encoding = False, fourier_degree = 5): #datashape is number of training shapes
        super(FC7, self).__init__()

        if positional_encoding is True:
            inputDimentions = z_dim + 2*fourier_degree*3
        else:
            inputDimentions = z_dim + 3
        
        self.fc1 = nn.Linear(inputDimentions, hidden_dim)
        self.ReLU = nn.ReLU(True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim - inputDimentions) #concatinative skip connection attaches here
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc9 = nn.Linear(hidden_dim, hidden_dim)
        self.fc10 = nn.Linear(hidden_dim, hidden_dim)
        self.fc11 = nn.Linear(hidden_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        
        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree

    def fourier_transform(self, x, L=5):
        cosines = torch.cat([torch.cos((2**l)*3.1415*x) for l in range(L)], -1)
        sines = torch.cat([torch.sin((2**l)*3.1415*x) for l in range(L)], -1)
        transformed_x = torch.cat((cosines,sines),-1)
        return transformed_x
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints): #instead of giving it a ind, give it a latent vector

        #Transform xyz points (i.e., mapping from R into a higher dimentional space R^2L) before passing them into the network to:
        #1) Capture higher frequency details (such as sharp corners)
        #2) Simplifies learning of being translation invariant - according to Artem Lukoianov
        if self.positional_encoding:
            batchSDFxyzPoints = self.fourier_transform(batchSDFxyzPoints, self.fourier_degree)

        input1 = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        output1 = self.ReLU(self.fc1(input1))
        output2 = self.ReLU(self.fc2(output1))
        output3 = self.ReLU(self.fc3(output2))
        output4 = self.ReLU(self.fc4(output3))
        input2 = torch.cat((output4, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output5 = self.ReLU(self.fc5(input2))
        output6 = self.ReLU(self.fc6(output5))
        output7 = self.ReLU(self.fc7(output6))
        output8 = self.ReLU(self.fc8(output7))
        input3 = torch.cat((output8, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)
        
        output9 = self.ReLU(self.fc9(input3))
        output10 = self.ReLU(self.fc10(output9))
        output11 = self.ReLU(self.fc11(output10))
        output12 = self.Tanh(self.fc12(output11))
 
        return output12

class FC8(nn.Module):

    def __init__(self, z_dim=128, hidden_dim = 256, positional_encoding = False, fourier_degree = 5): #datashape is number of training shapes
        super(FC8, self).__init__()

        if positional_encoding is True:
            inputDimentions = z_dim + 2*fourier_degree*3
        else:
            inputDimentions = z_dim + 3
        
        self.fc1 = nn.Linear(inputDimentions, hidden_dim)
        self.ReLU = nn.ReLU(True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim) #concatinative skip connection attaches here
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc10 = nn.Linear(hidden_dim, hidden_dim)
        self.fc11 = nn.Linear(hidden_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        
        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree

    def fourier_transform(self, x, L=5):
        cosines = torch.cat([torch.cos((2**l)*3.1415*x) for l in range(L)], -1)
        sines = torch.cat([torch.sin((2**l)*3.1415*x) for l in range(L)], -1)
        transformed_x = torch.cat((cosines,sines),-1)
        return transformed_x
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints): #instead of giving it a ind, give it a latent vector

        #Transform xyz points (i.e., mapping from R into a higher dimentional space R^2L) before passing them into the network to:
        #1) Capture higher frequency details (such as sharp corners)
        #2) Simplifies learning of being translation invariant - according to Artem Lukoianov
        if self.positional_encoding:
            batchSDFxyzPoints = self.fourier_transform(batchSDFxyzPoints, self.fourier_degree)

        input1 = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        output1 = self.ReLU(self.fc1(input1))
        output2 = self.ReLU(self.fc2(output1))
        output3 = self.ReLU(self.fc3(output2))
        input2 = torch.cat((output3, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output4 = self.ReLU(self.fc4(input2))
        output5 = self.ReLU(self.fc5(output4))
        output6 = self.ReLU(self.fc6(output5))
        input3 = torch.cat((output6, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output7 = self.ReLU(self.fc7(input3))
        output8 = self.ReLU(self.fc8(output7))
        output9 = self.ReLU(self.fc9(output8))
        input4 = torch.cat((output9, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output10 = self.ReLU(self.fc10(input4))
        output11 = self.ReLU(self.fc11(output10))
        output12 = self.Tanh(self.fc12(output11))
 
        return output12

class FC9(nn.Module):

    def __init__(self, z_dim=128, hidden_dim = 256, positional_encoding = False, fourier_degree = 5): #datashape is number of training shapes
        super(FC9, self).__init__()

        if positional_encoding is True:
            inputDimentions = z_dim + 2*fourier_degree*3
        else:
            inputDimentions = z_dim + 3
        
        self.fc1 = nn.Linear(inputDimentions, hidden_dim)
        self.ReLU = nn.ReLU(True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim - inputDimentions) #concatinative skip connection attaches here
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc9 = nn.Linear(hidden_dim, hidden_dim)
        self.fc10 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc11 = nn.Linear(hidden_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        
        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree

    def fourier_transform(self, x, L=5):
        cosines = torch.cat([torch.cos((2**l)*3.1415*x) for l in range(L)], -1)
        sines = torch.cat([torch.sin((2**l)*3.1415*x) for l in range(L)], -1)
        transformed_x = torch.cat((cosines,sines),-1)
        return transformed_x
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints): #instead of giving it a ind, give it a latent vector

        #Transform xyz points (i.e., mapping from R into a higher dimentional space R^2L) before passing them into the network to:
        #1) Capture higher frequency details (such as sharp corners)
        #2) Simplifies learning of being translation invariant - according to Artem Lukoianov
        if self.positional_encoding:
            batchSDFxyzPoints = self.fourier_transform(batchSDFxyzPoints, self.fourier_degree)

        input1 = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        output1 = self.ReLU(self.fc1(input1))
        output2 = self.ReLU(self.fc2(output1))
        input2 = torch.cat((output2, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output3 = self.ReLU(self.fc3(input2))
        output4 = self.ReLU(self.fc4(output3))
        input3 = torch.cat((output4, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output5 = self.ReLU(self.fc5(input3))
        output6 = self.ReLU(self.fc6(output5))
        input4 = torch.cat((output6, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output7 = self.ReLU(self.fc7(input4))
        output8 = self.ReLU(self.fc8(output7))
        input5 = torch.cat((output8, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output9 = self.ReLU(self.fc9(input5))
        output10 = self.ReLU(self.fc10(output9))
        input6 = torch.cat((output10, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output11 = self.ReLU(self.fc11(input6))
        output12 = self.Tanh(self.fc12(output11))
 
        return output12

class FC10(nn.Module):

    def __init__(self, z_dim=128, hidden_dim = 256, positional_encoding = False, fourier_degree = 5): #datashape is number of training shapes
        super(FC10, self).__init__()

        if positional_encoding is True:
            inputDimentions = z_dim + 2*fourier_degree*3
        else:
            inputDimentions = z_dim + 3
        
        self.fc1 = nn.Linear(inputDimentions, hidden_dim)
        self.ReLU = nn.ReLU(True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim) #concatinative skip connection attaches here
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        
        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree

    def fourier_transform(self, x, L=5):
        cosines = torch.cat([torch.cos((2**l)*3.1415*x) for l in range(L)], -1)
        sines = torch.cat([torch.sin((2**l)*3.1415*x) for l in range(L)], -1)
        transformed_x = torch.cat((cosines,sines),-1)
        return transformed_x
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints): #instead of giving it a ind, give it a latent vector

        #Transform xyz points (i.e., mapping from R into a higher dimentional space R^2L) before passing them into the network to:
        #1) Capture higher frequency details (such as sharp corners)
        #2) Simplifies learning of being translation invariant - according to Artem Lukoianov
        if self.positional_encoding:
            batchSDFxyzPoints = self.fourier_transform(batchSDFxyzPoints, self.fourier_degree)

        input1 = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        output1 = self.ReLU(self.fc1(input1))
        output2 = self.ReLU(self.fc2(output1))
        output3 = self.ReLU(self.fc3(output2))
        input2 = torch.cat((output3, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output4 = self.ReLU(self.fc4(input2))
        output5 = self.ReLU(self.fc5(output4))
        output6 = self.ReLU(self.fc6(output5))
        input3 = torch.cat((output6, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output7 = self.ReLU(self.fc7(input3))
        output8 = self.ReLU(self.fc8(output7))
 
        return output8

class FC11(nn.Module):

    def __init__(self, z_dim=128, hidden_dim = 256, positional_encoding = False, fourier_degree = 5): #datashape is number of training shapes
        super(FC11, self).__init__()

        if positional_encoding is True:
            inputDimentions = z_dim + 2*fourier_degree*3
        else:
            inputDimentions = z_dim + 3
        
        self.fc1 = nn.Linear(inputDimentions, hidden_dim)
        self.ReLU = nn.ReLU(True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim - inputDimentions) #concatinative skip connection attaches here
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim - inputDimentions)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        
        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree

    def fourier_transform(self, x, L=5):
        cosines = torch.cat([torch.cos((2**l)*3.1415*x) for l in range(L)], -1)
        sines = torch.cat([torch.sin((2**l)*3.1415*x) for l in range(L)], -1)
        transformed_x = torch.cat((cosines,sines),-1)
        return transformed_x
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints): #instead of giving it a ind, give it a latent vector

        #Transform xyz points (i.e., mapping from R into a higher dimentional space R^2L) before passing them into the network to:
        #1) Capture higher frequency details (such as sharp corners)
        #2) Simplifies learning of being translation invariant - according to Artem Lukoianov
        if self.positional_encoding:
            batchSDFxyzPoints = self.fourier_transform(batchSDFxyzPoints, self.fourier_degree)

        input1 = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        output1 = self.ReLU(self.fc1(input1))
        output2 = self.ReLU(self.fc2(output1))
        input2 = torch.cat((output2, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output3 = self.ReLU(self.fc3(input2))
        output4 = self.ReLU(self.fc4(output3))
        input3 = torch.cat((output4, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output5 = self.ReLU(self.fc5(input3))
        output6 = self.ReLU(self.fc6(output5))
        input4 = torch.cat((output6, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)

        output7 = self.ReLU(self.fc7(input4))
        output8 = self.ReLU(self.fc8(output7))
 
        return output8

#Model from Implicit Geometric Regularisation paper
class ImplicitNet(nn.Module):
    def __init__(self, z_dim, dims, skip_in=(), geometric_init=True, radius_init=1, beta=0):
        super().__init__()

        d_in = z_dim + 3
        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

        # self.activation_final = nn.Tanh()

    def forward(self, batchLatentVectors, batchSDFxyzPoints):

        input = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1)

        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x) #CHANGE TO SOFTPLUS - Gradients differentiable everywhere. Might help
            # else:
            #     x = self.activation_final(x) #REMOVE AND HAVE NO TANH AT THE END?

        return x


##########################
### MODEL: Auto-Decoder ORIGINAL CODE
##########################
class AutoDecoder(nn.Module):
    def __init__(self, z_dim=256): #datashape is number of training shapes
        super(AutoDecoder, self).__init__()
        
        self.decoder_stage1 = nn.Sequential(
            nn.Linear(z_dim+3, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 253))
        
        self.decoder_stage2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Tanh())
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints): #instead of giving it a ind, give it a latent vector

        input1 = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        decoder_stage1_out = self.decoder_stage1(input1) #feed that contatonated latent vector into decoder stage 1
        input2 = torch.cat((decoder_stage1_out, input1), dim=1) #concatonate the output of decoder stage 1 (vector of length 253) with the original input into decoder stage 1 (as done in the DeepSDF paper)
        decoder_stage2_out = self.decoder_stage2(input2)
        return decoder_stage2_out

########################################################################################################
### MODEL: SIREN - Implicit Neural Representations with Periodic Activation Functions
########################################################################################################
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, z_dim, hidden_features, hidden_layers, out_features, outermost_linear=False, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(z_dim+3, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,  np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, batchLatentVectors, batchSDFxyzPoints):

        input = torch.cat((batchLatentVectors, batchSDFxyzPoints), dim = 1) #contatonate each (repeated) latent vector with the 3D coordinates. Therefore, the latent vectors are the same (for each shape) but all have different concatonated 3D points
        output = self.net(input)
        
        return output        