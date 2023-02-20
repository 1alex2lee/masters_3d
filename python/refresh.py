
from PIL import Image, ImageDraw
import numpy as np
from PySide6.QtCore import QAbstractTableModel
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
import os
from python import model_control


def manufacturbility (force, velocity, blankthickness, temperature):
    return (force + velocity + blankthickness + temperature)/4


def field (value, parameter):

    # # Create a blank image with a white background
    # img = Image.new('RGB', (256, 384), (255, 255, 255))

    # # Get the drawing context
    # draw = ImageDraw.Draw(img)

    # # Draw different patterns based on the variables
    # # You can use the variables to control the size, color, and position of the patterns

    # # Draw the first pattern
    # draw.rectangle([(var1*400, var2*400), (var3*400, var4*400)], fill=(255, 0, 0))

    # # Draw the second pattern
    # draw.ellipse([(var2*400, var3*400), (var4*400, var1*400)], fill=(0, 255, 0))

    # # Draw the third pattern
    # draw.polygon([(var1*400, var2*400), (var3*400, var4*400), (var4*400, var1*400)], fill=(0, 0, 255))

    # # Draw the fourth pattern
    # draw.line([(var1*400, var2*400), (var3*400, var4*400)], fill=(255, 255, 0), width=5)

    # img = np.array(img)
    # img = np.moveaxis(img, 2, 0)

    img = np.ones((5,256,384))
    # img[0,:,:] = var1*img[0,:,:]
    # img[1,:,:] = var2*img[1,:,:]
    # img[2,:,:] = var3*img[2,:,:]
    # img[3,:,:] = var4*img[3,:,:]
    # img[4,:,:] = np.mean([var1,var2,var3,var4])*img[4,:,:]

    img[0,:,:] = np.load(os.path.join(os.getcwd(),"temp","input_1.npy"))
    img[1,:,:] = np.load(os.path.join(os.getcwd(),"temp","input_2.npy"))
    img[2,:,:] = np.load(os.path.join(os.getcwd(),"temp","input_1.npy"))
    img[3,:,:] = np.load(os.path.join(os.getcwd(),"temp","input_2.npy"))
    img[4,:,:] = value * np.ones((256,384))

    # m = model
    # m = model.load("Thinning")
    pred = model_control.predict(img)
    # pred = np.array(pred)
    # max = np.amax(pred)
    # min = np.amin(pred)
    # pred = np.around(255*(pred-min)/(max-min),0)

    # output = np.zeros((3,256,384))
    # output[0,:,:] = np.around(pred[0,:,:],0)
    # output[1,:,:] = np.around(pred[1,:,:],0)

    # output = np.moveaxis(output, 0, 2)

    # # print(output[1,1,:])

    # # output_img = Image.new('RGB', (256,384), (255, 255, 255))
    # # output_img.putdata(output)

    # output_img = Image.fromarray(output.astype(np.uint8))

    # # Save the image
    # output_img.save('../temp/outputfield1.png')
    # output_img.save('../temp/outputfield2.png')

    plt.imsave(os.path.join(os.getcwd(),"temp","outputfield1.png"),pred[0,:,:])
    plt.imsave(os.path.join(os.getcwd(),"temp","outputfield2.png"),pred[0,:,:])


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._data[0])


def bestruntable ():

    columns = ['Run #','Force','Velocity','Manufacturbility']
#    data = np.random.rand(3,4)
    data = [
              [4, 9, 2],
              [1, 0, 0],
              [3, 5, 0],
              [3, 3, 2],
              [7, 8, 9],
            ]

    table = TableModel(data)

    return table
