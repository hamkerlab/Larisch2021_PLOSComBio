#----------------------imports and environment---------------------------------
##Neuron - Modell for the analysis##
from ANNarchy import *
setup(dt=1.0,seed=1356)
import numpy as np
import scipy.io as sio
import os

from net import *
from matplotlib import gridspec
import matplotlib.pyplot as plt

# Show only a few number of natural scene patches
# R. Larisch, Technische Universitaet Chemnitz
# 2020-04-27

def getInput(images,patchsize):
    pictNbr = np.random.randint(0,10)
    length,width = images.shape[0:2]
    xPos = np.random.randint(0,length-patchsize)
    yPos = np.random.randint(0,width-patchsize)
    inputPatch = images[xPos:xPos+patchsize,yPos:yPos+patchsize,:,pictNbr]
    maxVal = np.max(images[:,:,:,pictNbr])
    return(inputPatch,maxVal)
#-------------------------------------------------------------------------------
def preprocessData(matData):
    # function to split the prewhitened images into on and off counterparts
    images = matData['IMAGES']
    w,h,n_images = np.shape(images)
    new_images = np.zeros((w,h,2,n_images))
    for i in range(n_images):
        new_images[images[:,:,i] > 0, 0, i] = images[images[:,:,i] > 0, i]
        new_images[images[:,:,i] < 0, 1, i] = images[images[:,:,i] < 0, i]*-1

    return(new_images)
#-------------------------------------------------------------------------------
def setActivInput(images, maxInput):
    inputPatch,maxVal = getInput(images,patchsize)
    if np.random.rand() <0.5:
        inputPatch=np.fliplr(inputPatch)
    if np.random.rand()<0.5:
        inputPatch=np.flipud(inputPatch)
    popInput.rates = inputPatch/maxVal*maxInput
    return(inputPatch)
#------------------------------------------------------------------------------
def startGetData(duration = 125.0,maxInput=125):
    print('start to generate some data')

    if not os.path.exists('./Output/V1Layer'):
        os.mkdir('./Output/V1Layer')

    nbrOfPatchesActiv = 5

    matData = sio.loadmat('./Input_Data/IMAGES.mat')
    images = preprocessData(matData)

    compile()
    loadWeights()

      #------init neuron recording-------#
    V1Mon = Monitor(popV1,['spike'])
    InhibMon=Monitor(popInhibit,['spike'])

    #-----------stop Time and simulate------#
    print('start Simulation for Activity')
    for i in range(nbrOfPatchesActiv):
        setActivInput(images,maxInput)
        simulate(duration)         

    #------get recording data---------#

    rec_frExc = V1Mon.get('spike')
    rec_frInh = InhibMon.get('spike')
    #----------------plot output---------------#
    onSetExc = np.zeros(int(nbrOfPatchesActiv*duration))
    onSetInh = np.zeros(int(nbrOfPatchesActiv*duration))


    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 20)
    plt.rc('ytick',labelsize = 20)

    for i in range(nbrOfPatchesActiv):
        onSetExc[int(i*duration)] = 150
        onSetInh[int(i*duration)] = 40
    spike_times,ranks = V1Mon.raster_plot(rec_frExc)
    plt.figure(figsize=(10,5))
    plt.plot(onSetExc,'r',lw=3)
    plt.plot(spike_times,ranks,'.',ms=5)
    plt.xlim(0,nbrOfPatchesActiv*duration)
    plt.ylim(0,144)
    plt.ylabel('neuron index',fontsize=22,weight = 'bold')
    plt.xlabel('time [ms]',fontsize=22,weight = 'bold')
    plt.savefig('./Output/V1Layer/Short_raster.png',bbox_inches='tight', pad_inches = 0.1,dpi=300)

#------------------------------------------------------------------------------
if __name__=="__main__":
    data = (sys.argv[1:])
    duration = 125.0
    maxInput = 125.0

    startGetData(duration,maxInput)

