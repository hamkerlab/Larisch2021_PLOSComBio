#----------------------imports and environment---------------------------------
##Neuron - Modell for the analysis##
from ANNarchy import *
setup(dt=1.0)#,seed=101)
import numpy as np
import os
import scipy.io as sio
from net import *

# Present random chosen scene patches
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
#------------------------------------------------------------------------------
def preprocessData(matData):
    # function to split the prewhitened images into on and off counterparts
    images = matData['IMAGES']
    w,h,n_images = np.shape(images)
    new_images = np.zeros((w,h,2,n_images))
    for i in range(n_images):
        new_images[images[:,:,i] > 0, 0, i] = images[images[:,:,i] > 0, i]
        new_images[images[:,:,i] < 0, 1, i] = images[images[:,:,i] < 0, i]*-1

    return(new_images)
#------------------------------------------------------------------------------
def setActivInput(images, maxInput):
    inputPatch,maxVal = getInput(images,patchsize)
    if np.random.rand() <0.5:
        inputPatch=np.fliplr(inputPatch)
    if np.random.rand()<0.5:
        inputPatch=np.flipud(inputPatch)
    popInput.rates = inputPatch/maxVal*maxInput
    return(inputPatch)
#------------------------------main function------------------------------------
def startActiv(duration = 125.0,maxInput=125):
    print('Start to generate activity data')
    nbrOfPatchesActiv = 30000#200000

    matData = sio.loadmat('./Input_Data/IMAGES.mat')
    images = preprocessData(matData)

    compile()
    loadWeights()
      #------init neuron recording------#
    LGNMon = Monitor(popLGN,['spike'])
    V1Mon = Monitor(popV1,['spike'])
    InhibMon=Monitor(popInhibit,['spike'])

    rec_frEx = np.zeros((numberOfNeurons,nbrOfPatchesActiv))
    inptPatches = np.zeros((patchsize,patchsize,2,nbrOfPatchesActiv))


    #-----------stop Time and simulate------# 
    print('start Simulation for Activity')
    for i in range(nbrOfPatchesActiv):
        inptPatches[:,:,:,i] = setActivInput(images,maxInput)
        simulate(duration)         
        spikesEx = V1Mon.get('spike')
        for j in range(numberOfNeurons):
            rateEx = len(spikesEx[j])*1000/duration
            rec_frEx[j,i] = rateEx

        if((i%(nbrOfPatchesActiv/10)) == 0):
            print("Round %i of %i" %(i,nbrOfPatchesActiv))

   
    #-save Data for later statistical analysis-#
    np.save('./work/Active_fr',rec_frEx)
    
    print('finish with Activation')
#------------------------------------------------------------------------------
if __name__=="__main__":
    duration = 125.0
    maxInput = 125.0

    if os.path.isfile('./Input_Data/IMAGES.mat'):
        startActiv(duration,maxInput)
    else:
        print("""No IMAGES.mat found, please download the file from:
        https://www.rctn.org/bruno/sparsenet/IMAGES.mat
        and put in the Input_Data directory""")
