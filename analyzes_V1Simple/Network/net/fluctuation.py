from ANNarchy import *
setup(dt=1.0,seed=101)
import matplotlib as mp
mp.use('Agg')
import numpy as np
import os
import scipy.io as sio
from net import *


def preprocessData(matData):
    # function to split the prewhitened images into on and off counterparts
    images = matData['IMAGES']
    w,h,n_images = np.shape(images)
    new_images = np.zeros((w,h,2,n_images))
    for i in range(n_images):
        new_images[images[:,:,i] > 0, 0, i] = images[images[:,:,i] > 0, i]
        new_images[images[:,:,i] < 0, 1, i] = images[images[:,:,i] < 0, i]*-1

    return(new_images)
#----------------------------------------------------------
def startTask(duration,maxInput):
    print('record activity on one input')
    compile()
    loadWeights()

    matData = sio.loadmat('./Input_Data/IMAGES.mat')
    images = preprocessData(matData)

    size = np.shape(images)[0]
    nbrPatches = 1000
    nbrSamples = 100    
    monV1 = Monitor(popV1,['spike'])

    rec_Input = np.zeros((nbrPatches,12,12,2))
    rec_frExc = np.zeros((numberOfNeurons,nbrPatches,nbrSamples))

    for i in range(nbrPatches):
        xPos = np.random.randint(0,size-patchsize)
        yPos = np.random.randint(0,size-patchsize)
        pictNbr = np.random.randint(0,10)
        inputPatch = images[xPos:xPos+patchsize,yPos:yPos+patchsize,:,pictNbr]
        maxVal = np.max(images[:,:,:,pictNbr])
        rec_Input[i] = inputPatch

        for j in range(nbrSamples):
            reset(populations=True,projections=False)
            popInput.rates = inputPatch/maxVal* maxInput#*contrastF[np.random.randint(3)]
            simulate(duration)

            spikesEx = monV1.get('spike')

            for n in range(numberOfNeurons):
                rateEx = len(spikesEx[n])#*1000/duration
                rec_frExc[n,i,j] = rateEx


    np.save('./work/fluctuation_frExc',rec_frExc)
    np.save('./work/fluctuation_Input',rec_Input)
#------------------------------------------------------------------------------
if __name__=="__main__":
    duration = 125.0
    maxInput = 125.0

    if os.path.isfile('./Input_Data/IMAGES.mat'):
        startTask(duration,maxInput)
    else:
        print("""No IMAGES.mat found, please download the file from:
        https://www.rctn.org/bruno/sparsenet/IMAGES.mat
        and put in the Input_Data directory""")
