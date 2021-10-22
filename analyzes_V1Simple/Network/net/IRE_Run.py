from ANNarchy import *
setup(dt=1.0,seed=101)
import matplotlib as mp
mp.use('Agg')
import numpy as np
import os
from net import *
import scipy.io as sio

# Split the natural scenes into single overlapping patches to reconstruct the scenes after wards
# R. Larisch, Technische Universitaet Chemnitz
# 2020-04-27

#------------------------------------------------------------------------------------
def makeShortInput(image):
    w,h = np.shape(image)[0:2]
    nbrOfPatchesIRE = w/patchsize * h/patchsize
    patches = np.zeros((nbrOfPatchesIRE,patchsize,patchsize,2))
    for i in range(w/patchsize):
        for j in range(h/patchsize):
            patches[(i*w/patchsize)+(j) ,:,:,:] = image[0+(i*patchsize):patchsize+(i*patchsize) ,0+(j*patchsize):patchsize+(j*patchsize) ,:]
    return(nbrOfPatchesIRE,patches)
#-------------------------------------------------------------------------------------
def makePixelWiseInput(image,pixelStep):
    w,h = np.shape(image)[0:2]
  
    nbrOfPatchesIRE = int(((w-patchsize)/pixelStep) * ((h-patchsize)/pixelStep))
    patches = np.zeros((nbrOfPatchesIRE,patchsize,patchsize,2))
    index=0
    for i in range(0,w-patchsize ,pixelStep):
        for j in range(0,h-patchsize,pixelStep):
            patches[index,:,:,:] = image[0+(1*i):patchsize+(1*i), 0+(1*j):patchsize+(1*j), :]
            index=index+1
    return(patches)
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
#------------------------------main function------------------------------------
def startIRE(duration = 125.0,maxInput=125):
    print('Start to Validate the IRE')
    #imagesMat = sio.loadmat('Input_Data/input_IRE.mat')
    #image = imagesMat['inputImage']
    imagesMat = sio.loadmat('./Input_Data/IMAGES.mat')
    images = preprocessData(imagesMat)
    w = 504
    h = w
    pixelStep = 3
    nbrOfPatchesIRE = int(((w-patchsize)/pixelStep) * ((h-patchsize)/pixelStep))

    output = np.zeros((nbrOfPatchesIRE,patchsize,patchsize,2))

    compile()
    loadWeights()

    #-----init recording-----#
    V1Mon = Monitor(popV1,['spike'])
    InhibMon=Monitor(popInhibit,['spike'])
    LGNMon = Monitor(popLGN,['spike'])
    rec_frLGN = np.zeros((patchsize*patchsize*2,10,nbrOfPatchesIRE)) 
    rec_frEx = np.zeros((numberOfNeurons,10,nbrOfPatchesIRE))
    inp_Images = np.zeros((10,w,h,2))

    for p in range(10):
        image= images[:,:,:,p]
        image = image[4:512-4,4:512-4]
        inp_Images[p] = image
        wMax = np.max(np.abs(image))
        #w,h = np.shape(image)[0:2]
        patches = makePixelWiseInput(image,pixelStep)#makeShortInput(image)
        print('start IRE Simulation')
        #print(nbrOfPatchesIRE)
        repeats = 20
        for i in range(nbrOfPatchesIRE):
            for k in range(repeats):
                popInput.rates=(patches[i,:,:,:]/wMax) * maxInput
                simulate(duration)         
                spikesLGN = LGNMon.get('spike')
                spikesEx = V1Mon.get('spike')
                for j in range(numberOfNeurons*2):
                    rateLGN = len(spikesLGN[j])*1000.0/duration
                    rec_frLGN[j,p,i] += rateLGN            
                for j in range(numberOfNeurons):
                    rateEx = len(spikesEx[j])*1000.0/duration
                    rec_frEx[j,p,i] += rateEx

            rec_frLGN[:,p,i] /= repeats
            rec_frEx[:,p,i] /= repeats
            if((i%1000) == 0):
                print("Round %i of %i" %(i,nbrOfPatchesIRE))

    #---------- plot output datas-- --------#
    np.save('./work/IRE_fr',rec_frEx)
    np.save('./work/IRE_singlePatches',patches)
    np.save('./work/IRE_LGNFR',rec_frLGN)
    np.save('./work/IRE_Images',inp_Images)
    print("finish with IRE")
#------------------------------------------------------------------------------
if __name__=="__main__":
    data = (sys.argv[1:])
    duration = 125.0
    maxInput = 125.0
    if os.path.isfile('./Input_Data/IMAGES.mat'):
        startIRE(duration,maxInput)
    else:
        print("""No IMAGES.mat found, please download the file from:
        https://www.rctn.org/bruno/sparsenet/IMAGES.mat
        and put in the Input_Data directory""")
    

