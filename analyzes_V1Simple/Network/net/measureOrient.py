from ANNarchy import *
setup(dt=1.0)#,seed=1001)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import Gabor as gabor
import os.path
from net import *
#-------------------------------------------------------------------------------
def createInput(parameters,maxInput):
    inputIMG = gabor.createGaborMatrix(parameters,patchsize,patchsize)
    maxVal = np.max(np.abs(inputIMG))
    sizeX,sizeY=np.shape(inputIMG)
    inputPatch = np.zeros((sizeX,sizeY,2))
    inputPatch[:,:,0] =np.clip(inputIMG,0,np.max(inputIMG))
    inputPatch[:,:,1] =np.abs(np.clip(inputIMG,np.min(inputIMG),0))
    popInput.rates = (inputPatch/maxVal)*maxInput
#--------------------------------------------------------------------------------
def runNet(duration,maxInput,orienSteps):
    compile()
    loadWeights()
    nbrOfInputs = int(360/orienSteps)

    repeats = 50
    V1Mon=Monitor(popV1,['spike'])
    InhibMon=Monitor(popInhibit,['spike'])

    rec_frEx= np.zeros((numberOfNeurons,patchsize,patchsize,nbrOfInputs))
    rec_frInh= np.zeros((numberOfInhibNeurons,patchsize,patchsize,nbrOfInputs))

    orientation = np.zeros((patchsize,patchsize,nbrOfInputs))

    for x in range(patchsize):
        print('Line number: '+str(x))
        for y in range(patchsize):
            for i in range(nbrOfInputs):
                    orientation[x,y,i] = orienSteps*np.pi/180.0*i
                    parameters = np.array((1.,0.0,0.13,0.2,0.1*patchsize,np.pi/2.0,6.0,6.0,0.0))
                    parameters[1] = orientation[x,y,i]
                    parameters[6] = x
                    parameters[7] = y
                    frEx = np.zeros(numberOfNeurons)
                    frInh= np.zeros(numberOfInhibNeurons)
                    #inputFR = contrastbase**l
                    inputFR = maxInput
                    createInput(parameters,inputFR)
                    for r in range(repeats):
                        simulate(duration)
                        spikesEx = V1Mon.get('spike')
                        spikesInh = InhibMon.get('spike')
                        for j in range(numberOfNeurons):
                            rateEx = len(spikesEx[j])*1000/duration
                            rec_frEx[j,x,y,i] += rateEx
                            if (j < (numberOfInhibNeurons)):
                                rateInh = len(spikesInh[j])*1000/duration
                                rec_frInh[j,x,y,i] += rateInh
                                    
                
                    rec_frEx[:,x,y,i] /= repeats
                    rec_frInh[:,x,y,i] /=repeats

        #-save Data for later statistical analysis-#
    np.save('./work/gabor_frEx',rec_frEx)
    np.save('./work/gabor_frInhib',rec_frInh)
    np.save('./work/gabor_orientation',orientation)

    return()

#--------------------------------------------------------------------------------
def startTuningCurves(duration,maxInput):
    print('start to determine the tuning Curves')
    print('Simulation time : '+str(duration)+' ms')
    print('maximum Input: '+str(maxInput))
    print('------------------------------------------')

    orienSteps = 6
    runNet(duration,maxInput,orienSteps)    
#------------------------------------------------------------------------------
if __name__=="__main__":

    duration = 125.0
    maxInput = 75.0#100.0
    startTuningCurves(duration,maxInput)

