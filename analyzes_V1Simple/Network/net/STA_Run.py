from ANNarchy import *
setup(dt=1.0,seed=101)
import matplotlib as mp
mp.use('Agg')
import numpy as np
import os
from net import *

# Present white noise patches for Spike-Triggered-Average
# R. Larisch, Technische Universitaet Chemnitz
# 2020-04-27

#-------------------------------------------------------------------------------
def setInputSTA():
    inputPatch = np.clip(np.random.normal(10.0,30.0,(patchsize,patchsize,2)),0,100.0)
    popInput.rates = inputPatch 
#------------------------------main function------------------------------------
def startSTA(duration=125.0):
    print('Start to calculate the STA')

    nbrOfPatchesSTA =50000 #
    compile()
    loadWeights()
      #------init neuron recording------#
    V1Mon = Monitor(popV1,['spike'])
    InhibMon=Monitor(popInhibit,['spike'])

    rec_frEx = np.zeros((numberOfNeurons,nbrOfPatchesSTA))

    rec_frInh= np.zeros((numberOfInhibNeurons,nbrOfPatchesSTA))

    Input = np.zeros((nbrOfPatchesSTA,patchsize,patchsize,2))
    print('start STA Simulation')
    for i in range(nbrOfPatchesSTA):
        setInputSTA()
        simulate(duration)         
        spikesEx = V1Mon.get('spike')
        spikesInh = InhibMon.get('spike')
        for j in range(numberOfNeurons):
            rateEx = len(spikesEx[j])*1000/duration
            rec_frEx[j,i] = rateEx
            if (j < (numberOfInhibNeurons)):
                rateInh = len(spikesInh[j])*1000/duration
                rec_frInh[j,i] = rateInh
        Input[i,:,:,:] = popInput.rates
        if((i%(nbrOfPatchesSTA/10)) == 0):
            print("Round %i of %i" %(i,nbrOfPatchesSTA))

    np.save('./work/STA_Input',Input)
    np.save('./work/STA_frExc',rec_frEx)
    np.save('./work/STA_frInh',rec_frInh)

    print("finish with STA and STC")

#------------------------------------------------------------------------------
if __name__=="__main__":
    data = (sys.argv[1:])
    duration = 125.0
    startSTA(duration)

