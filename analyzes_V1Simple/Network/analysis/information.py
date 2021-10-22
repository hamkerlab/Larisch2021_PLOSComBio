import matplotlib.pyplot as plt
import numpy as np
import os
#------------------------------------------------------------------------------
# calculate the bit/spike information transmission after Vinje and Galant, 2002
# one time bin = 125 ms long-> every input presentation is one time bin
#------------------------------------------------------------------------------
# calculate the total response entropy (H(r)) as the overall variablity of the response of a neuron over all stimuli
def calcResponseEntropy(fr):
    nbrCells,nbrInputs,nbrRepeats = np.shape(fr)
    entropyCellResponse = np.zeros(nbrCells)
    #meanFR = np.mean(fr,axis=2)
    #print(np.max(meanFR))
    frA = np.reshape(fr,(nbrCells,nbrInputs*nbrRepeats))
    for i in range(nbrCells):    
        maxFr = int(np.max(frA[i]))
        probFR,histFR = np.histogram(frA[i],maxFr)
        idx = np.where(probFR > 0)
        probFR = probFR[idx].astype(float)/np.sum(probFR[idx])
        entropyCellResponse[i] = - np.sum(probFR*np.log2(probFR)) #log2 ?

    meanPopFR = np.mean(frA,axis=0)#np.mean(meanFR,axis=0)
    maxFr = int(np.max(meanPopFR))
    probFR,histFR = np.histogram(meanPopFR,maxFr)
    probFR = probFR.astype(float)/np.sum(probFR)
    entropyPopResponse = - np.sum(probFR*np.log2(probFR))

    return(entropyCellResponse,entropyPopResponse)
#------------------------------------------------------------------------------
# calculate the conditinal response entropy (H(r|s)), equal to noise entropy, as the average variability in responses evoked by a single stimulus
def calcNoiseEntropy(fr):
    nbrCells,nbrInputs,nbrRepeats = np.shape(fr)

    noiseEntropy = np.zeros((nbrCells,nbrInputs))
    for i in range(nbrCells):
        cellResp = fr[i,:,:]
        stim = np.squeeze(np.where(np.sum(cellResp,axis=1) > 0.0))
        for s in stim:
            maxResp =int(np.max(cellResp[s]))
            probFR,histFR = np.histogram(cellResp[s],maxResp)
            idx = np.where(probFR > 0)
            probFR = probFR[idx].astype(float)/np.sum(probFR[idx])
            noiseEntropy[i,s] = -np.sum(probFR * np.log2(probFR))
    return(noiseEntropy)
#------------------------------------------------------------------------------
def analyseInforTrans():

    if not os.path.exists('./Output/Activ'):
        os.mkdir('./Output/Activ')

    if not os.path.exists('./Output/Activ/Stats'):
        os.mkdir('./Output/Activ/Stats')

    frV1 = np.load('./work/fluctuation_frExc.npy')
    binSize = 125 #ms
    cellEntropy,popEntropy = calcResponseEntropy(frV1)
    noiseEntropyPerPatch = calcNoiseEntropy(frV1)
    noiseEntropy = np.mean(noiseEntropyPerPatch,axis=1)
    mutualInformationPerBin = cellEntropy - noiseEntropy
    mutualInformationPerSecond = mutualInformationPerBin/binSize
    meanFR = np.mean(frV1,axis=2)
    meanFR = np.mean(meanFR,axis=1)
    mutualInformationPerMeanSpike = mutualInformationPerBin/(meanFR)

    plt.figure()
    plt.hist(mutualInformationPerMeanSpike)
    plt.title(np.round(np.mean(mutualInformationPerMeanSpike),4))
    plt.xlabel('mutual information per Spike')
    plt.ylabel('# cells')
    plt.savefig('./Output/Activ/Stats/mutualPerSpikeVG_hist.png')
#------------------------------------------------------------------------------
if __name__=="__main__":
    analyseInforTrans()
