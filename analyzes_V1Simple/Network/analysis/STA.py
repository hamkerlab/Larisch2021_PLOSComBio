import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path

# R. Larisch, Technische Universitaet Chemnitz
# 2015-06-23
# ueberarbeiten, STA und STC dynamisch fuer jede moegliche Neuronenpopulation!

#------------------------------------------------------------------------------
def setSubplotDimension(a):
    x = 0
    y = 0
    if ( (a % 1) == 0.0):
        x = a
        y =a
    elif((a % 1) < 0.5):
        x = round(a) +1
        y = round(a)
    else :
        x = round(a)
        y = round(a)
    return (x,y)   
#------------------------------------------------------------------------------
def calcSTA(inpt,frExc,frInh):
    # calculate based on the simulation results the Spike - Triggered - Average of exitatory and inhibitory Neurons
    # Input -> matrix with all input patches from the simulation, shape = (number of input patches, size of one patch )
    # frEx -> fire rate of the exitatory neurons, for every input patch, shape=(number of Neurons, number of input patches)
    # frInh -> fire rate of the inhibitory neurons, for every input patch, shape=(number of Neurons, number of input patches)

    print('calculate STA')
    inpt = inpt/np.max(inpt)
    nbrOfNeurons = np.shape(frExc)[0]
    patchsize = np.shape(inpt)[1]
    STAV1 = np.zeros((nbrOfNeurons,2,patchsize,patchsize))
    STAIN = np.zeros((int(nbrOfNeurons/4),2,patchsize,patchsize)) 
    for i in range(nbrOfNeurons):
        spk = np.squeeze(frExc[i,np.nonzero(frExc[i,:])])
        actInp = inpt[np.nonzero(frExc[i,:])]
        sta= actInp.T*spk
        sta= np.sum(sta,axis=3)
        sta /= np.sum(spk)
        STAV1[i] = sta
        if i < (int(nbrOfNeurons/4)):
            spk = np.squeeze(frInh[i,np.nonzero(frInh[i,:])])   
            actInp = inpt[np.nonzero(frInh[i,:])]
            sta= actInp.T*spk
            sta= np.sum(sta,axis=3)
            sta /= np.sum(spk)
            STAIN[i] = sta
            
    return(STAV1,STAIN)
#-----------------------------------------------------------------------------
def plotSTA(STAV1,STAIN):
    print('plot STA')
    fig = plt.figure(figsize=(8,8))
    x,y = setSubplotDimension(np.sqrt(np.shape(STAV1)[0]))
    for i in range(np.shape(STAV1)[0]):
        field = (STAV1[i,0,:,:] - STAV1[i,1,:,:])
        plt.subplot(x,y,i+1)
        plt.axis('off')
        im = plt.imshow(field.T,cmap=plt.get_cmap('gray'),aspect='auto',interpolation='none')#,vmin=wMin,vmax=wMax)
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STA/STAEX.png',bbox_inches='tight', pad_inches = 0.1,dpi=300)

    fig = plt.figure(figsize=(8,8))
    x,y = setSubplotDimension(np.sqrt(np.shape(STAIN)[0]))
    for i in range(np.shape(STAIN)[0]):
        field = (STAIN[i,0,:,:] - STAIN[i,1,:,:])
        plt.subplot(x,y,i+1)
        im = plt.imshow(field.T,cmap=plt.get_cmap('gray'),aspect='auto',interpolation='none')#,vmin=wMin,vmax=wMax)
        plt.axis('off')
    plt.subplots_adjust(hspace=0.25,wspace=0.25)
    fig.savefig('./Output/STA/STAIN.png',bbox_inches='tight', pad_inches = 0.1,dpi=300)

#------------------------------------------------------------------------------

def calculateSTA():
    print('Start to estimate STA')

    if not os.path.exists('./Output/STA'):
        os.mkdir('./Output/STA')

    Input = np.load('./work/STA_Input.npy')
    frEx  = np.load('./work/STA_frExc.npy')
    frInh = np.load('./work/STA_frInh.npy')
    STAV1,STAIN = calcSTA(Input,frEx,frInh)
    
    #---plotSTA----#
    plotSTA(STAV1,STAIN)

    print("finish with STA")
#------------------------------------------------------------------------------
if __name__=="__main__":
    calculateSTA()
