import numpy as np
import matplotlib.pyplot as plt
import os

# Sparseness based on Rolls-Tovee measure (Rolls and Tovee,1995)
# by Vinje and Gallant(Vinje and Gallant,2000)
# 0< value <1 , higher than sparser distributions
def calculateVinjeGallantSparseness(frEx):
    nbrOfNeurons,nbrOfPatches = np.shape(frEx)
    normFrEx = frEx/np.max(frEx)


    sparsenessOverInput = []
        
    for i in range(nbrOfPatches):
        frPatch = normFrEx[:,i]
        s1 = np.sum(frPatch/np.float(nbrOfNeurons))**2
        s2 = np.sum(frPatch**2)/np.float(nbrOfNeurons)
        if s2 !=0.0:
            d = 1 - (1/np.float(nbrOfNeurons))
            sparsenessOverInput.append((1-(s1/s2))/d)
    
    return(sparsenessOverInput)



def calculateAndPlotVinjeGallant(frEx):
    print('Sparseness at Vinje and Gallant')
    sparsOverInputVG = calculateVinjeGallantSparseness(frEx)


    plt.figure()
    plt.hist(sparsOverInputVG,15)
    plt.xlim(xmin=0.0,xmax=1.0)
    plt.title('mean: '+str(np.round(np.mean(sparsOverInputVG),4)))
    plt.xlabel('Population Sparseness',fontsize=18)
    plt.ylabel('# of Inputs',fontsize=18)
    plt.savefig('./Output/Activ/Sparseness/sparsVinjeGallant_histInput.png',bbox_inches='tight', pad_inches = 0.1)

    np.save('./work/V&G_Population',sparsOverInputVG)

def calculateAndPlotSparseness():
    print('Start to calculate and Plot the Sparseness')
    if not os.path.exists('Output/Activ/'):
        os.mkdir('Output/Activ/')
    if not os.path.exists('Output/Activ/Sparseness/'):
        os.mkdir('Output/Activ/Sparseness/')

    frEx = np.load('work/Active_fr.npy')

    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 18)
    plt.rc('ytick',labelsize = 18)

    calculateAndPlotVinjeGallant(frEx)

    print('Finish with Sparseness!')

if __name__=="__main__":
    calculateAndPlotSparseness()
