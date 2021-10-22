from ANNarchy import *
setup(dt=1.0,seed=1001)
import matplotlib.pyplot as plt
import Gabor_sinus as gabor
import os.path
from net import *

# Present sinus gratings in different orientations and contrast levels to measure the tuning curves
# R. Larisch, Technische Universitaet Chemnitz
# 2020-04-27

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
def runNet(maxDegree,stepDegree,stepPhase,stepFrequency,duration,maxInput):
    compile()
    loadWeights()
    phaseShifts = np.pi / float(stepPhase)
    nbrOfInputs = int(maxDegree/stepDegree)
    frequShift = (0.15 - 0.05)/float(stepFrequency)  #from 0.5 to 1.5 * patchsize#
    repeats = 50
    contrastLVLs = 7

    V1Mon=Monitor(popV1,['spike','vm','g_Exc','g_Inh'])

    rec_frEx = np.zeros((numberOfNeurons,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))
    rec_Exc_gExc= np.zeros((numberOfNeurons,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))
    rec_Exc_gInh= np.zeros((numberOfNeurons,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))
    rec_Exc_vm  = np.zeros((numberOfNeurons,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))


    InhibMon = Monitor(popInhibit,['spike'])
    rec_frInh = np.zeros((numberOfInhibNeurons,nbrOfInputs,stepPhase,stepFrequency,contrastLVLs,repeats))

    orientation = np.zeros((nbrOfInputs))
   
    
    for i in range(nbrOfInputs):
        orientation[i] = stepDegree*np.pi/180.0*i
        print("Orientation: "+str(orientation[i]))
        for k in range(stepPhase):
            for f in range(stepFrequency):
                for c in range(contrastLVLs):
                    freq = (0.05+ (0.05*f))*patchsize 
                    phase = phaseShifts*k
                    #parameters = np.array((1.,0.0,0.13,0.2,np.pi/2.0,np.pi/2.0,patchsize,patchsize,0.0))
                    parameters = np.array((1.,0.0,0.13,0.2,0.1*patchsize,np.pi/2.0,patchsize,patchsize,0.0))
                    parameters[1] = orientation[i]
                    parameters[4] = freq
                    parameters[5] = phase
                    parameters[6] = patchsize/2
                    parameters[7] = patchsize/2
                    inputFR = maxInput/contrastLVLs + (maxInput/contrastLVLs)*c
                    for r in range(repeats):
                        createInput(parameters,inputFR)
                        simulate(duration)
                        spikesEx = V1Mon.get('spike')
                        gExcEx = V1Mon.get('g_Exc')
                        gInhEx = V1Mon.get('g_Inh')
                        vmEx = V1Mon.get('vm')
                        spikesInh = InhibMon.get('spike')
                        #gExcIn = InhibMon.get('g_Exc')

                        for j in range(numberOfNeurons):
                            rateEx = len(spikesEx[j])*1000/duration
                            rec_frEx[j,i,k,f,c,r] = rateEx
                            rec_Exc_gExc[j,i,k,f,c,r] = np.mean(gExcEx[:,j])
                            rec_Exc_gInh[j,i,k,f,c,r] = np.mean(gInhEx[:,j])
                            rec_Exc_vm[j,i,k,f,c,r] = np.mean(vmEx[:,j])

                            if (j < (numberOfInhibNeurons)):
                                rateInh = len(spikesInh[j])*1000/duration
                                rec_frInh[j,i,k,f,c,r] = rateInh

        #-save Data for later statistical analysis-#
    np.save('./work/TuningCurve_sinus_frEx',rec_frEx)
    np.save('./work/TuningCurve_sinus_gExc_Ex',rec_Exc_gExc)
    np.save('./work/TuningCurve_sinus_gInh_Ex',rec_Exc_gInh)
    np.save('./work/TuningCurve_sinus_VM_Ex',rec_Exc_vm)

    np.save('./work/TuningCurve_sinus_orientation',orientation)
    np.save('./work/TuningCurve_sinus_frInhib',rec_frInh)

    return()

#--------------------------------------------------------------------------------
def calcTuningCC(fr):
    #estimate the best fit for tuning curve over all parameters -> every cell can have different parameter
    nb_cells,nb_degree,nb_phases,nb_freq,trials = np.shape(fr)
    tuningC = np.zeros((nb_cells,nb_degree,trials))
    meanFR = np.mean(fr,axis=4)
    parameters = np.zeros((nb_cells,3)) # orientation, phase, frequency

    for i in range(nb_cells):
        goodIndx = (np.where(meanFR[i] ==np.max(meanFR[i])))
        sumFR = np.sum(meanFR[i,goodIndx[1],goodIndx[2]],axis=1)
        probBestIdx = np.where(sumFR ==np.max(sumFR)) # Index with probaly the best tuning Curve
        orientID = goodIndx[0][probBestIdx[0][0]]
        phaseID = goodIndx[1][probBestIdx[0][0]]
        freqID = goodIndx[2][probBestIdx[0][0]]
        tuningC[i] = fr[i,:,phaseID,freqID,:]
        parameters[i] =[orientID,phaseID,freqID]

    return(tuningC,parameters)
#--------------------------------------------------------------------------------
def calcTuningC_variable(data,params):
    #get the curve of the neuron variable, depending on the fit of the tuning curve
    nb_cells,nb_degree,nb_phases,nb_freq,trials = np.shape(data)
    tuningC = np.zeros((nb_cells,nb_degree,trials))

    for i in range(nb_cells):
        orientID,phaseID,freqID =params[i,:]
        tuningC[i] = data[i,:,int(phaseID),int(freqID),:]
    return(tuningC)
#--------------------------------------------------------------------------------
def calcRelativTuningCurve(tuningCurve):
    # shift the values of the neuron fire rates, that they are sorted relative to the optimal orientation in the middle
    n_cells, n_degrees, n_trials = np.shape(tuningCurve)
    relativTC = np.zeros((n_cells,n_degrees,n_trials))
    # average over trials 
    mean_TC = np.mean(tuningCurve,axis=2)
    for i in range(n_cells):
        orientIdx = np.where(mean_TC[i] == np.max(mean_TC[i]))#Index for orientation, wehre the fire rate is maximum
        relativTC[i] = np.roll(tuningCurve[i,:,:], int(n_degrees/2) - orientIdx[0][0])
    return(relativTC)

def calcRelativTuningCurves(sc,vm,gE,gI):
    # shift the values of the neuron fire rates, that they are sorted relative to the optimal orientation in the middle
    n_cells,n_orient,n_cont,n_rep = np.shape(sc)
    mean_SC = np.mean(sc,axis=3)
    mean_SC = np.mean(mean_SC,axis=2)

    r_m_SC = np.zeros((n_cells,n_orient))
    r_SC = np.zeros((n_cells,n_orient,n_cont,n_rep))
    r_VM = np.zeros((n_cells,n_orient,n_cont,n_rep))
    r_gE = np.zeros((n_cells,n_orient,n_cont,n_rep))
    r_gI = np.zeros((n_cells,n_orient,n_cont,n_rep))
    for i in range(n_cells):
        orientIdx = np.where(mean_SC[i,:] == np.max(mean_SC[i,:]))#Index for orientation, wehre the fire rate is maximum
        r_m_SC[i] = np.roll(mean_SC[i,:], int(n_orient/2) - orientIdx[0][0])
        r_SC[i,:] = np.roll(sc[i,:], int(n_orient/2) - orientIdx[0][0],axis=0)
        r_VM[i,:] = np.roll(vm[i,:], int(n_orient/2) - orientIdx[0][0],axis=0)
        r_gE[i,:] = np.roll(gE[i,:], int(n_orient/2) - orientIdx[0][0],axis=0)
        r_gI[i,:] = np.roll(gI[i,:], int(n_orient/2) - orientIdx[0][0],axis=0)
    return(r_SC,r_VM,r_gE,r_gI)

#--------------------------------------------------------------------------------
def startTuningCurves(duration,maxInput):
    print('start to determine the tuning Curves over sinus gratings')
    print('Simulation time : '+str(duration)+' ms')
    print('maximum Input: '+str(maxInput)+ ' Hz')
    print('------------------------------------------')
    
    if not os.path.exists('./Output/TC_sinus/'):
        os.mkdir('./Output/TC_sinus/')
    if not os.path.exists('./Output/TC_sinus/excitatory/'):
        os.mkdir('./Output/TC_sinus/excitatory/')


    maxDegree = 360
    orienSteps = 8#2#5
    phaseSteps = 8#8
    frequSteps = 4#4
    runNet(maxDegree,orienSteps,phaseSteps,frequSteps,duration,maxInput)    

    frExc = np.load('./work/TuningCurve_sinus_frEx.npy')
    frInh = np.load('./work/TuningCurve_sinus_frInhib.npy')
    orient =np.load('./work/TuningCurve_sinus_orientation.npy')
    gExc_Exc = np.load('./work/TuningCurve_sinus_gExc_Ex.npy')
    gInh_Exc = np.load('./work/TuningCurve_sinus_gInh_Ex.npy')
    vm_Exc = np.load('./work/TuningCurve_sinus_VM_Ex.npy')

    nb_cells,nb_degree,nb_phases,nb_freq,nb_contrast,trials = np.shape(frExc)

    print('Start to calculate the relativ tuning curve of each neuron')

    tuningCExc = np.zeros((nb_cells,nb_degree,nb_contrast,trials))
    tuningCgExcE = np.zeros((nb_cells,nb_degree,nb_contrast,trials))
    tuningCgInhE = np.zeros((nb_cells,nb_degree,nb_contrast,trials))
    tuningCvmE = np.zeros((nb_cells,nb_degree,nb_contrast,trials))

    relativTCExc = np.zeros((nb_cells,nb_degree,nb_contrast,trials))
    relativTCgExcE = np.zeros((nb_cells,nb_degree,nb_contrast,trials))
    realtivTCgInhE = np.zeros((nb_cells,nb_degree,nb_contrast,trials))
    realtivTCvmE = np.zeros((nb_cells,nb_degree,nb_contrast,trials))

    tuningCInhib = np.zeros((int(nb_cells/4),nb_degree,nb_contrast,trials))
    relativTCInhib = np.zeros((int(nb_cells/4),nb_degree,nb_contrast))
   
    parameters_Exc = np.zeros((nb_cells,3,nb_contrast))
    parameters_Inh = np.zeros((int(nb_cells/4),3,nb_contrast))

    for i in range(nb_contrast):
        tuningCExc[:,:,i,:],parameters_Exc[:,:,i] = calcTuningCC(frExc[:,:,:,:,i,:])
        tuningCgExcE[:,:,i,:] = calcTuningC_variable(gExc_Exc[:,:,:,:,i,:],parameters_Exc[:,:,i])
        tuningCgInhE[:,:,i,:] = calcTuningC_variable(gInh_Exc[:,:,:,:,i,:],parameters_Exc[:,:,i])
        tuningCvmE[:,:,i,:] = calcTuningC_variable(vm_Exc[:,:,:,:,i,:],parameters_Exc[:,:,i]) 

        tuningCInhib[:,:,i,:],parameters_Inh[:,:,i] = calcTuningCC(frInh[:,:,:,:,i,:])


    # shift the tuning curve relativ to preffered orientation
    relativTCExc,realtivTCvmE,relativTCgExcE,realtivTCgInhE = calcRelativTuningCurves(tuningCExc,tuningCvmE,tuningCgExcE,tuningCgInhE)

    np.save('./work/TuningCurves_sinus_Exc',tuningCExc)
    np.save('./work/TuningCurves_sinus_Exc_gE',tuningCgExcE)
    np.save('./work/TuningCurves_sinus_Exc_gI',tuningCgInhE)
    np.save('./work/TuningCurves_sinus_Exc_VM',tuningCvmE)
    np.save('./work/TuningCurves_sinus_Exc_parameters',parameters_Exc)

    np.save('./work/TuningCurves_sinus_Inh',tuningCInhib)
    np.save('./work/TuningCurves_sinus_Inh_parameters',parameters_Inh)

    np.save('./work/TuningCurvesRelativ_sinus_Exc',relativTCExc)
    np.save('./work/TuningCurvesRelativ_sinus_Exc_gE',relativTCgExcE)
    np.save('./work/TuningCurvesRelativ_sinus_Exc_gI',realtivTCgInhE)
    np.save('./work/TuningCurvesRelativ_sinus_Exc_VM',realtivTCvmE)
#------------------------------------------------------------------------------
if __name__=="__main__":
    data = (sys.argv[1:])
    duration = 125.0
    maxInput = 100.0#75.0

    startTuningCurves(duration,maxInput)

