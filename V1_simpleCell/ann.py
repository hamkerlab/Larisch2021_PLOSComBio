#----------------------imports and environment---------------------------------
import matplotlib as mp
mp.use('Agg')
from ANNarchy import *
setup(dt=1.0)#,seed=1)
from time import *
import plots as plt
import numpy as np
import scipy.io as sio
#------------------------global Variables------------------------------------
nbrOfPatches = 400000
duration = 125
patchsize = 12
inputNeurons = patchsize*patchsize*2
n_exN = patchsize*patchsize
n_inN = int(n_exN/4)
#---------------------------------neuron definitions-------------------------

## Neuron Model for LGN/Input Layer ##
params = """
EL = -70.4      :population
VTrest = -50.4  :population
taux = 15.0     :population
"""

inpt_eqs ="""
    dg_exc/dt = EL/1000 : min=EL, init=-70.6
    Spike = 0.0
    dresetvar / dt = 1/(1.0) * (-resetvar)
    dxtrace /dt = 1/taux * (- xtrace )
    """

spkNeurLGN = Neuron(parameters=params,
                          equations=inpt_eqs,
                          reset="""g_exc=EL 
                                   Spike = 1.0
                                   resetvar=1.0
                                   xtrace+= 1/taux""", 
                          spike="""g_exc > VTrest""")

## Neuron Model for V1-Layer, after Clopath et al.(2010) ##
params = """
gL = 30.0         :population
DeltaT = 2.0      :population
tauw = 144.0      :population
a = 4.0           :population  
b = 0.0805        :population  
EL = -70.6        :population
C = 281.0         :population  
tauz = 40.0       :population  
tauVT= 50.0       :population  
Isp = 400.0       :population  
VTMax = -30.4     :population 
VTrest = -50.4    :population  
taux = 15.0       :population  
tauLTD = 10.0     :population  
tauLTP= 7.0       :population  
taumean = 750.0   :population  
tau_gExc = 1.0    :population  
tau_gInh= 10.0    :population
"""


"""
To enable the correct up- and downswing of the membrane potential after a spike, the equation is divided in different parts.
Via the 'state' parameter, the correct part is used for calculation in the current time step.
For a better understanding of the implementation, the line

dvm/dt = if state>=2:+3.462 else: if state==1:-(vm+51.75)+1/C*(Isp - (wad+b))+g_Exc-g_Inh else:1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z ) + g_Exc -g_Inh: init = -70.6

can be rewritten as

alpha = if state>=2: beta else: if state==1: gamma else: delta: init = epsilon.

with:
alpha = dvm/dt
beta = +3.462
gamma = -(vm+51.75)+1/C*(Isp - (wad+b))+g_Exc-g_Inh
delta = 1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z ) + g_Exc -g_Inh
epsilon = -70.6

With alpha = dvm/dt,this is the normal differential term for the change of the membrane potential (vm) through time.
The epsilon = -70.6 is the initialization value of vm. 
The if-statements, depending on a variable called 'state', decide how the membrane potential should be calculated. 
The state variable decreases each timestep until it reaches 0, and will be set to 2 after a spike.
So if state >=2 calculate beta. If state ==1, calculate gamma. Otherwise calculate delta. Where beta and gamma are there to calculate correctly the up- and downswing for 2ms after a spike, 
and delta is the normal equation for the membrane potential as mentioned in Clopath et al. (2010).
"""


neuron_eqs = """
noise = Normal(0.0,1.0)
dvm/dt = if state>=2:+3.462 else: if state==1:-(vm+51.75)+1/C*(Isp - (wad+b))+g_Exc-g_Inh else:1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z ) + g_Exc -g_Inh: init = -70.6
dvmean/dt = (pos(vm - EL)**2 - vmean)/taumean    :init = 0.0
dumeanLTD/dt = (vm - umeanLTD)/tauLTD : init=-70.0
dumeanLTP/dt = (vm - umeanLTP)/tauLTP : init =-70.0
dxtrace /dt = (- xtrace )/taux
dwad/dt = if state ==2:0 else:if state==1:+b/tauw else: (a * (vm - EL) - wad)/tauw : init = 0.0
dz/dt = if state==1:-z+Isp-10 else:-z/tauz  : init = 0.0
dVT/dt =if state==1: +(VTMax - VT)-0.4 else:(VTrest - VT)/tauVT  : init=-50.4
dg_Exc/dt = -g_Exc/tau_gExc
dg_Inh/dt = -g_Inh/tau_gInh
state = if state > 0: state-1 else:0
Spike = 0.0
dresetvar / dt = 1/(1.0) * (-resetvar)
           """
spkNeurV1 = Neuron(parameters = params,equations=neuron_eqs,spike="""(vm>VT) and (state==0)""",
                         reset="""vm = 29.0
                                  state = 2.0 
                                  VT = VTMax
                                  Spike = 1.0
                                  resetvar = 1.0
                                  xtrace+= 1/taux""")

#----------------------------------synapse definitions----------------------

#----- Synapse from Poisson to Input-Layer -----#
inputSynapse =  Synapse(
    parameters = "",
    equations = "",
    pre_spike = """
        g_target += w
                """
)

#--STDP Synapses after Clopath et. al(2010)for Input- to Exitatory- Layer--#
equatSTDP = """
    ltdTerm = if w>wMin : (aLTD*(post.vmean/urefsquare)*pre.Spike * pos(post.umeanLTD - thetaLTD)) else : 0.0
    ltpTerm = if w<wMax : (aLTP * pos(post.vm - thetaLTP) *(pre.xtrace)* pos(post.umeanLTP - thetaLTD)) else : 0.0
      dw/dt = ( -ltdTerm + ltpTerm) :min=0.0,explicite"""

parameterFF="""
urefsquare = 60.0       :projection
thetaLTD = -70.6        :projection
thetaLTP = -45.3        :projection
aLTD = 0.00014*0.75     :projection
aLTP = 0.00018*0.75     :projection
wMin = 0.0              :projection
wMax =5.0               :projection
"""

ffSyn = Synapse( parameters = parameterFF,
    equations= equatSTDP, 
    pre_spike='''g_target += w''')


#------STDP Synapse like above, with other parameters for input -> inhibitory ----#
parameterInptInhib="""
urefsquare = 55.0       :projection
thetaLTD = -70.6        :projection
thetaLTP = -45.3        :projection
aLTD = 0.00014 *0.3     :projection
aLTP = 0.00018 *0.3     :projection
wMin = 0.0              :projection
wMax = 3.0              :projection
"""

ff2Syn = Synapse(parameters = parameterInptInhib,
    equations=equatSTDP,
    pre_spike = '''g_target +=w ''')          
#------STDP Synapse like above, with other parameters for exitatory -> inhibitory ----#
parameterInhib="""
urefsquare = 55.0       :projection
thetaLTD = -70.6        :projection
thetaLTP = -45.3        :projection
aLTD = 0.00014 *0.1     :projection
aLTP = 0.00012 *0.1     :projection
wMin = 0.0              :projection
wMax = 1.0              :projection
"""

InhibSyn = Synapse( parameters = parameterInhib,
    equations=equatSTDP , 
    pre_spike='''g_target += w''')

 
##-> delete later, if not necessary
#------------- iSTDP Synapse for Inhibitory- to Exitatory- Layer -----------------------#

equatInhib = ''' w = w :min=hMin, max=hMax
                 dtracePre /dt = - tracePre/ taupre
                 dtracePost/dt = - tracePost/ taupre'''

parameter_iSTDPback="""
taupre = 10                 :projection
aPlus = 10.0*10**(-6)       :projection
Offset = 40.0 *10**(-2)     :projection
hMin=0.0                    :projection
hMax =0.7                   :projection
"""
inhibBackSynapse = Synapse(parameters = parameter_iSTDPback,
                    equations = equatInhib,
                    pre_spike ='''
                         g_target +=w
                         w+= aPlus * (tracePost - Offset)
                         tracePre  += 1 ''',
                    post_spike='''
                         w+=aPlus * (tracePre)
                         tracePost += 1''')
                         
#------------------- iSTDP Synapse for Lateral Inhibitory ----------------------------#

parameter_iSTDPlat="""
taupre = 10                 :projection
aPlus = 10.0*10**(-6)       :projection
Offset = 60.0*10**(-2)      :projection
hMin=0.0                    :projection
hMax =0.5                   :projection
"""
inhibLatSynapse = Synapse(parameters = parameter_iSTDPlat,
                    equations = equatInhib,
                    pre_spike ='''
                         g_target +=w
                         w+= aPlus * (tracePost - Offset)
                         tracePre  += 1 ''',
                    post_spike='''
                         w+=aPlus * (tracePre)
                         tracePost += 1''')

#-----------------------population defintions-----------------------------------#
popInput = PoissonPopulation(geometry=(patchsize,patchsize,2),rates=50.0)
popLGN = Population(geometry=(patchsize,patchsize,2),neuron=spkNeurLGN )
popV1 = Population(geometry=n_exN,neuron=spkNeurV1, name="V1")
popInhibit = Population(geometry=n_inN, neuron = spkNeurV1)
#-----------------------projection definitions----------------------------------
#projPreLayer_PostLayer
projInput_LGN = Projection(
    pre = popInput,
    post = popLGN,
    target = 'exc',
    synapse = inputSynapse
).connect_one_to_one(weights = 30.0)

projLGN_V1 = Projection(
    pre=popLGN, 
    post=popV1, 
    target='Exc',
    synapse=ffSyn
).connect_all_to_all(weights = Normal(0.015,2.0))

projLGN_Inhib = Projection(
    pre = popLGN,
    post= popInhibit,
    target='Exc',
    synapse=ff2Syn
).connect_all_to_all(weights = Uniform(0.0175,2.15))

projV1_Inhib = Projection(
    pre = popV1,
    post = popInhibit,
    target = 'Exc',
    synapse = InhibSyn
).connect_all_to_all(weights = Uniform(0.0175,0.25))

projInhib_V1 = Projection(
    pre = popInhibit,
    post= popV1,
    target = 'Inh',
    synapse = inhibBackSynapse
).connect_all_to_all(weights = 0.0)

projInhib_Lat = Projection(
    pre = popInhibit,
    post = popInhibit,
    target = 'Inh',
    synapse = inhibLatSynapse
).connect_all_to_all(weights = 0.0)

#----------------------------further functions---------------------------------
#-------------------------------------------------------------------------------
def shuffleWeights(wMat):
    # load, reshape and shuffle the weights to break the specificity
    post,pre = np.shape(wMat)
    arr = np.reshape(wMat,post*pre)
    np.random.shuffle(arr)
    wMat = np.reshape(wMat,(post,pre))
    return(wMat)
#-------------------------------------------------------------------------------
def loadWeights():
    w = np.loadtxt('./input/InhibW.txt')
    projLGN_Inhib.w = shuffleWeights(w)
    w = np.loadtxt('./input/V1toIN.txt')
    projV1_Inhib.w = shuffleWeights(w)
    w = np.loadtxt('./input/INtoV1.txt')
    projInhib_V1.w = shuffleWeights(w)
    w = np.loadtxt('./input/INLat.txt')
    projInhib_Lat.w = shuffleWeights(w)
#-------------------------------------------------------------------------------
def createDir():
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output/V1toIN'):
        os.mkdir('output/V1toIN')
    if not os.path.exists('output/exitatory'):
        os.mkdir('output/exitatory')
    if not os.path.exists('output/inhibitory'):
        os.mkdir('output/inhibitory')
    if not os.path.exists('output/V1Layer'):
        os.mkdir('output/V1Layer')
    if not os.path.exists('output/InhibitLayer'):
        os.mkdir('output/InhibitLayer')
#------------------------------------------------------------------------------
def normWeights():
    #print('Norm the Weights!')
    weights= projLGN_V1.w
    for i in range(n_exN):
        onoff  = np.reshape(weights[i],(patchsize,patchsize,2))
        onNorm = np.sqrt(np.sum(onoff[:,:,0]**2))
        offNorm= np.sqrt(np.sum(onoff[:,:,1]**2))
        onoff[:,:,0]*=offNorm/onNorm
        weights[i] = np.reshape(onoff,patchsize*patchsize*2)
    projLGN_V1.w = weights
    weights = projLGN_Inhib.w
    for i in range(n_inN):
        onoff  = np.reshape(weights[i],(patchsize,patchsize,2))
        onNorm = np.sqrt(np.sum(onoff[:,:,0]**2))
        offNorm= np.sqrt(np.sum(onoff[:,:,1]**2))
        onoff[:,:,0]*=offNorm/onNorm
        weights[i] = np.reshape(onoff,patchsize*patchsize*2)
    projLGN_Inhib.w = weights
#-------------------------------------------------------------------------------
def saveValues(values,valName,name):
    if (name == 'Exi'):
        np.save('output/V1Layer/'+valName,values)
    if (name == 'Inhib'):
        np.save('output/InhibitLayer/'+valName,values)
#-------------------------------------------------------------------------------
def saveWeights(nbr =0):
    np.savetxt('output/exitatory/V1weight_%i.txt'%(nbr)  ,projLGN_V1.w)
    np.savetxt('output/exitatory/InhibW_%i.txt'%(nbr), projLGN_Inhib.w)
    np.savetxt('output/V1toIN/V1toIN_%i.txt'%(nbr)  ,projV1_Inhib.w)
    np.savetxt('output/inhibitory/INtoV1_%i.txt'%(nbr) ,projInhib_V1.w)
    np.savetxt('output/inhibitory/INLat_%i.txt'%(nbr) ,projInhib_Lat.w)
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
def getInput(images,patchsize):
    pictNbr = np.random.randint(0,10)
    length,width = images.shape[0:2]
    xPos = np.random.randint(0,length-patchsize)
    yPos = np.random.randint(0,width-patchsize)
    inputPatch = images[xPos:xPos+patchsize,yPos:yPos+patchsize,:,pictNbr]
    maxVal = np.max(images[:,:,:,pictNbr])
    return(inputPatch,maxVal)
#------------------------------------------------------------------------------
def setInputPatch(images,patchsize):
    inputPatch,maxVal = getInput(images,patchsize)
    if np.random.rand() <0.5:
        inputPatch=np.fliplr(inputPatch)
    if np.random.rand()<0.5:
        inputPatch=np.flipud(inputPatch)
    popInput.rates = inputPatch/maxVal *125.0
#------------------------------main function------------------------------------
def run():

    createDir()

    matData = sio.loadmat('./input/IMAGES.mat')
    images = preprocessData(matData)

    compile()
    #loadWeights() # load weights of a previous simulation
    # de-/activate the synaptic plasticity
    projLGN_Inhib.plasticity = True
    projV1_Inhib.plasticity = True
    projInhib_V1.plasticity = True
    projInhib_Lat.plasticity = True
    #------- neuron Monitors --------#
    #------- neuron Monitors --------#
    V1MonP = Monitor(popV1,['vm','vmean','umeanLTD','umeanLTP'],period=duration)
    InhibMonP = Monitor(popInhibit,['vm','vmean'],period=duration)
    V1Mon = Monitor(popV1,['g_Exc','g_Inh','spike'])
    InhibMon=Monitor(popInhibit,['g_Exc','g_Inh','spike'])
    #--------synapse Monitors--------#
    dendriteFF = projLGN_V1.dendrite(0)
    ffWMon = Monitor(dendriteFF,['w','ltdTerm','ltpTerm'],period=duration)
    dendriteFFI = projLGN_Inhib.dendrite(0)
    ffIMon= Monitor(dendriteFFI,'w',period=duration)
    dendriteExIn = projV1_Inhib.dendrite(0)
    exInMon = Monitor(dendriteExIn,'w',period=duration)
    dendriteInEx = projInhib_V1.dendrite(0)
    inExMon = Monitor(dendriteInEx,'w',period=duration)
    dendriteInLat = projInhib_Lat.dendrite(0)
    inLatMon = Monitor(dendriteInLat,'w',period=duration)

    #------Objects to save different variables--------#
    rec_frEx = np.zeros((nbrOfPatches,n_exN))
    rec_V1_gExc=np.zeros((nbrOfPatches,n_exN))
    rec_V1_gInh=np.zeros((nbrOfPatches,n_exN))
    rec_frInh= np.zeros((nbrOfPatches,n_inN))
    rec_Inhib_gExc=np.zeros((nbrOfPatches,n_inN))
    rec_Inhib_gInh=np.zeros((nbrOfPatches,n_inN))

    t1 = time() # start time counter to measure the computational time
    print('Start simulation')

    for i in range(nbrOfPatches):
        setInputPatch(images,patchsize)
        simulate(duration)       
        if ((i*duration)%20000) == 0: # norm weights every 20s as mentioned in Clopath et al. (2010)
            normWeights() 
        spikesEx = V1Mon.get('spike')
        gExcEx = V1Mon.get('g_Exc')
        gInhEx = V1Mon.get('g_Inh')
        spikesInh = InhibMon.get('spike')
        gExcInh = InhibMon.get('g_Exc')
        gInhInh = InhibMon.get('g_Inh')
        for j in range(n_exN):
            rec_V1_gExc[i,j] = np.mean(gExcEx[:,j])
            rec_V1_gInh[i,j] = np.mean(gInhEx[:,j])
            rateEx = len(spikesEx[j])*1000/duration
            rec_frEx[i,j] = rateEx
            if (j < (n_inN)):
                rec_Inhib_gExc[i,j]=np.mean(gExcInh[:,j])
                rec_Inhib_gInh[i,j]=np.mean(gInhInh[:,j])
                rateInh = len(spikesInh[j])*1000/duration
                rec_frInh[i,j] = rateInh         
        if((i%(nbrOfPatches/10)) == 0):
            print("Round %i of %i" %(i,nbrOfPatches))
            saveWeights(i)     
      
    t2 = time()

    saveWeights(nbrOfPatches)


    #------get recorded data---------#
    
    ffW = ffWMon.get('w')
    ffLTD = ffWMon.get('ltdTerm')
    ffLTP = ffWMon.get('ltpTerm')
    ffI = ffIMon.get('w')
    exInW = exInMon.get('w')
    inExW = inExMon.get('w')
    inLatW= inLatMon.get('w')

    l2VM = V1MonP.get('vm')
    l2vMean =  V1MonP.get('vmean')
    l2umeanLTD = V1MonP.get('umeanLTD')
    l2umeanLTP = V1MonP.get('umeanLTP')
    iLvMean =InhibMonP.get('vmean')
    iLVM = InhibMonP.get('vm')


    #--------print Time difference-----------#
    print("time of simulation= "+str((duration*nbrOfPatches)/1000)+" s")
    print("time of calculation= "+str(t2-t1)+" s")
    print("factor= "+str((t2-t1)/(duration*nbrOfPatches/1000)))
    

  #----------------plot output---------------#

    for i in range(1):
        fig = mp.pyplot.figure()
        fig.add_subplot(4,1,1) 
        mp.pyplot.plot(l2umeanLTD[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        mp.pyplot.plot(l2umeanLTD[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        mp.pyplot.plot(l2umeanLTD[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        mp.pyplot.plot(l2umeanLTD[:,3+(4*i)])
        mp.pyplot.savefig('output/V1Layer/l2umeanLTD_'+str(i)+'.png')

    for i in range(1):
        fig = mp.pyplot.figure()
        fig.add_subplot(4,1,1) 
        mp.pyplot.plot(l2umeanLTP[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        mp.pyplot.plot(l2umeanLTP[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        mp.pyplot.plot(l2umeanLTP[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        mp.pyplot.plot(l2umeanLTP[:,3+(4*i)])
        mp.pyplot.savefig('output/V1Layer/l2umeanLTP_'+str(i)+'.png')


    for i in range(1):
        fig = mp.pyplot.figure()
        fig.add_subplot(4,1,1) 
        mp.pyplot.plot(rec_frEx[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        mp.pyplot.plot(rec_frEx[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        mp.pyplot.plot(rec_frEx[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        mp.pyplot.plot(rec_frEx[:,3+(4*i)])
        mp.pyplot.savefig('output/V1Layer/frEx_'+str(i)+'.png')


    plt.plotgEx(rec_V1_gExc,'Exi')
    plt.plotgEx(rec_Inhib_gExc,'Inhib')
    plt.plotgInh(rec_V1_gInh,'Exi')
    plt.plotgInh(rec_Inhib_gInh,'Inhib')
    mp.pyplot.close('all')

    plt.plotVM(l2VM,'Exi')
    plt.plotVM(iLVM,'Inhib')
    plt.plotVMean(l2vMean,'Exi')#
    plt.plotVMean(iLvMean,'Inhib')
    mp.pyplot.close('all')

    mp.pyplot.figure()
    mp.pyplot.plot(np.mean(rec_V1_gExc,axis=1),label='gExc')
    mp.pyplot.plot(-1*np.mean(rec_V1_gInh,axis=1),label='-gInh')
    mp.pyplot.plot(np.mean(rec_V1_gExc,axis=1)-np.mean(rec_V1_gInh,axis=1),label= 'gExc-gInh')
    mp.pyplot.legend()
    mp.pyplot.savefig('output/gExc_gInh.png')

    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(ffW,axis=1))
    mp.pyplot.savefig('output/ffWMean.png')

    print("finish")
#------------------------------------------------------------------------------------
if __name__ == "__main__":
    if os.path.isfile('./input/IMAGES.mat'):
        run()
    else:
        print("""No IMAGES.mat found, please download the file from:
        https://www.rctn.org/bruno/sparsenet/IMAGES.mat
        and put in the *input* directory""")
