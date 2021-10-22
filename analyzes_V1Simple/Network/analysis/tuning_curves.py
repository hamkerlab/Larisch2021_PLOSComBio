import matplotlib.pyplot as plt
import numpy as np
import os.path

def estimateOrientationBandwidth(relativTCExc,minAngle,maxAngle,path):
    # algorithm to estimate the OBW:
    # get the index of the first point, what is over 1/sqrt(2)
    # get the index of the first pont, what is below 1/sqrt(2)
    # calculate the mean of both to get an aproximated value for the OBW
    # because not all TCs are symmetrical 
    # -> calculate OBW for the left and the right edge of the TC and take the mean 

    nbrOfNeurons,nbrOfSteps = np.shape(relativTCExc)

    relativTC_new = np.copy(relativTCExc)

    orientations = np.linspace(minAngle,maxAngle,nbrOfSteps+1)
    orienSteps = np.abs(orientations[1] - orientations[0])
    halfO = np.zeros(nbrOfNeurons)
    for i in range(nbrOfNeurons):
        # test: substract min activity to compensate the offset in the activity (!)
        #minFr = np.min(relativTC_new[i])
        #relativTC_new[i] -= minFr

        # test: normalize activity between one and zero (!)
        maxFr = np.max(relativTC_new[i])
        relativTC_new[i] /= maxFr


        maxFr = np.max(relativTC_new[i])
        idxMax= np.where(relativTC_new[i] == maxFr)[0]
        idxMax = (int(np.mean(idxMax)))        
        idxMaxHalf = np.asarray(np.where(relativTC_new[i] >= (maxFr/np.sqrt(2.0))))[0]
        idxMinHalf = np.asarray(np.where(relativTC_new[i] <= (maxFr/np.sqrt(2.0))))[0]
        
        # TC with high frequency and periodic asymptote, the boarders can be again over 1/sqrt(2)
        # look for the point under 1/sqrt(2) what is nearest on the maximum, for the left edge!
        diff_Idx = idxMax - idxMinHalf

        if (len(diff_Idx[diff_Idx>0]) >0): # check if a left edge exist
            # if exist, get the index
            idx = list(diff_Idx).index(min(diff_Idx[diff_Idx>0]))
            idxMinHalfL = idxMinHalf[idx]
            idxMaxHalfL = idxMinHalfL+1
        else:
            # if not, take the first index as index
            idxMinHalfL = 0
            idxMaxHalfL = idxMinHalfL+1

        # look for the point under 1/sqrt(2) what is nearest on the maximum, for the right edge!

        if (len(diff_Idx[diff_Idx<0]) >0): # check if a right edge exist
            # if, get the index
            idx = list(diff_Idx).index(max(diff_Idx[diff_Idx<0]))
            idxMinHalfR = idxMinHalf[idx]
            idxMaxHalfR = idxMinHalfR-1
        else:
            # if not, take the last index as index
            idxMinHalfR = len(relativTC_new[i])-1
            idxMaxHalfR = idxMinHalfR-1

        maxHalfL = (np.abs(idxMaxHalfL - idxMax)) * orienSteps # upper OBW for the left edge
        minHalfL = (np.abs(idxMinHalfL - idxMax)) * orienSteps # lower OBW for the left edge
        
        maxHalfR = (np.abs(idxMaxHalfR - idxMax)) * orienSteps # upper OBW for the right edge
        minHalfR = (np.abs(idxMinHalfR - idxMax)) * orienSteps # lower OBW for the right edge

        obwL = (maxHalfL+minHalfL)/2.
        obwR = (maxHalfR+minHalfR)/2.

        halfO[i] = (obwL+obwR)/2.
    return(halfO)

#--------------------------------------------------------------------------------
def plotOrientationBandwith(halfO,path,matter):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(halfO,'o')
    plt.xlabel('neuron index')
    plt.ylabel('orientation bandwith [degree]')
    ax.annotate('mean: %f'%np.mean(halfO),xy=(1, 2), xytext=(140, np.max(halfO)+5.0))
    plt.savefig('./Output/TC_'+path+'/BW_'+matter+'.png',bbox_inches='tight', pad_inches = 0.1)

    hist = np.histogram(halfO,9)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(halfO,color='lightgrey',linewidth=2.0,bins=np.arange(0.0,60.0,4))
    plt.axvline(x=np.mean(halfO), color = 'k',ls='dashed',linewidth=5.0)
    plt.xlabel('orientation bandwith ')
    plt.ylabel('number of neurons')
    plt.title('mean: %f'%np.mean(halfO))
    plt.savefig('./Output/TC_'+path+'/BW_hist_'+matter+'.png',bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def startAnalyse_Sinus():
    print('Start analysis sinus gratings')
    #---- tuning curves over sinus gratings ---#
    tcExc = np.load('./work/TuningCurves_sinus_Exc.npy')
    tCExcSinRelativ = np.load('./work/TuningCurvesRelativ_sinus_Exc.npy')
    relativTC_exc_gE = np.load('./work/TuningCurvesRelativ_sinus_Exc_gE.npy')
    relativTC_exc_gI = np.load('./work/TuningCurvesRelativ_sinus_Exc_gI.npy')

    paramTC_exc = np.load('./work/TuningCurves_sinus_Exc_parameters.npy')
    orientations = np.load('./work/TuningCurve_sinus_orientation.npy')

    nbrCells,nbrPatches,nbrContrast,n_trials = np.shape(tCExcSinRelativ)

    tcExc = np.mean(tcExc,axis=3)
    tCExcSinRelativ = np.mean(tCExcSinRelativ,axis=3)
    relativTC_exc_gE = np.mean(relativTC_exc_gE,axis=3)
    relativTC_exc_gI = np.mean(relativTC_exc_gI,axis=3)   


    halfA = int(nbrPatches/2)

    bwEstHalf =np.zeros((nbrCells,nbrContrast))


    cvI= np.zeros((int(nbrCells/4),nbrContrast))
    bwEstHalf_I =np.zeros((int(nbrCells/4),nbrContrast))

    for i in range(nbrContrast):
        tcExcRelat = tCExcSinRelativ[:,int(halfA/2):(nbrPatches-int(halfA/2)),i] # calculate the OBW on the relative TC with -90 + 90 degree  
        minAngle,maxAngle = [-90.0,90.0]
        bwEstHalf[:,i] = estimateOrientationBandwidth(tcExcRelat,minAngle,maxAngle,'sinus')        
       
    np.save('./work/OrientBandwithEst_Half_Sinus',bwEstHalf)


    contLVL = 5

    sortTC_gExc = np.zeros((nbrCells,nbrPatches)) 
    sortTC_gInh = np.zeros((nbrCells,nbrPatches))
    for i in range(nbrCells):  
        idx = np.argsort(relativTC_exc_gE[i,:,contLVL])
        sortTC_gExc[i] = relativTC_exc_gE[i,idx,contLVL]
        sortTC_gInh[i] = relativTC_exc_gI[i,idx,contLVL]

    shift = 90.0
    plotOrientationBandwith(bwEstHalf[:,contLVL],'sinus','est_Pi')

    plt.figure()
    plt.plot(np.mean(bwEstHalf,axis=0),'-o')
    plt.xlabel('contrast level')
    plt.ylabel('mean OBW [°]')
    plt.ylim(10,70)
    plt.savefig('./Output/TC_sinus/MeanbandWith.png')

    ori_idx = np.asarray(paramTC_exc[:,0,contLVL],dtype='int32')
    pref_or = orientations[ori_idx]/np.pi * 180.0

    plt.figure()
    plt.hist(pref_or)
    plt.ylabel('# of fitted neurons')
    plt.xlabel('Orientation [degrees]')
    plt.savefig('./Output/TC_sinus/orientation_hist.png')


    ## Calculate orientation diversity index
    #1. Create a distribition over the observed pref. orientations 
    n_cells = np.shape(pref_or)
    n_bins = 24
    n_ele_real,bins = np.histogram(pref_or,n_bins) 
    #2. Create an idealized distribution     
    n_ele_idealized = (np.ones(n_bins)*n_cells/n_bins)/n_cells
    n_ele_real = n_ele_real/n_cells
    #3. Calcualte the ODI
    div = np.exp(np.nansum( n_ele_real*(np.log(n_ele_real/n_ele_idealized))))
    print(f"ODI = {div}")
    

#------------------------------------------------------------------------------
def startAnalyse_contrast():
    rel_VM = np.load('./work/TuningCurvesRelativ_sinus_Exc_VM.npy')
    rel_GE = np.load('./work/TuningCurvesRelativ_sinus_Exc_gE.npy')
    rel_GI = np.load('./work/TuningCurvesRelativ_sinus_Exc_gI.npy')
    rel_SC = np.load('./work/TuningCurvesRelativ_sinus_Exc.npy')

    rel_SC /=8 # devide by 8 to get the number of spikes not the firing rate (128 ms = 1/8 second)

    print(np.shape(rel_VM))

    nb_cells,nb_or,nb_contrast,repeats = np.shape(rel_VM)

    low_cont = 1
    high_cont = 6

    ref_cont = 5

    vt_rest = -50.4
    n_idx = 4
    i = 0

    # mean and std over the repetitions (!!) not over cells
    relVM_oT_oR = np.mean(rel_VM,axis=3)
    relVM_oT_oR_std = np.std(rel_VM,axis=3)

    relGE_oT_oR = np.mean(rel_GE,axis=3)
    relGI_oT_oR = np.mean(rel_GI,axis=3)
    relSC_oR = np.mean(rel_SC,axis=3)


    x = np.linspace(0,nb_or-1,nb_or)


    relVM_oT_oR_oC = np.mean(relVM_oT_oR,axis=0)
    relVM_oT_oR_std_median_oC = np.median(relVM_oT_oR_std,axis=0) # median over all STD's over the Repitions (median over cells)
    relVM_oT_oR_std_mean_oC = np.mean(relVM_oT_oR_std,axis=0) # mean


    relGE_oT_oR_oC = np.mean(relGE_oT_oR,axis=0)
    relGI_oT_oR_oC = np.mean(relGI_oT_oR,axis=0)
    relSC_oR_oC = np.mean(relSC_oR,axis=0)


    markers=['-o','--','-','-.','-^']
    fill = ['none','full','full','full','full']
    color='seagreen'
    relGE_oT_oR_sort = relGE_oT_oR
    relSC_oR_sort = relSC_oR
    for i in range(nb_cells):
        idxEx = np.argsort(relGE_oT_oR[i,:,ref_cont])
        relGE_oT_oR_sort[i,:,:] = relGE_oT_oR[i,idxEx]
        relSC_oR_sort[i,:,:] =relSC_oR[i,idxEx] 


    # plot mean VM, exc input, inh input, and more
    std_relVM_oT_oR_oC = np.std(relVM_oT_oR,axis=0,ddof=1)


    print('---')
    print(np.shape(relVM_oT_oR_std))
    mean_relVM_oT_oR_std_oC = np.mean(relVM_oT_oR_std,axis=0) # mean over the cells of the std of repitions


    markers=['-.','--','-','-o',':']

    plt.figure(figsize=(20,12))

    plt.subplot(231)
    for i in range(low_cont,6):
        plt.plot(relSC_oR_oC[:,i],markers[i-1],label='contrast LVL: %i'%i)
    plt.legend()
    plt.ylabel('spike count')
    plt.xlabel('orientation')
    plt.xticks(np.linspace(0,nb_or,5),np.linspace(-180,180,5))

    plt.subplot(232)
    for i in range(low_cont,6):
        plt.plot(relVM_oT_oR_oC[:,i],markers[i-1],label='contrast LVL: %i'%i)
    plt.legend()
    plt.ylabel('average membrane potential')
    plt.xlabel('orientation')
    plt.xticks(np.linspace(0,nb_or,5),np.linspace(-180,180,5))

    plt.subplot(233)
    for i in range(low_cont,6):
        plt.plot(relVM_oT_oR_std_mean_oC[:,i],markers[i-1],label='contrast LVL: %i'%i)
    plt.legend()
    plt.ylabel('membrane potential STD')
    plt.xlabel('orientation')
    plt.xticks(np.linspace(0,nb_or,5),np.linspace(-180,180,5))

    plt.subplot(234)
    for i in range(low_cont,6):
        plt.plot(relGE_oT_oR_oC[:,i],markers[i-1],label='contrast LVL: %i'%i)
    plt.legend()
    plt.ylabel('excitatory input')
    plt.xlabel('orientation')
    plt.ylim(0.0,15)
    plt.xticks(np.linspace(0,nb_or,5),np.linspace(-180,180,5))

    plt.subplot(235)
    for i in range(low_cont,6):
        plt.plot(relGI_oT_oR_oC[:,i],markers[i-1],label='contrast LVL: %i'%i)
    plt.legend()
    plt.ylabel('inhibitory input')
    plt.xlabel('orientation')
    plt.ylim(0.0,15)
    plt.xticks(np.linspace(0,nb_or,5),np.linspace(-180,180,5))

    plt.subplot(236)
    for i in range(low_cont,6):
        plt.plot(relGE_oT_oR_oC[:,i]- relGI_oT_oR_oC[:,i],markers[i-1],label='contrast LVL: %i'%i)
    plt.legend()
    plt.ylabel('excitatory - inhibitory input')
    plt.xlabel('orientation')
    plt.xticks(np.linspace(0,nb_or,5),np.linspace(-180,180,5))


    plt.savefig('./Output/TC_sinus/excitatory/meanVM_OT_OR_allC.png',bbox_inches='tight', pad_inches = 0.1,dpi=300)    
    
    ##############

    color='seagreen' # depending on model 'tomato', 'steelblue', 'yellow'
    color_high='springgreen'
    color_low = 'darkgreen'

    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)

    s_legend = 20
    s_font = 25

    nb_cells,nb_or,nb_contrast,repeats = np.shape(rel_VM)

    maxInput = 100
    step = maxInput/7.
    total = step+(4*step)

    low_cont = 1
    high_cont = 6

    ref_cont = 5

    vt_rest = -50.4


    markers=['-o','--','-','-.','-^']
    mfc = ['white',color,color,color,color]

    shift = 90.0
    anglStep = 360.0/nb_or
    shiftStep = int(shift/anglStep)

    low_label = int((step + (low_cont*step))/total *100)
    high_label = int((step + (5*step))/total *100)

    ### plot mean VM with mean sigma (per neuron) for high and low contrast ##
    # mean and std over the repetitions (!!) not over cells
    relVM_oT_oR = np.mean(rel_VM,axis=3)
    relVM_oT_oR_std = np.std(rel_VM,axis=3)


    relVM_oT_oR_oC = np.mean(relVM_oT_oR,axis=0)
    relVM_oT_oR_std_mean_oC = np.mean(relVM_oT_oR_std,axis=0) # mean
    mean_relVM_oT_oR_std_oC = np.mean(relVM_oT_oR_std,axis=0) # mean over the cells of the std of repitions
    percU_oT_oR_oC = np.percentile(relVM_oT_oR_oC,95) # 95th percentile
    percL_oT_oR_oC = np.percentile(relVM_oT_oR_oC,5) # 5th percentile


    roll_VM_low = np.roll(relVM_oT_oR_oC[:,low_cont],len(relVM_oT_oR_oC[:,low_cont])-shiftStep)
    roll_VM_low_std = np.roll(mean_relVM_oT_oR_std_oC[:,low_cont],len(mean_relVM_oT_oR_std_oC[:,low_cont])-shiftStep)

    roll_VM_high = np.roll(relVM_oT_oR_oC[:,5],len(relVM_oT_oR_oC[:,5])-shiftStep)
    roll_VM_high_std = np.roll(mean_relVM_oT_oR_std_oC[:,5],len(mean_relVM_oT_oR_std_oC[:,5])-shiftStep)

    x_roll = np.linspace(0,len(relVM_oT_oR_oC[:,low_cont])-1,len(relVM_oT_oR_oC[:,low_cont]))

    plt.figure(figsize=(7,7))
#    contLevel = (step + (i*step))/total *100


    plt.plot(x_roll,roll_VM_low,markers[low_cont-1],color=color,label=str(low_label)+'%',linewidth=2.5,markersize=9,mfc=mfc[low_cont-1])#,label='contrast LVL: %i'%i)
    plt.fill_between(x_roll,roll_VM_low - roll_VM_low_std,roll_VM_low + roll_VM_low_std,color=color_low,alpha=0.3 )

    plt.plot(x_roll,roll_VM_high,markers[5-1],color=color,label=str(high_label)+'%',linewidth=2.5,markersize=9,mfc=mfc[5-1])#,label='contrast LVL: %i'%i)
    plt.fill_between(x_roll,roll_VM_high - roll_VM_high_std,roll_VM_high + roll_VM_high_std,color=color_high,alpha=0.3 )


    plt.hlines(vt_rest,xmin=0,xmax=nb_or,colors='black', linestyles='-.',label='spike threshold',linewidth=2.5)
    plt.legend(prop={'size': s_legend})
    plt.ylabel(r'Average membrane potential $[mV]$',fontsize=s_font)
    plt.xlabel(r'Orientation $[\degree]$',fontsize=s_font)
    plt.ylim(-70,-5)
    plt.xticks(np.linspace(0,nb_or,5),np.linspace(-90,270,5))
    plt.savefig('./Output/TC_sinus/excitatory/meanVM_OT_OR_vm_stdOR_MembrDiff_extra.png',bbox_inches='tight', pad_inches = 0.1,dpi=300)


    ### plot mean Spike Count with mean sigma (per neuron) for high and low contrast ##
    # mean and std over the repetitions (!!) not over cells
    rel_SC_oT_oR = np.mean(rel_SC,axis=3)
    rel_SC_oT_oR_std = np.std(rel_SC,axis=3)


    rel_SC_oT_oR_oC = np.mean(rel_SC_oT_oR,axis=0)
    rel_SC_oT_oR_std_mean_oC = np.mean(rel_SC_oT_oR_std,axis=0) # mean
    mean_rel_SC_oT_oR_std_oC = np.mean(rel_SC_oT_oR_std,axis=0) # mean over the cells of the std of repitions

    roll_SC_low = np.roll(rel_SC_oT_oR_oC[:,low_cont],len(rel_SC_oT_oR_oC[:,low_cont])-shiftStep)
    roll_SC_low_std = np.roll(mean_rel_SC_oT_oR_std_oC[:,low_cont],len(mean_rel_SC_oT_oR_std_oC[:,low_cont])-shiftStep)

    roll_SC_high = np.roll(rel_SC_oT_oR_oC[:,5],len(rel_SC_oT_oR_oC[:,5])-shiftStep)
    roll_SC_high_std = np.roll(mean_rel_SC_oT_oR_std_oC[:,5],len(mean_rel_SC_oT_oR_std_oC[:,5])-shiftStep)

    x_roll = np.linspace(0,len(rel_SC_oT_oR_oC[:,low_cont])-1,len(rel_SC_oT_oR_oC[:,low_cont]))

    plt.figure(figsize=(7,7))
#    contLevel = (step + (i*step))/total *100

    plt.plot(x_roll,roll_SC_low,markers[low_cont-1],color=color,label=str(low_label)+'%',linewidth=2.5,markersize=9,mfc=mfc[low_cont-1])#,label='contrast LVL: %i'%i)
    plt.fill_between(x_roll,roll_SC_low - roll_SC_low_std,roll_SC_low + roll_SC_low_std,color=color_low,alpha=0.3 )

    plt.plot(x_roll,roll_SC_high,markers[5-1],color=color,label=str(high_label)+'%',linewidth=2.5,markersize=9,mfc=mfc[5-1])#,label='contrast LVL: %i'%i)
    plt.fill_between(x_roll,roll_SC_high - roll_SC_high_std,roll_SC_high + roll_SC_high_std,color=color_high,alpha=0.3 )
 
    plt.legend(prop={'size': s_legend})
    plt.ylabel('Spike count',fontsize=s_font)
    plt.xlabel(r'Orientation $[\degree]$',fontsize=s_font)
    plt.ylim(0,30)
    plt.xticks(np.linspace(0,nb_or,5),np.linspace(-90,270,5))
    plt.savefig('./Output/TC_sinus/excitatory/meanSC_OT_OR_SC_stdOR_SCDiff_extra.png',bbox_inches='tight', pad_inches = 0.1,dpi=300)


    ### plot mean difference between excitatory and inhibitory current with mean sigma (per neuron) for high and low contrast ##
    # mean and std over the repetitions (!!) not over cells

    rel_GD = rel_GE - rel_GI # differencet between relative gE and gI

    rel_GD_oT_oR = np.mean(rel_GD,axis=3)
    rel_GD_oT_oR_std = np.std(rel_GD,axis=3)

    rel_GD_oT_oR_oC = np.mean(rel_GD_oT_oR,axis=0)
    rel_GD_oT_oR_std_mean_oC = np.mean(rel_GD_oT_oR_std,axis=0) # mean
    mean_rel_GD_oT_oR_std_oC = np.mean(rel_GD_oT_oR_std,axis=0) # mean over the cells of the std of repitions

    roll_GD_low = np.roll(rel_GD_oT_oR_oC[:,low_cont],len(rel_GD_oT_oR_oC[:,low_cont])-shiftStep)
    roll_GD_low_std = np.roll(mean_rel_GD_oT_oR_std_oC[:,low_cont],len(mean_rel_GD_oT_oR_std_oC[:,low_cont])-shiftStep)

    roll_GD_high = np.roll(rel_GD_oT_oR_oC[:,5],len(rel_GD_oT_oR_oC[:,5])-shiftStep)
    roll_GD_high_std = np.roll(mean_rel_GD_oT_oR_std_oC[:,5],len(mean_rel_GD_oT_oR_std_oC[:,5])-shiftStep)

    x_roll = np.linspace(0,len(rel_GD_oT_oR_oC[:,low_cont])-1,len(rel_GD_oT_oR_oC[:,low_cont]))

    plt.figure(figsize=(7,7))
#    contLevel = (step + (i*step))/total *100

    plt.plot(x_roll,roll_GD_low,markers[low_cont-1],color=color,label=str(low_label)+'%',linewidth=2.5,markersize=9,mfc=mfc[low_cont-1])#,label='contrast LVL: %i'%i)
    plt.fill_between(x_roll,roll_GD_low - roll_GD_low_std,roll_GD_low + roll_GD_low_std,color=color_low,alpha=0.3 )

    plt.plot(x_roll,roll_GD_high,markers[5-1],color=color,label=str(high_label)+'%',linewidth=2.5,markersize=9,mfc=mfc[5-1])#,label='contrast LVL: %i'%i)
    plt.fill_between(x_roll,roll_GD_high - roll_GD_high_std,roll_GD_high + roll_GD_high_std,color=color_high,alpha=0.3 )
 
    plt.legend(prop={'size': s_legend})
    plt.ylabel('Sum of excitatory and inhibitory input',fontsize=s_font)
    plt.xlabel(r'Orientation $[\degree]$',fontsize=s_font)
    plt.ylim(-0.5,10)
    plt.xticks(np.linspace(0,nb_or,5),np.linspace(-90,270,5))
    plt.savefig('./Output/TC_sinus/excitatory/meanGD_OT_OR_GD_stdOR_GDDiff_extra.png',bbox_inches='tight', pad_inches = 0.1,dpi=300)


    # spike count as a function of input excitatory current 

    rel_SC_oR = np.mean(rel_SC,axis=3)
    rel_GE_oR = np.mean(rel_GE,axis=3)

    print(np.shape(rel_SC))
    print(np.shape(rel_SC_oR))

    rel_SC_oR = rel_SC_oR[:,0:int(nb_or/2)+1,:] # only use the first half of the TC
    rel_GE_oR = rel_GE_oR[:,0:int(nb_or/2)+1,:]

    rel_SC_oR_sort = np.zeros((nb_cells,int(nb_or/2)+1,nb_contrast))
    rel_GE_oR_sort = np.zeros((nb_cells,int(nb_or/2)+1,nb_contrast))
    #ref_cont
    for i in range(nb_cells):
        argIdx = np.argsort(rel_GE_oR[i,:,ref_cont])
        rel_SC_oR_sort[i] = rel_SC_oR[i,argIdx]
        rel_GE_oR_sort[i] = rel_GE_oR[i,argIdx]

    plt.figure()
    plt.plot(np.mean(rel_GE_oR_sort[:,:,1],axis=0)[0::3],np.mean(rel_SC_oR_sort[:,:,1],axis=0)[0::3],'-s',fillstyle='none',color='seagreen',markersize=10,linewidth=2, label='low contrast')
    plt.plot(np.mean(rel_GE_oR_sort[:,:,5],axis=0)[0::2],np.mean(rel_SC_oR_sort[:,:,5],axis=0)[0::2],'-^',fillstyle='none',color='seagreen',markersize=10,linewidth=2, label='high contrast')
    plt.legend()
    plt.xlabel(r'$g_{Exc} [nA]$ ',fontsize=22)
    plt.ylabel('Spike count',fontsize=22)   
    plt.ylim(0.0,40.0)
    plt.savefig('./Output/TC_sinus/excitatory/TC_contrast_gain.png',dpi=300,bbox_inches='tight')

if __name__ == "__main__":

    if not os.path.exists('./Output/TC_sinus/'):
        os.mkdir('./Output/TC_sinus/')
    if not os.path.exists('./Output/TC_sinus/excitatory/'):
        os.mkdir('./Output/TC_sinus/excitatory/')

    startAnalyse_Sinus()
    startAnalyse_contrast()
