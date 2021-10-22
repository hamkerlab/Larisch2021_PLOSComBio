import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.io as sio
import os


# Description !!
#
#
#--------------------------- calculating functions ---------------------------#
def calcIRE_Image(frEx,ffW,patchsize):
    # reconstruction of the input Image,  
    # based on the network weights and the cell activity
    pixelStep = 3
    post,pre = np.shape(ffW)
    numberOfNeurons,nbrOfPatches=np.shape(frEx)
    w = (int(np.sqrt(nbrOfPatches))*pixelStep) + patchsize # * patchsize
    h = w
    ire_Image = np.zeros((w,h,2))
    counter_M = np.ones((w,h,2))
    for x in range(0,w-patchsize,pixelStep):
        for y in range(0,h - patchsize,pixelStep):
            patchActivity=np.zeros((patchsize,patchsize,2))
            xPosMin=0 + (1*x)
            xPosMax=patchsize+ (1*x)
            yPosMin=0+ (1*y)
            yPosMax=patchsize+ (1*y)
            patchIndex = int((y/pixelStep) +(x/pixelStep)*((h-patchsize)/pixelStep))
            #print(patchIndex)
            for neuron in range(numberOfNeurons):
                weight = np.reshape(ffW[neuron,:],(patchsize,patchsize,2))
                patchActivity=patchActivity + (weight* frEx[neuron,patchIndex]) 
            ire_Image[xPosMin:xPosMax,yPosMin:yPosMax,:] = ire_Image[xPosMin:xPosMax,yPosMin:yPosMax,:] + (patchActivity/numberOfNeurons)
            counter_M[xPosMin:xPosMax,yPosMin:yPosMax,:]+= 1#*np.mean(frEx[:,patchIndex])/numberOfNeurons
    ire_Image = ire_Image/(counter_M)
    return(ire_Image)
#------------------------------------------------------------------------------
def calcIRE(input_Image,ire_Image):
    # via NMSE after Spartling(2012)
    # for [0,1] normalized Images

    #old used Error
    errorMatrix =  np.sum((input_Image - ire_Image)**2)/(np.sum(ire_Image**2))
    errorAbs =  np.sum((np.abs(input_Image) - np.abs(ire_Image))**2)/(np.sum(ire_Image**2))

    # not used error alternative
    #w,h = np.shape(ire_Image)
    #ireV = np.reshape(ire_Image, w*h)
    #iptV = np.reshape(input_Image,w*h)
    #errorMatrix =  (np.sum(iptV-ireV)**2)/np.sum((np.abs(iptV) - np.abs(ireV))**2)
    #errorAbs =  0.0
    return(errorMatrix,errorAbs)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def calcIREoverRMS(input_Image,ire_Image):
    h,w=(np.shape(ire_Image))
    n = h*w
    error = np.sqrt(np.sum((input_Image-ire_Image)**2)*1/n )
    return(error)
#------------------------------------------------------------------------------
def calcLGNIRE(frLGN,patchsize):
# reconstruction of the input Image,  
    # based on the network weights and the cell activity
    pixelStep = 3
    nbrCells,nbrPatches=np.shape(frLGN)
    w = (int(np.sqrt(nbrPatches))*pixelStep) + patchsize # * patchsize
    h = w
    ire_Image = np.zeros((w,h,2))
    counter_M = np.ones((w,h,2))
    for x in range(0,w-patchsize,pixelStep):
        for y in range(0,h - patchsize,pixelStep):
            patchActivity=np.zeros((patchsize,patchsize,2))
            xPosMin=0 + (1*x)
            xPosMax=patchsize+ (1*x)
            yPosMin=0+ (1*y)
            yPosMax=patchsize+ (1*y)
            patchIndex = int((y/pixelStep) +(x/pixelStep)*((h-patchsize)/pixelStep))
            patchActivity = np.reshape(frLGN[:,patchIndex],(patchsize,patchsize,2))
            ire_Image[xPosMin:xPosMax,yPosMin:yPosMax,:] = ire_Image[xPosMin:xPosMax,yPosMin:yPosMax,:] + (patchActivity)
            counter_M[xPosMin:xPosMax,yPosMin:yPosMax,:]+= 1#*np.mean(frEx[:,patchIndex])/numberOfNeurons
    ire_Image = ire_Image/(counter_M)
    return(ire_Image)
#------------------------------------------------------------------------------
def startIREAnalysis():
    plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 25)
    plt.rc('ytick',labelsize = 25)

    if not os.path.exists('./Output/IRE'):
        os.mkdir('./Output/IRE')


    #----load necessary datas----# 
    frIREs = np.load('work/IRE_fr.npy')
    patchesIRE = np.load('work/IRE_singlePatches.npy')
    ffW = np.loadtxt('Input_network/V1weight.txt')
    frLGNs = np.load('work/IRE_LGNFR.npy')
    inputImages=np.load('work/IRE_Images.npy')

    patchsizeIRE = np.shape(patchesIRE)[1]
    
    ire = np.zeros(10)
    print('Calculate IRE Image and IRE')
    for i in range(10):
        frIRE = frIREs[:,i,:]
        frLGN = frLGNs[:,i,:]
        inputImage= inputImages[i,:,:,:]

        ireImage = calcIRE_Image(frIRE,ffW,patchsizeIRE)
        np.save('./Output/IRE/IREImage_'+str(i),ireImage)
        ireLGN = calcLGNIRE(frLGN,patchsizeIRE)

        inputImage=inputImage[:,:,0] - inputImage[:,:,1]
        ireImage = ireImage[:,:,0] - ireImage[:,:,1]
        ireLGN = ireLGN[:,:,0]- ireLGN[:,:,1]
        
        imageSize = np.shape(inputImage)[0]

        inputSTD = (inputImage-np.mean(inputImage))/np.std(inputImage)
        lgnSTD = (ireLGN-np.mean(ireLGN))/np.std(ireLGN)
        ireSTD = (ireImage-np.mean(ireImage))/np.std(ireImage)

        inputImageArr = np.reshape(inputImage,imageSize*imageSize)
        ireImageArr =np.reshape(ireImage,imageSize*imageSize)
        ireLGNArr = np.reshape(ireLGN,imageSize*imageSize)

        inputSTDArr = np.reshape(inputSTD,imageSize*imageSize)
        lgnSTDArr = np.reshape(lgnSTD,imageSize*imageSize)
        ireSTDArr = np.reshape(ireSTD,imageSize*imageSize)

        rmsSTDNormLGN = calcIREoverRMS((inputImage-np.mean(inputImage))/np.std(inputImage),(ireLGN-np.mean(ireLGN))/np.std(ireLGN))
        rmsSTDNormLGNV1 = calcIREoverRMS((ireLGN-np.mean(ireLGN))/np.std(ireLGN),(ireImage-np.mean(ireImage))/np.std(ireImage))
        rmsSTDNorm = calcIREoverRMS((inputImage-np.mean(inputImage))/np.std(inputImage),(ireImage-np.mean(ireImage))/np.std(ireImage))

        stat = {'IRE_Input_LGN_STDNorm':rmsSTDNormLGN,
                'IRE_Input_V1_STDNorm':rmsSTDNorm,
                'IRE_LGN_V1_STDNorm':rmsSTDNormLGNV1}
        json.dump(stat,open('./Output/IRE/IRE_'+str(i)+'.txt','w'))
        ire[i] = rmsSTDNorm

    print('Plot Image and IRE')
    
    plt.figure()
    plt.plot(ire,'o')
    plt.title('MeanIRE= '+str(np.round(np.mean(ire),6) ))
    plt.ylabel('IRE')
    plt.xlabel('Image index')
    plt.ylim(ymin=0.5,ymax=1.2)
    plt.savefig('./Output/IRE/IRE_all.png')

    print('Finish with IRE')
#-----------------------------------------------------
if __name__=="__main__":
    startIREAnalysis()
