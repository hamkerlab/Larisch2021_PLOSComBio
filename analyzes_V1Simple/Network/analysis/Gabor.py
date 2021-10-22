import numpy as np
import matplotlib.pyplot as plt

# create 2D Gabor 

def makeGaborMatrix(xPos,yPos,amplitude,theta,sigmaX,sigmaY,k,phi,x_c,y_c,offset,patchsize):

    #xPos = index/12
    #yPos = index - 12*xPos
    xPos = (xPos - x_c) / np.float(patchsize)
    yPos = (yPos - y_c) / np.float(patchsize)

    xp = xPos*np.cos(theta)+yPos*np.sin(theta)
    yp = -xPos*np.sin(theta)+yPos*np.cos(theta)
    #xp = xPos*np.cos(np.pi/180.0*theta)+yPos*np.sin(np.pi/180.0*theta)
    #yp = -xPos*np.sin(np.pi/180.0*theta)+yPos*np.cos(np.pi/180.0*theta)
    cosinus = np.cos(2.0*np.pi*xp*k - phi)
    #expotential = np.exp(-1.0/2.0* (xp**2/sigmaX**2 + yp**2/sigmaY**2))
    expotential = np.exp(-(xp**2/(2.0*sigmaX**2)) - (yp**2/(2.0*sigmaY**2)))
    gabor = offset + amplitude*expotential*cosinus
    return(gabor)

def checkBounds(parameters,bounds):
    for i in range(9):
        parameters[i] = bounds[i,0] if (parameters[i] < bounds[i,0]) else parameters[i]
        parameters[i] = bounds[i,1] if (parameters[i] > bounds[i,1]) else parameters[i]
    return(parameters)

def createGabor(parameters,patchsizeX,patchsizeY,bounds):
    # create 2D Gabor
    # parameters -> parameterset for gabor function
    # patchsizeX/Y -> size along X and Y axis
    # bounds -> to check if parameters in correct bounds
    parameters = checkBounds(parameters,bounds)
    amplitude = parameters[0]  
    theta     = parameters[1]
    sigmaX    = parameters[2]
    sigmaY    = parameters[3]
    k         = parameters[4] #period
    phi       = parameters[5] #phase
    x_c       = parameters[6]
    y_c       = parameters[7]
    offset    = parameters[8]
    gaborM = np.zeros((patchsizeX,patchsizeY))
    for x in range(patchsizeX):
        for y in range(patchsizeY):
            gaborM[x,y] = makeGaborMatrix(x,y,amplitude,theta,sigmaX,sigmaY,k,phi,x_c,y_c,offset,patchsizeX)
    return(np.reshape(gaborM,(patchsizeX*patchsizeY)))

def createGaborMatrix(parameters,patchsizeX,patchsizeY):
    amplitude = parameters[0] 
    theta     = parameters[1]
    sigmaX    = parameters[2]
    sigmaY    = parameters[3]
    k         = parameters[4] #period
    phi       = parameters[5] #phase
    x_c       = parameters[6]
    y_c       = parameters[7]
    offset    = parameters[8]
    gaborM = np.zeros((patchsizeX,patchsizeY))
    for x in range(patchsizeX):
        for y in range(patchsizeY):
            gaborM[x,y] = makeGaborMatrix(x,y,amplitude,theta,sigmaX,sigmaY,k,phi,x_c,y_c,offset,patchsizeX)
    return(gaborM)
