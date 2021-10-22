import matplotlib.pyplot as plt
import numpy as np
import os.path
#------------------------------------------------------------------------------
def calcTemplateMatch(X,Y):
    tm = 0
    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    if (normX != 0 and normY !=0):
        tm = (np.dot(X,Y) / (normX*normY))
    return(tm)
#------------------------------------------------------------------------------
def calcTM_input(inpt):
    print(np.shape(inpt))
    n_patches,w,h,d = np.shape(inpt)
    tm = np.zeros((n_patches,n_patches))
    for i in range(n_patches):
        img_i = inpt[i,:,:,0] - inpt[i,:,:,1] 
        v_inpt_i = np.reshape(img_i,w*h)
        for j in range(n_patches):
            img_j = inpt[j,:,:,0] - inpt[j,:,:,1]
            v_inpt_j = np.reshape(img_j,w*h)
            tm[i,j] = calcTemplateMatch(v_inpt_i,v_inpt_j)
    np.save('./work/fluctuation_tm_input',tm)
    return(tm) 
#------------------------------------------------------------------------------
def estimateD_prime(spkCV1,inpt):
    print(np.shape(spkCV1))
    n_cells,n_patches,n_trails =np.shape(spkCV1)
    
    tmInpt = calcTM_input(inpt)
    #tmInpt = np.load('./work/fluctuation_tm_input.npy')

    n_patches = n_patches-1
    d_prime = np.zeros(n_patches)
    sort_idx = np.argsort(tmInpt[0,:]*-1)#sort index decreasing over tm
    for i_patch in range(1,n_patches+1):
        resp_proj1 = np.zeros(n_trails)
        resp_proj2 = np.zeros(n_trails)

        #mean over the acticity of the choosen patches
        mu1 = np.mean(spkCV1[:,0,:],1)
        mu2 = np.mean(spkCV1[:,sort_idx[i_patch],:],1)
    
        diffvec = mu1 - mu2

        for i_trails in range(n_trails):
            resp_proj1[i_trails] = np.dot(spkCV1[:,0,i_trails],diffvec)
            resp_proj2[i_trails] = np.dot(spkCV1[:,sort_idx[i_patch],i_trails],diffvec)

        mu_proj1 = np.mean(resp_proj1)
        var_proj1 = np.var(resp_proj1)   

        mu_proj2 = np.mean(resp_proj2)
        var_proj2 = np.var(resp_proj2)

        d_prime[i_patch-1] = np.abs(mu_proj1- mu_proj2) / np.sqrt( 0.5*(var_proj1 + var_proj2) )
    return(d_prime)
#------------------------------------------------------------------------------
def startAnalysis():

    spkCV1 = np.load('./work/fluctuation_frExc.npy')
    inpt = np.load('./work/fluctuation_Input.npy')

    d_prime = estimateD_prime(spkCV1,inpt)

    plt.figure()
    plt.plot(d_prime,'o')
    plt.savefig('./Output/d_prime.png')



#------------------------------------------------------------------------------
if __name__ == "__main__":
    startAnalysis()
