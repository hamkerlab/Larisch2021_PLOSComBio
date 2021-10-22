import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

#------------------------------------------------------------------------------
def plotExcToInhibOrientation(matrix,manner,log):
    #plt.rc('font',weight = 'bold')

    plt.figure()
    #---define own colormap---#
    colors = [(0, 0, 0), (0.75,0.6, 0.3), (1, 1, 1)] # black->red->white
    n_bin = 5  # Discretizes the interpolation into bins
    cmap_name = 'my_hot'
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
    if log:
        plt.rc('xtick',labelsize = 12)
        plt.rc('ytick',labelsize = 12)
        meanW = np.mean(matrix)
        maxW = np.max(matrix)
        cM = np.copy(matrix)
        #cM[np.where(cM < maxW*0.9)] = 0.0
        plt.imshow(cM.T**2,cmap=plt.get_cmap('afmhot'),interpolation='none')#,aspect='auto') #afmhot
        plt.xlabel('E cell rank',fontsize=18)
        plt.ylabel('',fontsize=20)
    else:
        plt.rc('xtick',labelsize = 15)
        plt.rc('ytick',labelsize = 15)
        beta=0.5
        plt.imshow(matrix,cmap=plt.get_cmap('afmhot',20),interpolation='none',norm = MidpointNormalize(midpoint=np.max(matrix)*beta))#,aspect='auto')
        plt.ylabel('I cell rank',fontsize=20)
        #plt.xlabel('excitatory neurons',weight='bold',fontsize=20)  
    cbar = plt.colorbar(fraction=0.1, pad=0.01,ticks=[0.0,np.max(matrix)/2,np.max(matrix)])
    cbar.set_label('weight value',fontsize=18)
    cbar.ax.set_yticklabels(['0.0','','max'])
    if manner == 'oMLateralInhib':
        plt.rc('xtick',labelsize = 18)
        plt.rc('ytick',labelsize = 18)
        plt.xlabel('I cell rank',fontsize=23)
        plt.ylabel('I cell rank',fontsize=23)
        cbar.ax.tick_params(labelsize=18)
    plt.savefig('./Output/Orientation_'+manner+'.png',dpi=300,bbox_inches='tight', pad_inches = 0.1)
#------------------------------------------------------------------------------
def calcOrientMatricOverTC(weightsV1_IN,orientations,frExc,frInh):
    contLvL = 4
    print(np.shape(frExc))
    print(np.shape(frInh))
    #frExc = frExc[:,:,:,:]
    #frInh = frInh[:,:,:,:]
    nbrOfExc = np.shape(frExc)[0]
    nbrOfInh = np.shape(frInh)[0]
    matrix = np.zeros((nbrOfInh,nbrOfExc))
    orExc = np.zeros(nbrOfExc)
    orInh = np.zeros(nbrOfInh)
    for i in range(nbrOfExc):
        maxIdxExc = np.where(frExc[i] == np.max(frExc[i]))
        orExc[i] = orientations[maxIdxExc[0][0],maxIdxExc[1][0],maxIdxExc[2][0]]
    for i in range(nbrOfInh):
        maxIdxInh = np.where(frInh[i] == np.max(frInh[i]))
        orInh[i] = orientations[maxIdxInh[0][0],maxIdxInh[1][0],maxIdxInh[2][0]]
    sortIdxExc = np.argsort(orExc)
    sortIdxInh = np.argsort(orInh)
    for i in range(nbrOfInh):
        for j in range(nbrOfExc):
            matrix[i,j] = weightsV1_IN[sortIdxInh[i],sortIdxExc[j]]
    return(matrix,orExc,orInh)
#------------------------------------------------------------------------------
def calcOrientMatrixInhibOverTC(weightsLatIN,orientations,frInh):
    nbrOfInh = np.shape(frInh)[0]
    latW = np.zeros((nbrOfInh,nbrOfInh))
    for i in range(nbrOfInh):
        latW[i,0:i] = weightsLatIN[i,0:i]
        latW[i,i+1:nbrOfInh] = weightsLatIN[i,i:nbrOfInh-1]
    matrix = np.zeros((nbrOfInh,nbrOfInh))
    orInh = np.zeros(nbrOfInh)
    for i in range(nbrOfInh):
        maxIdxInh = np.where(frInh[i] == np.max(frInh[i]))
        orInh[i] = orientations[maxIdxInh[0][0],maxIdxInh[1][0],maxIdxInh[2][0]]
    sortIdxInh = np.argsort(orInh)
    for i in range(nbrOfInh):
        for j in range(nbrOfInh):
            matrix[i,j] = latW[sortIdxInh[i],sortIdxInh[j]]
    return(matrix)

#------------------------------------------------------------------------------
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.45, 1] #[0, 0.45, 1]
        return np.ma.masked_array(np.interp(value, x, y))
#------------------------------------------------------------------------------
def plotOrientationWeights(matrixEI,matrixIE,orExc,orInh,manner):
    plt.close('all')

    #plt.rc('font',weight = 'bold')
    plt.rc('xtick',labelsize = 15)
    plt.rc('ytick',labelsize = 15)

    nullfmt = mp.ticker.NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.15, 0.65
    bottom, height = 0.1, 0.65
    fig_height = 0.65
    bottom_h = left_h = left + width + 0.02

    rect_IMG_EI= [left,bottom+height+0.1,width,height]
    rect_IMG_IE= [left,bottom,width,height]
    rect_histX = [left,bottom_h+height+0.1,width,0.2]
    rect_histY = [left_h,bottom,0.05,height]
    rect_histY2 = [left_h,bottom+height+0.1,0.05,height]

   # start with a rectangular Figure
    fig = plt.figure(1, figsize=(15, 3))

    axIMG_EI = plt.axes(rect_IMG_EI)
    axIMG_IE = plt.axes(rect_IMG_IE)
    axHistX = plt.axes(rect_histX)
    axHistY = plt.axes(rect_histY)
    axHistY2 = plt.axes(rect_histY2)

    # no labels
    axIMG_EI.xaxis.set_major_formatter(nullfmt)
    axHistX.xaxis.set_major_formatter(nullfmt)
    axHistY.xaxis.set_major_formatter(nullfmt)
    axHistY.yaxis.set_major_formatter(nullfmt)
    axHistY2.xaxis.set_major_formatter(nullfmt)
    axHistY2.yaxis.set_major_formatter(nullfmt)

    my_cmap = plt.get_cmap('afmhot',20)#,30
    beta = 0.5
    # plot with imshow
    axp=axIMG_EI.imshow(matrixEI,cmap=my_cmap,interpolation='none',aspect='auto',norm = MidpointNormalize(midpoint=np.max(matrixEI)*beta))
    #axIMG_EI.set_xlabel('E cell rank',weight='bold',fontsize=17)
    #axIMG_EI.set_ylabel('I cell rank',weight='bold',fontsize=17)
    #axIMG_EI.set_xticklabels(range(0,10),np.linspace(0,144,3),fontsize=12)
    #axIMG_EI.set_yticklabels(np.linspace(0,36,3),fontsize=12)

    axpp = axIMG_IE.imshow(matrixIE.T,cmap=my_cmap,interpolation='none',aspect='auto',norm = MidpointNormalize(midpoint=np.max(matrixIE)*beta))#,vmin=np.max(matrixIE)/3.0)
    axIMG_IE.set_xlabel('E cell rank',fontsize=22)
    #axIMG_IE.set_ylabel('I cell rank',weight='bold',fontsize=17)

    # normal plots
    axHistX.plot(orExc[np.argsort(orExc)],'o')
    #axHistX.set_ylabel('Orientation',weight='bold',fontsize=15)
    axHistY.plot(orInh[np.argsort(orInh*-1)],np.linspace(0,len(orInh),len(orInh)),'o')
    axHistY2.plot(orInh[np.argsort(orInh*-1)],np.linspace(0,len(orInh),len(orInh)),'o')

    # set limits by hand
    axHistX.set_xlim((0,len(orExc)))
    axHistX.set_ylim((0,np.around(np.max(orExc)+0.05,2)))
    axHistX.set_yticks(np.linspace(0,np.around(np.max(orExc),1),3))
    axHistX.set_yticklabels((r'$0.0$',r'$\pi$',r'$2\pi$'),fontsize=12)

    axHistY.set_xlim((0,np.around(np.max(orInh)+0.05,1) ))
    axHistY.set_ylim((0,len(orInh)))
    axHistY.set_xticks(np.linspace(0,np.around(np.max(orExc),1),3))
    axHistY.set_xticklabels((r'$0.0$',r'$\pi$',r'$2\pi$'),fontsize=12)

    plt.text(-94.0,10.0,"I cell rank",rotation=90.,size=22)

    cbaxes = fig.add_axes([0.075, 0.2, 0.02, 1.5])
    #cbaxes = plt.axes([0.8, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(axpp,cax=cbaxes,ticks=[0.0,np.max(matrixIE)/2,np.max(matrixIE) ],norm = MidpointNormalize(midpoint=np.max(matrixIE)*beta))
    cbar.set_label('weight value',ha='left',fontsize=22)
    cbar.ax.set_yticklabels(['0.0','','max'])
    cbaxes.yaxis.set_ticks_position('left')
    cbaxes.yaxis.set_label_position('left')

    plt.savefig('./Output/Orientation_'+manner+'.png',dpi=300,bbox_inches='tight', pad_inches = 0.1)
    plt.close('all')
    
    
#------------------------------------------------------------------------------
def startAnalyseRelation():

    orientationsTC = np.load('./work/gabor_orientation.npy')
    frExc = np.load('./work/gabor_frEx.npy')
    frInh = np.load('./work/gabor_frInhib.npy')

    weightsIN_V1 = np.loadtxt('./Input_network/INtoV1.txt')
    weightsV1_IN = np.loadtxt('./Input_network/V1toIN.txt')
    weightsLatIN =np.loadtxt('./Input_network/INLat.txt')

    oMTC,orExc,orInh = calcOrientMatricOverTC(weightsV1_IN,orientationsTC,frExc,frInh)
    oMTCb,orInh,orExc = calcOrientMatricOverTC(weightsIN_V1,orientationsTC,frInh,frExc)
    oMInhib = calcOrientMatrixInhibOverTC(weightsLatIN,orientationsTC,frInh)

    plotExcToInhibOrientation(oMInhib,'oMLateralInhib',False)
    plotOrientationWeights(oMTC,oMTCb,orExc,orInh,'oMTuningCurves')
#------------------------------------------------------------------------------
if __name__=="__main__":
    startAnalyseRelation()
