import os

# main function - startAnalysis():
# main function to analyse the results of the newtwork simulation
#   e.g: - start fitting with Gabor-Functions
#        - start calculation of STA and STC
#        - start the image reconstrunction functions

# - create all necessary folders 
# - test if all necessary input files exists
# - test if all python files are exists
# - step by step analysis

#for further informations check the specific python files

def createFolders():
    if not os.path.exists('Output'):
        os.mkdir('Output')
#--------------------------------------------------------------------------
def inputCorrect():
    if not os.path.isfile('Input_network/V1weight.txt'):
        print('No Feed Forward Weights for V1 Layer (V1weight.txt)')
        return(False)

    if not os.path.isfile('Input_network/INLat.txt'):
        print('No Weights for lateral Inhibition (INLat.txt)')
        return(False)

    if not os.path.isfile('Input_network/INtoV1.txt'):
        print('No inhibitory Weights for V1 Layer (INtoV1.txt)')
        return(False)

    if not os.path.isfile('Input_network/V1toIN.txt'):
        print('No Feed Forward Weights for Inhibitory Layer (V1toIN.txt)')
        return(False)

    if not os.path.isfile('Input_Data/IMAGES.mat'):
        print("""No IMAGES.mat found, please download the file from:
        https://www.rctn.org/bruno/sparsenet/IMAGES.mat
        and put in the Input_Data directory""")
        return(False)

    return(True)
#-------------------------------------------------------------------------
def startAnalysis():
    print('Start to analyse the experiments.')
    print('----------------')
    createFolders()
    if not inputCorrect():
        print('Error in Input')
        return()
    else:
        print('Start with STA')
        print('---------------------')
        os.system('python Network/analysis/STA.py')
        print('Start to analyze the response on natural scenes')
        print('---------------------')
        os.system('python Network/net/generateData.py --clean')
        os.system('python Network/analysis/sparseness.py')
        os.system('python Network/analysis/information.py')
        os.system('python Network/analysis/fluctuation.py')
        print('Start with the reconstruction capability')
        print('---------------------')
        os.system('python Network/analysis/IRE.py')
        print('Start with the tuning curves')
        print('---------------------')
        os.system('python Network/analysis/tuning_curves.py')
        os.system('python Network/analysis/relation_Gabor.py')

#-------------------------------------------------------------------------
if __name__=="__main__":
    startAnalysis()
