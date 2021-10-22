## Methods to analyze the V1 layer 4 simple cell model

### Structure

It follows a list of the directories, with the containing files and a short description of the python scripts

* **input_Data** :
  * contains the natural scene data set from Olshausen and Field (1996) (IMAGES.mat)
  * can be download from https://www.rctn.org/bruno/sparsenet/IMAGES.mat
* **Network/net/** :
  * *STA_Run.py*: Script to present the network noisy input to perform the Spike Triggered Average analysis.
    * Inputs are saved in *work/STA_Input.npy*
    * Firing rate of the excitatory population in *work/STA_frExc.npy*
    * Firing rate of the inhibitory population in *work/STA_frInh.npy*
  * *activity_Run.py*: Script to present 30000 natural scene patches and to record the activity of:
    * the LGN (*work/Active_frLGN.npy*)
    * the excitatory(*work/Active_fr.npy*)
    * inhibitory population (*work/Active_frInhib.npy*)
  * *generateData.py*: Show a few natural scene patches and record the spike timings to plot the raster plot of the excitatory population.
    * Raster plot created in *Output/V1Layer/Short_raster.jpg*
  * *fluctuation.py*: Show single natural scene patches 100 times.
    * Record firing rate of excitatory population(*./work/fluctuation_frExc.npy*)
    * Save presented input patches (*./work/fluctuation_Input.npy*)
  * *IRE_Run.py*: Cut the ten natural scene images into patches and present them one after another to record the firing rates.
    * Images (*work/IRE_Images.npy*)
    * Patches (*work/IRE_singlePatches.npy*)
    * Firing rates of the LGN population(*work/IRE_LGNFR.npy*)
    * Firing rates of the excitatory population (*work/IRE_fr.npy*).
  * *tuningCurves_sinus.py*: Script to present sinusoidal gratings to measure tuning curves on different contrast levels and calculate the tuning curve relative to the preferred orientation. Saves of the excitatory population:
    * Firing rate (*work/TuningCurves_sinus_Exc.npy*)
    * Membrane potential (*work/TuningCurves_sinus_Exc_VM*)
    * Excitatory input current (*work/TuningCurves_sinus_Exc_gE*)
    * Inhibitory input current(*work/TuningCurves_sinus_Exc_gI*)
    * Related parameters (*work/TuningCurves_sinus_Exc_parameters.npy*)
  * Further the values for the relative curve for:
    * Firing rates(*work/TuningCurvesRelativ_sinus_Exc.npy*)
    * Membrane potential(*work/TuningCurvesRelativ_sinus_Exc_VM.npy*)
    * Excitatory input currents(*work/TuningCurvesRelativ_sinus_Exc_gE.npy*)
    * Inhibitory input current (*work/TuningCurvesRelativ_sinus_Exc_gI.npy*).
  * And the absolute tuning curves of the inhibitory population (*work/TuningCurves_sinus_Inh.npy*) and some related parameters (*work/TuningCurves_sinus_Inh_parameters.npy*)
  * *measureOrient.py*: Script to measure the preferred orientation of cell by presenting a 2D-Gabor
    * Firing rate of excitatory population (*work/gabor_frEx.npy*)
    * Firing rate of inhibitory population (*work/gabor_frInhib.npy*)
    * List of orientations (*work/gabor_orientation.npy*)
  * *net.py*: Definition of the network for the analyzes. Implemented with ANNarchy. Needs in the *Input_network* directory the weight matrices of the network as .txt files( *V1weight.txt*, *InhibW.txt*, *V1toIN.txt*, *INtoV1.txt*, *INLat.txt*; without numbers, for example: V1weight_400000.txt -> V1weight.txt).
  * *Gabor.py*: Script to define a 2D-Gabor function
  * *Gabor_sinus.py*: Script to define a 2D- Sinusoidal Grating
* **Network/analysis/** :
  * *STA.py*: Calculate the Spike Triggered Average.
    * Input:
      * *work/STA_frExc.npy*
      * *work/STA_frInh.npy*
    * Output:
      * *Output/STA/STAEX.jpg*: STA of the excitatory population
      * *Output/STA/STAIN.jpg*: STA of the inhibitory population
  * *sparseness.py*: Calculate the population spareseness after.
    * Input:
      * *work/Active_fr.npy*
    * Output:
      * *Activ/Sparseness/sparsVinjeGallant_histInput.png*: Histogramm over the population sparseness values
  * *information.py*: Calculate the mutual information in bits/spike
    * Input:
      * *work/fluctuation_frExc.npy*
    * Output:
      * *Output/Activ/Stats/mutualPerSpikeVG_hist.png*
  * *IRE.py*: Calculate the image reconstruction error for the ten natural scenes.
    * Input:
      * *work/IRE_Images.npy*
      * *work/IRE_singlePatches.npy*
      * *work/IRE_LGNFR.npy*
      * *work/IRE_fr.npy*
    * Output:
      * *Output/IRE/IREImage_0.npy* - *IREImage_9.npy*: Reconstructed images
      * *Output/IRE/IRE_0.txt* - *IRE_9.txt*: The normalized mean square error for all images
      * *Output/IRE/IRE_all.png*: Scatter plot showing all errors
  * *tuning_curves.py*: Analyse the tuning curves on sinusoidal grating input and different cotrast levels. Tuning curve values relativ to preffered orientation are used.
    * Input:
      * *work/TuningCurvesRelativ_sinus_Exc.npy*
      * *work/TuningCurvesRelativ_sinus_Exc_VM.npy*
      * *work/TuningCurvesRelativ_sinus_Exc_gE.npy*
      * *work/TuningCurvesRelativ_sinus_Exc_gI.npy*
      * *work/TuningCurves_sinus_Exc_parameters.npy*
    * Output:
      * *Output/TC_sinus/excitatory/meanVM_OT_OR_allC.png*: 3x2 gridd plot to present the tuning curves, the average population membrane potential, the standard deviation of the population membrane potential, the average excitatory and inhibitory input current, and the difference between them on different contrast levels.
      * *Output/TC_sinus/excitatory/meanVM_OT_OR_vm_stdOR_MembrDiff_extra.png*: Average population membrane potential for low and high contrast level
      * *Output/TC_sinus/excitatory/meanSC_OT_OR_SC_stdOR_SCDiff_extra.png*: Average popluation spike count for high and low contrast
      * *Output/TC_sinus/excitatory/meanGD_OT_OR_GD_stdOR_GDDiff_extra.png*: Difference between excitatory and inhibitory input current for high and low stimuli contrast.
      * *Output/TC_sinus/excitatory/TC_contrast_gain.png* : Spike count as function of the input current for low and high stimuli contrast
      * *Output/TC_sinus/BW_est_Pi.png*: Scatter plot to show the orienation band with (OBW) for all cells of the excitatory population
      * *Output/TC_sinus/BW_hist_est_Pi.png*: Histogramm over the OWB of the excitatory population
      * *Output/TC_sinus/MeanbandWith.png*: Mean population OWB of the excitatory population for differen levels of input contrast
      * *Output/TC_sinus/orientation_hist.png*: Histogram over the measured preffered orientations
  * *relation_Gabor.py*: Analyze the connection between similar and different orientation neurons between the excitatory and inhibitory population
    * Input:
      * *work/gabor_orientation.npy*
      * *work/gabor_frEx.npy*
      * *work/gabor_frInhib.npy*
    * Output:
      * *Output/Orientation_oMTuningCurves.jpg*: Between excitatory and inhibitory population
      * *Output/Orientation_oMLateralInhib.jpg*: Between neurons of the inhibitory population

Following directories will be created:

* **work**: contains all the analysis related data
* **Output**: contains all the figures for the analysis

### How to start

Before you start the complete evaluation process, make sure you copied the weight matrices of the network in to the **Input_network** directory, without the numbers. Please **note**: The complete analyze process, depending on your system, can take around **16** hours. To start the process type:

```
python start.py
```

To start just the analyze and plotting routins on the recorded data of the experiments type (takes only 10 minutes):

```
python start_analysis.py
```

Or to start a single experiment you can type:

```
python Network/net/*.py" 
```

For example:

```
python Network/net/IRE_Run.py" 
```

Start a related analyse python script with:

```
python Network/analysis/*.py 
```

For example:

```
python Network/analysis/IRE.py 
```

### Files in the work directory

All the necessary data, which are generating to processing the experiments, are saved in the **work** directory. It follows a list with all the generated \*.npy files and there structure. Please **note**: '#' is to read as 'number of' (for example: '#Excitatory cells' to read as 'Number of excitatory cells')

* *Active_fr.npy*
  * Source: **Network/net/activity_Run.py**
  * Structure: [#Excitatory cells, #Input patches]
* *fluctuation_frExc.npy*
  * Source: **Network/net/fluctuation.py**
  * Structure: [#Excitatory cells X #Input patches X #Repetitions]
* *fluctuation_Input.npy*
  * Source: **Network/net/fluctuation.py**
  * Structure: [#Input patches X Patch height X Patch width X 2]
* *IRE_fr.npy*
  * Source: **Network/net/IRE_Run.py**
  * Structure: [#Excitatory cells X #Natural scenes X #Patches]
* *IRE_Images.npy*
  * Source: **Network/net/IRE_Run.py**
  * Structure:[#Natural scenes X 504 X 504 X2 ]
    * **Note**: the natural scenes are cropped to a resolution of 504 x 504 pixels to cut out 12 x 12 pixel sized patches
* *IRE_LGNFR.npy*
  * Source: **Network/net/IRE_Run.py**
  * Structure: [#LGN cells X #Natural scenes X #Patches]
* *IRE_singlePatches.npy*
  * Source: **Network/net/IRE_Run.py**
  * Structure: [#Patches X Patch high X Patch width]
* *STA_frExc.npy*
  * Source: **Network/net/STA_Run.py**
  * Structure: [#Excitatory cells X #Input patches]
* *STA_frInh.npy*
  * Source:**Network/net/STA_Run.py**
  * Structure: [#Inhibitory cells X #Input patches]
* *STA_Input.npy*
  * Source:**Network/net/STA_Run.py**
  * Structure:[#Input patches X Patch high X Patch width X 2]
* *gabor_frEx.npy*
  * Source: **Network/net/measureOrient.py**
  * Structure: [#Excitatory cells X Patch high X Patch width X #Orientations]
* *gabor_frInhib.npy*
  * Source: **Network/net/measureOrient.py**
  * Structure: [#Inhibitory cells X Patch high X Patch width X #Orientations]
* *gabor_orientation.npy*
  * Source: **Network/net/measureOrient.py**
  * Structure: [ Patch high X Patch width X #Orientations]
* *TuningCurve_sinus_frEx.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X #Orientaions X #Phases X #Frequency X #Contrast levels X #Repeats]
* *TuningCurve_sinus_frInhib.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Inhibitory cells X #Orientaions X #Phases X #Frequency X #Contrast levels X #Repeats]
* *TuningCurve_sinus_gExc_Ex.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X #Orientaions X #Phases X #Frequency X #Contrast levels X #Repeats]
* *TuningCurve_sinus_gInh_Ex.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X #Orientaions X #Phases X #Frequency X #Contrast levels X #Repeats]
* *TuningCurve_sinus_orientation.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Orientaions]
* *TuningCurve_sinus_VM_Ex.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X #Orientaions X #Phases X #Frequency X #Contrast levels X #Repeats]
* *TuningCurves_sinus_Exc.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X #Orientaions X #Contrast levels X #Repeats]
* *TuningCurves_sinus_Exc_gE.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X #Orientaions X #Contrast levels X #Repeats]
* *TuningCurves_sinus_Exc_gI.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X #Orientaions X #Contrast levels X #Repeats]
* *TuningCurves_sinus_Exc_parameters.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X 3 X #Contrast levels]
    * Note: the three parameters which are saved are orienation, phase and frequency of the sinusoidal gratings that lead to the highest activity
* *TuningCurves_sinus_Exc_VM.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X #Orientaions X #Contrast levels X #Repeats]
* *TuningCurves_sinus_Inh.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Inhibitory cells X #Orientaions X #Contrast levels X #Repeats]
* *TuningCurves_sinus_Inh_parameters.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure:[#Inhibitory cells X 3 X #Contrast levels]
    * Note: the three parameters which are saved are orienation, phase and frequency of the sinusoidal gratings that lead to the highest activity
* *TuningCurvesRelativ_sinus_Exc.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure: [#Excitatory cells X #Orientaions X #Contrast levels X #Repeats]
* *TuningCurvesRelativ_sinus_Exc_gE.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure:[#Excitatory cells X #Orientaions X #Contrast levels X #Repeats]
* *TuningCurvesRelativ_sinus_Exc_gI.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure:[#Excitatory cells X #Orientaions X #Contrast levels X #Repeats]
* *TuningCurvesRelativ_sinus_Exc_VM.npy*
  * Source: **Network/net/tuningCurves_sinus.py**
  * Structure:[#Excitatory cells X #Orientaions X #Contrast levels X #Repeats]
* *fluctuation_tm_input.npy*
  * Source: **Network/analysis/fluctuation.py**
  * Structure: [#Input patches X #Input patches]
* *V&G_Population.npy*
  * Source: **Network/analysis/sparseness.py**
  * Structure: [#Excitatory cells]
* *OrientBandwithEst_Half_Sinus.npy*
  * Source: **Network/analysis/tuning_curves.py**
  * Structure: [#Excitatory cells, #Contrast levels]
