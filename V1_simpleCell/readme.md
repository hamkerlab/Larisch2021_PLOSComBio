## V1 simple cell model from the Larisch, Gönner, Teichmann, Hamker (2021) publication

### Structure

* **input** directory:
  * **must** contain the IMAGES.mat file with the pre-whitened natural scenes from Olhausen and Field (1996) (downloadable from https://www.rctn.org/bruno/sparsenet/IMAGES.mat)
  * **can** contain the weight matrices as .txt files from a previous simulation, or to process in the simulation (note(!): remove the numbers from the file names, for example, V1weight_400000.txt -> V1weight.txt)
* **ann.py** :
  * Python script with the complete network definition, containing triplet STDP learning rule from Clopath et al. (2010), iSTDP rule from Vogels et al. (2011)
  * Present the network $400,000$ randomly chosen natural scenes patches for each $125ms$
  * *loadWeights()* function to load the weights of a previous stimulation (needs weight matrices in the **input** directory)
  * creates the **output** directory to save the weights every $40,000$ stimuli, and for some plots:
    * **excitatory**: contains the weight matrices from LGN to E (V1weight_N.txt) and from LGN to I (InhibW_N.txt)
    * **inhibitory**: contains weight matrices from I to E (INtoV1_N.txt) and from I to I (INLat_N.txt)
    * **V1toIN**: contains the weight matrices from E to I (V1toIN_N.txt)
    * **InhibitLayer**: Contains plots of the excitatory input current (gExc_0.png), inhibitory input current(gInh_0.png), membrane potential(membran_0.png) and the u\\ (vmean_0.png) over the simulation for four inhibitory neurons
    * **V1Layer**: Contains plots of the excitatory input current (gExc_0.png), inhibitory input current(gInh_0.png), membrane potential(membran_0.png), the u\\ (vmean_0.png), the firing rate(frEx.png), $u_-$ (l2umeanLTD_0.png) and $u_+$ (l2umeanLTD_0.png) over the simulation for four excitatory neurons
    * gExc_gInh.png shows the average excitatory input current and average inhibitory input current of the excitatory population during the simulation
    * ffWMean.png the average incoming weight to one excitatory cell over the simulation
  * plots.py: Python script for plotting population-related plots
  * receptive.py: Can be used after simulation to be copied into the output directory to calculate the receptive fields of the excitatory and inhibitory cells over their feed-forward weights. Receptive fields can be found in the **ONOFF** directory

### Simulation

Start the simulation with:

```
python ann.py
```

or

```
python ann.py -j4
```

to use 4 CPU cores to speed up the calculation.