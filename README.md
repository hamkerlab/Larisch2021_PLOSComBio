## Python scripts for the V1 simple cell model from Larisch,R., Gönner,L., Teichmann,M., Hamker, F.H. (2021)

#### Dependencies

* Python v3.6 (Python >v3.6 and Python v2.7 not tested)
* Numpy >= v1.11.0
* Matplotlib >= v1.5.1
* ANNarchy >= v4.6.8.1 (actual master release can be found [here](https://bitbucket.org/annarchy/annarchy/downloads/?tab=branches) and an installation guide is provided in the [documentation](https://annarchy.readthedocs.io/en/stable/intro/Installation.html) )

#### Structure

The **V1_simpleCell** directory contains the python script to learn the V1 layer 4 models on natural scenes.

Start the learning with:

```
python ann.py
```

or

```
python ann.py -j4
```

to use 4 CPU cores to speed up the calculation.

The **analyzes_V1Simple** directory contains all the experiments to evaluate the model. Before starting the evaluation, copy the weight matrices into the **Input_network** directory.

Start the complete evaluation with:

```
python start.py
```
