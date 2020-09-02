# storm_forecast Project Repository

Copyright : 2019, Quantmetry

Research subject on tropical storms prediction, based on a RAMP competition. Source data is also comming from the same source

Source : https://ramp.studio/problems/storm_forecast

## 1. Setup your virtual environment and activate it

Goal : create a local virtual environment in the folder `./.venv/`.

- First: check your python3 version:

    ```
    $ python3 --version
    # examples of outputs:
    Python 3.6.2 :: Anaconda, Inc.
    Python 3.7.2

    $ which python3
    /Users/benjamin/anaconda3/bin/python3
    /usr/bin/python3
    ```

    - If you don't have python3 and you are working on your mac: install it from [python.org](https://www.python.org/downloads/)
    - If you don't have python3 and are working on an ubuntu-like system: install from package manager:

        ```
        $ apt-get update
        $ apt-get -y install python3 python3-pip python3-venv
        ```

- Now that python3 is installed create your environment and activate it:

    ```
    $ make init
    $ source activate.sh
    ```

    You sould **allways** activate your environment when working on the project.

    If it fails with one of the following message :
    ```
    "ERROR: failed to create the .venv : do it yourself!"
    "ERROR: failed to activate virtual environment .venv! ask for advice on #dev "
    ```

    instructions on how to create an environment by yourself can be found in the
    [tutorials about virtual environments](https://gitlab.com/quantmetry/qmtools/TemplateCookieCutter/blob/master/tutorials/virtualenv.md)


## 2. Install the project's requirements
If you want to use GPU, you need to install the appropriate NVIDIA drivers and CUDA libraries first. For compatibility issues, you need to use the 10.1 version of CUDA. For a proprer and easy installation of these, check the [tensorflow documentation](https://www.tensorflow.org/install/gpu). The tensorflow library (version 2.1.0) will automatically use CUDA, there is no need to use the tensorflow-gpu library.
You need to add the following line to your `.bashrc`:

```
export CUDA_HOME=/usr/local/cuda-10.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
```

You can then activate your environment and install the libraries.
```
(path/to/here/.venv)$ make install
```

*WARNING*: If you are using GPU, you must ensure determinism between successive runs (especially for optimizing hyperparameters). Please ensure you have the following lines at the beginning of the `main.py` file:
```
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

## 3. Start coding

To run the code, une the `make main` command, which will run the `model/main.py` file.
*WARNING* by default, the backend used for Keras is Tensorflow. If you want to change this, you need to pass the backend parameter to the make command as follow: `make backend=theano main`, if you want to use Theano backend for example.
