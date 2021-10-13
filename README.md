This repository contains the machine learning code used in the paper: Nowcasting thunderstorm hazards using machine learning: the impact of data sources on performance, Natural Hazards and Earth System Sciences, 2021, https://doi.org/10.5194/nhess-2021-171

# Installation

You need NumPy, Pandas, Matplotlib, LightGBM, SKLearn and NetCDF4 for Python.

Clone the repository, then, in the main directory, run
```bash
$ python setup.py develop
```
(if you plan to modify the code) or
```bash
$ python setup.py install
```
if you just want to run it.

# Downloading data

The dataset can be found at the following Zenodo repository: https://doi.org/10.5281/zenodo.5566730

Download the NetCDF file. You can place it in the `data` directory but elsewhere on the disk works too.

# Running

Go to the `scripts` directory. There, running
```bash
python gradboost_experiments.py --nc_file <path_to_netcdf_file>
```
will reproduce all the machine learning experiments of the paper. Alternatively, it may be more convenient to start an interactive Python prompt and examine the function `all_experiments` in `gradboost_experiments.py`, then run code from that function line by line.

After the code has finished running, you will find the figures in the `figures` directory. Some small differences to the results in the paper are possible due to different implementations etc.
