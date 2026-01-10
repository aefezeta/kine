# kine

_A Python package for the video reconstruction of variable (and static) astronomical objects_

`kine` is a Python package designed for the video reconstruction of variable and sparse VLBI data by modeling the time-dependent brightness distribution of the observed source through a completely unsupervised _Neural Field_.

![kine video reconstruction from EHT-like data](gif/kine_EHT.gif)

## Documentation

Documentation will be added here.

Some example scripts on how to use `kine` can be found in [examples](https://github.com/aefezeta/kine/tree/main/examples).

## Installation

`kine` relies on the `JAX` library for GPU computations and requires a careful installation of CUDA-related packages and others. For reference, a working conda environment can be found in [environment.yaml](https://github.com/aefezeta/kine/tree/main/environment.yaml). Detailed instructions on the installation will be provided in the near future.

## Developers

`kine` is developed and maintained by:
 - Antonio Fuentes antoniofuentesfdez@gmail.com
 - Marianna Foschi foschimarianna@gmail.com
 - Brandon Zhao byzhao@caltech.edu

 ## Citation

 If you use `kine` in your publication, please cite:
  - Foschi, M., Zhao, B., Fuentes, A. et al. "Video reconstruction of
    variable interferometric observations with neural fields." Under rev.
    in Nature (2026).
  - Fuentes, A., Foschi, M. et al. "Validation of horizon-scale
    Sagittarius A* video reconstruction with kine" Under rev. (2026).