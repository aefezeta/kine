# kine

`kine` is a Python package designed for video reconstruction of variable and sparse radio-interferometric data. It models the time-dependent brightness distribution of the observed source through a fully unsupervised _Neural Field_, or coordinate-based neural network.

![kine video reconstruction from EHT-like data](gif/kine_EHT.gif)

### Features

`kine` currently supports full polarimetric video and image reconstruction, simultaneous static and dynamic video decomposition, simultaneous fitting of complex telescope gains, and GPU-based Non-Uniform Fast Fourier Transform (NUFFT) computations.

## Documentation

Full documentation will be available soon.

Some (quite comprehensive) example scripts on how to use `kine` can be found in [examples](https://github.com/aefezeta/kine/tree/main/examples). To run the code, simply execute the provided _bash_ wrapper from within the folder:

    $ bash run_kine.sh

## Installation

`kine` relies on the `JAX` library for GPU computations and requires a careful installation of CUDA-related packages and others. For reference, a working conda environment can be found in [environment.yml](https://github.com/aefezeta/kine/tree/main/environment.yml). Detailed instructions on the installation will be provided in the near future.

## Developers

`kine` is developed and maintained by:
 - Antonio Fuentes (antoniofuentesfdez@gmail.com)
 - Marianna Foschi (foschimarianna@gmail.com)
 - Brandon Zhao (byzhao@caltech.edu)

 ## Citation

 If you use `kine` in your publication, please cite:
1. Foschi, M., Zhao, B., Fuentes, A. et al. "Video reconstruction of variable interferometric observations with neural fields." Under rev. in Nature (2026).
2. Fuentes, A., Foschi, M. et al. "Validation of horizon-scale Sagittarius A* video reconstructions with kine" Under rev. (2026).