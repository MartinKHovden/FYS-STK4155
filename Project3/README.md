# FYS-STK project 3
## Solving PDEs and eigenvalue problems using neural networks
In this project we use neural networks for solving PDEs. In particular, we will focus on the diffusion equation. The results using the neural network is compared to the forward euleur method. We will see that the precision of the classic numerical methods are better. However, there are problems where the numerical methods don't work. Using neural networks might then be a good choice.

## Structure of the repository
In the plots folder you will find plots produced with the various scripts in the repository. The
plots are named according to the problem they solve and which parameters are used. These can be used as benchmarks together with the tables in the report.
In the file diffusion_solvers.py various methods are found for solving the diffusion equation with numerical methods. The analytical solution functions can also be found there.
The remaining scripts and jupyter notebooks contains code for solving the part of the project corresponding to the name of the file.

## Testing
To run test on the library, run the following command in the terminal:
```python
C:... > pytest diffusion_solvers.py
```
Make sure to be in the correct folder.
