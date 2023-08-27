Denoising diffusion models for dynamical systems
================================================

This repository is based on the code of

> Not yet available

--------

This repository represents a python module useable for denoising diffusion 
models and its application to dynamical systems.
The repository includes the source code for the module itself and examples with
toy models.
The module is programmed for research purpose in an easily extendable modular
way.

The scripts and module is written in PyTorch [[1]](#1), Pytorch lightning [
[2]](#2).

The folder structure is as follows:
```
.
|-- ddm_dynamical       # The Python source code
    |-- dynamical       # Toy models and a Runge-Kutta fourth-order integrator
    |-- head_param      # Paramterizations defining the head of the neural network
    |-- layers          # Layers for the neural network
    |-- sampler         # Sampler to generate new data
    |-- scheduler       # Noise scheduler
|-- examples            # A folder with examples
    |-- lorenz_96       # Examples with the Lorenz96 system
|-- environment.yml     # A possible environment configuration
|-- LICENSE             # The license file
|-- README.md           # This readme
```

If you have further questions, please feel free to contact me (@tobifinn) or to
create a GitHub issue.

--------
## References
<a id="1">[1]</a> https://pytorch.org/

<a id="2">[2]</a> https://www.pytorchlightning.ai/
