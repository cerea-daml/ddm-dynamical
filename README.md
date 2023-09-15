Denoising diffusion models for geophysical systems
==================================================

This repository is a further development of https://github.com/cerea-daml/ddm-attractor towards large-scale geophysical systems.

If you want to cite 

> Not yet available

--------

This repository represents python modules useable for denoising diffusion 
models and its application to geophysical systems.
The repository includes the source code for the module itself and examples with toy models, e.g., with the Lorenz 1996 model.
The module is programmed for research purpose in an easily extendable modular way.

The focus of the denoising diffusion models is on latent diffusion models to enable a large-scale application.
Additionally, the training schemes have a built-in support for data masks.
This allows the diffusion models to be applied on data for sea ice and computational fluid dynamics.

The scripts and modules are written in PyTorch [[2]](#2), PyTorch lightning [[3]](#3).

The folder structure is as follows:
```
.
|-- ddm_dynamical       # The Python source code
    |-- callbacks       # Common PyTorch lightning callbacks
    |-- decoder         # Example decoder modules
    |-- dynamical       # Toy models and a Runge-Kutta fourth-order integrator
    |-- encoder         # Example encoder modules
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

<a id="2">[2]</a> https://pytorch.org/

<a id="3">[3]</a> https://www.pytorchlightning.ai/
