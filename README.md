Denoising diffusion models for geophysical systems
==================================================

This repository is a further development of 
https://github.com/cerea-daml/ddm-attractor towards large-scale geophysical 
systems.

If you want to cite 

> Not yet available

--------

This repository represents a python toolbox useable for denoising diffusion 
models and its application to geophysical systems.
The repository includes the source code for the module itself and examples with 
toy models, e.g., with the Lorenz 1996 model.
The module is programmed for research purpose in an easily extendable modular 
way.
Focused on scalable approaches, the modules should enable a large-scale 
geophysical application.
Additionally, the modules have a built-in support for data masking, allowing an 
application to ocean and sea-ice data where land might be masked out.

The scripts and modules are written in PyTorch [[2]](#2), 
PyTorch lightning [[3]](#3).

The `examples` folder includes an example based on the Lorenz 1996 model on how 
to apply the toolbox to train diffusion models. 
Another example is the https://github.com/cerea-daml/diffusion-nextsim-regional 
repository where the toolbox has been used to train diffusion models for 
regional surrogate modeling in the Arctic.

The folder structure is as follows:
```
.
|-- ddm_dynamical               # The Python source code
    |-- callbacks               # Common PyTorch lightning callbacks
    |-- decoder                 # Example decoder modules
    |-- encoder                 # Example encoder modules
    |-- layers                  # Common neural network layers like embeddings
    |-- sampler                 # Sampler to generate new data
    |-- scheduler               # Noise scheduler
    |-- optimal_model.py        # PyTorch module where the training data is 
    stored to get the optimal solution
    |-- parameterization.py     # The parameterizations specifying the output 
    of the diffusion model
    |-- utils.py                # Utility functions like estimating a masked 
    average
    |-- weighting.py            # Modules with different weightings can be used 
    to train diffusion models.
|-- examples                    # A folder with examples
    |-- lorenz_96               # Examples with the Lorenz96 system
|-- environment.yml             # A possible environment configuration
|-- LICENSE                     # The license file
|-- README.md                   # This readme
```
The different submodules and files represent specific parts of a diffusion
model, further explained in the following:

Instead of acting in data space, the here instantiated diffusion models act in a
latent space, where the `encoder` maps from data to latent space and the 
`decoder` maps back from latent to data space.
Two different processes can specify the `encoder` and `decoder`, the physical 
part can correspond for a mapping from physical space into a normalized space, 
e.g., for a Gaussian, the data is normalized by a given mean and standard
deviation.
The neural network part can specify a mapping into a compressed space like in a
classical autoencoder.
The loss method specified in the `decoder` creates a common interface to train 
such autoencoder.

Trained with a given target dataset, diffusion models diffuse the data mapped 
into latent space by adding Gaussian noise until all the information in the data
is replaced by pure random noise.
The time spent at a given noise amplitude is specified by the noise `scheduler`.
The neural network is trained to reverse this process and to denoise data at a
given noise amplitude.
Different `parameterizations` can parameterize the neural network output.
The chosen `parameterization` also determines the loss function to train the
denoiser; the loss function here corresponds to the loss function for noise
prediction if no conditioning on the noise scheduler and weighting would be
applied.
`Weighting` modules can overweight specific noise amplitudes during training,
this can be useful if e.g. the neural network should focus more on large-scale
features.
`Callbacks` can help to watch the training of the diffusion model, e.g., by
tracking the adaption of the noise scheduler during training.
In addition to noised data samples, the neural network gets conditional
information, which can be processed and embedded by specific neural network
`layers`.

After training, the neural network can be used as denoiser to generate new
samples.
In mathematical terms, the sample generation corresponds to an integration of a
stochastic differential equation (SDE) or an ordinary differential equation
(ODE).
The `sampler` specifies then the integration scheme and if either a SDE or ODE
is integrated in pseudo-time.
The time stepping in the integration scheme depends on the number of steps and
the noise `scheduler` chosen for the sampling.

With the Gaussian diffusion process, the `optimal model` can be estimated based
on the training dataset where the optimal model corresponds to a Gaussian
mixture of the training dataset samples.
While this optimal model can be used for theoretical purposes, it just recovers
the samples from the training dataset and suffers from the curse of
dimensionality, possibly failing to generalize beyond the given samples.

If you have further questions, please feel free to contact me (@tobifinn) or to
create a GitHub issue.

--------
## References
<a id="1">[1]</a> https://pytorch.org/

<a id="2">[2]</a> https://pytorch.org/

<a id="3">[3]</a> https://www.pytorchlightning.ai/
