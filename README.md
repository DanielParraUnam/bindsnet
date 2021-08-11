<p align="center"><img width="25%" src="docs/logo.png"/></p>

A Python package used for simulating spiking neural networks (SNNs) on CPUs or GPUs using [PyTorch](http://pytorch.org/) `Tensor` functionality.

BindsNET is a spiking neural network simulation library geared towards the development of biologically inspired algorithms for machine learning.

This package is used as part of ongoing research on applying SNNs to machine learning (ML) and reinforcement learning (RL) problems in the [Biologically Inspired Neural & Dynamical Systems (BINDS) lab](http://binds.cs.umass.edu/).

Check out the [BindsNET examples](https://github.com/BindsNET/bindsnet/tree/master/examples) for a collection of experiments, functions for the analysis of results, plots of experiment outcomes, and more. Documentation for the package can be found [here](https://bindsnet-docs.readthedocs.io).

[![Build Status](https://travis-ci.com/BindsNET/bindsnet.svg?branch=master)](https://travis-ci.com/BindsNET/bindsnet)
[![Documentation Status](https://readthedocs.org/projects/bindsnet-docs/badge/?version=latest)](https://bindsnet-docs.readthedocs.io/?badge=latest)
[![HitCount](http://hits.dwyl.io/Hananel-Hazan/bindsnet.svg)](http://hits.dwyl.io/Hananel-Hazan/bindsnet)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/bindsnet_/community)

## Requirements

- Python 3.6
- `requirements.txt`

## Setting things up

### Using pip
To build the `bindsnet` package from source in a local directory, clone the GitHub repository and install with the following commands

```
git clone https://github.com/DanielParraUnam/bindsnet
pip install ./bindsnet
```

Or, to install in editable mode (allows modification of package without re-installing):

```
pip install -e ./bindsnet
```

To install the packages necessary to interface with the [OpenAI gym RL environments library](https://github.com/openai/gym), follow their instructions for installing the packages needed to run the RL environments simulator (on Linux / MacOS).

## More Information

For information about getting started, interfacing with [OpenAI gym RL environments library](https://github.com/openai/gym), running tests, benchmarking as well as citations, contributors and general background; please visit the main repository at https://github.com/BindsNET/bindsnet.

## License
GNU Affero General Public License v3.0
