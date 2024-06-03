# Introduction

This repository contains the source code for a future article to be submitted. 
It investigates the effect of fatigue dynamics on the maximum repetitions of biceps curls.

The repository is made available as accompanying material of [Te publication](https://www.sciencedirect.com/science/article/pii/S0167945724000058)::

```bibtex
@article{Puchaud2023,
title = {The interplay of fatigue dynamics and task achievement using optimal control predictive simulation},
journal = {Human Movement Science},
volume = {94},
pages = {103182},
year = {2024},
issn = {0167-9457},
doi = {https://doi.org/10.1016/j.humov.2024.103182},
url = {https://www.sciencedirect.com/science/article/pii/S0167945724000058},
author = {P. Puchaud and B. Michaud and M. Begon},
keywords = {Biomechanics, Predictive simulation, Direct multiple shooting, NMPC, Biceps curls, Kinematic variability},
abstract = {Predictive simulation of human motion could provide insight into optimal techniques. In repetitive or long-duration tasks, these simulations must predict fatigue-induced adaptation. However, most studies minimize cost function terms related to actuator activations, assuming it minimizes fatigue. An additional modeling layer is needed to consider the previous use of muscles to reveal adaptive strategies to the decreased force production capability. Here, we propose interfacing Xia's three-compartment fatigue dynamics model with rigid-body dynamics. A stabilization invariant was added to Xia's model. We simulated the maximum repetition of dumbbell biceps curls as an optimal control problem (OCP) using direct multiple shooting. We explored three cost functions (minimizing torque, fatigue, or both) and two OCP formulations (full-horizon and sliding-horizon approaches). We adapted Xia's model by adding a stabilization invariant coefficients S=105 for direct multiple shooting. Sliding-horizon OCPs achieved 20 to 21 repetitions. The kinematic strategy slowly deviated from a plausible dumbbell lifting task to a swinging strategy as fatigue onset increasingly compromised the humerus to remain vertical. In full-horizon OCPs, the latter kinematic strategy was used over the whole motion, resulting in 32 repetitions. We showed that sliding-horizon OCPs revealed a reactive strategy to fatigue when only torque was included in the cost function, whereas an anticipatory strategy was revealed when the fatigue term was included in the cost function. Overall, the proposed approach has the potential to be a valuable tool in optimizing performance and helping reduce fatigue-related injuries in a variety of fields.}
}
```
and includes scripts, models and materials to reproduce figures and results of that work.

# Status
| Type | Status |
|---|---|
| License | <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-success" alt="License"/></a> |
| Zenodo  | [![DOI](https://zenodo.org/badge/521383672.svg)](https://zenodo.org/badge/latestdoi/521383672) |

In order to run the code, you need to install the following packages from pyomeca:

`conda install -c conda-forge biorbd=1.9.9`
`conda install -c conda-forge bioviz=2.3.0`
`conda install -c conda-forge bioptim=3.0.1`

# Contents from the paper
Predictive simulation of human motion could provide insight into optimal techniques. In repetitive or long-duration tasks, these simulations must predict fatigue-induced adaptation.  
However, most studies minimize cost function terms related to actuator activations, assuming it minimizes fatigue.  
An additional modeling layer is needed to consider the previous use of muscles to reveal adaptive strategies to the decreased force production capability. 
Here, we propose interfacing Xia's three-compartment fatigue dynamics model with rigid-body dynamics. A stabilization invariant was added to Xia’s model. We simulated the maximum repetition of dumbbell biceps curls as an optimal control problem (OCP) using direct multiple shooting. 
We explored three cost functions (minimizing minimum torque, fatigue, or both) and two OCP formulations (full-horizon and sliding-horizon approaches).  
We found that Xia's model modified with the stabilization invariant (coefficients $S=\{10,5\}$) was adapted to direct multiple shooting. 
Sliding-horizon OCPs achieved 20 to 21 repetitions. The kinematic strategy slowly deviated from a plausible dumbbell lifting task to a swinging strategy as fatigue onset increasingly compromised the ability to keep the arm vertical. 
In full-horizon OCPs, the latter  kinematic strategy was used over the whole motion, resulting in 32 repetitions. 
We showed that sliding-horizon OCPs revealed a reactive strategy to fatigue when only torque was included in the cost function, whereas an anticipatory strategy was revealed when the fatigue term was included in the cost function. Overall, the proposed approach has the potential to be a valuable tool in optimizing performance and helping reduce fatigue-related injuries in a variety of fields. 
