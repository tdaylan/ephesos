# Ephesos

## Introduction

Integrating over the sky-projected brightness distribution of bodies in a system, `ephesos.eval_modl()` generates the relative flux light curve of a system of stars, compact objects, and planets. You can see an example model evaluation below, where a hot Jupiter transits a Sun-like star.

![in this plot](https://github.com/tdaylan/ephesus/blob/master/visuals/lcur.png)


## Model
### Limb darkening
Stars manifest a darkening towards their limbs beyond that expected from Lambertian scattering. Even though simulations of stellar atmospheres allow the prediction of the limb darkening, the failure of these models to accurately interpolate stellar parameters can significantly bias inference when a wrong limb darkening model is used. Miletos allows the user to marginalize over the limb darkening using an parametrization that is efficient to sample from (Kipping 2013).


You can find example uses under the examples folder.

