# Ephesos

## Introduction

Integrating over the sky-projected brightness distribution of bodies in a system, `ephesos.eval_modl()` generates the relative flux light curve of a system of stars, compact objects, and planets. You can see an example model evaluation below, where a hot Jupiter transits a Sun-like star.

![in this plot](https://github.com/tdaylan/ephesus/blob/master/visuals/lcur.png)

Ephesos can be used to model the following features in light curves:

- dimming due to transits of stellar companions, planets, and their moons,
- variations due to star spots and faculae,
- occultations between planets and such stellar surface features,
- phase variations due to tidal deforming, Doppler beaming, and surface temperature distribution of the companion,
- ingress and egress anomalies during the eclipse of a companion due to strong gradients in its the surface brightness distribution, which is also known as eclipse mapping,
- microlensing due to a compact companion such as a white dwarf, neutron star, or black hole,
- reflected light from a companion.


## Model
### Limb darkening
Stars manifest a darkening towards their limbs beyond that expected from Lambertian scattering. Even though simulations of stellar atmospheres allow the prediction of the limb darkening, the failure of these models to accurately interpolate stellar parameters can significantly bias inference when a wrong limb darkening model is used.


You can find example uses under the examples folder.

