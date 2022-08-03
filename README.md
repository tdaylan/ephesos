# Ephesus

## Introduction

Ephesus is a library for exoplanet science. It provides functions for processing (reading TESS and Kepler data, flattening, phase-folding and plotting) light curves. It can be used to generate synthetic light curves.

## functions
Ephesus library contains a suite of functions useful in time-domain and/or exoplanet research.

### Peridic box finder

Using the Box Least Squares (BLS) algorithm, `ephesus.srch_pbox()` searches for periodic boxes in time-series data.


### Light Curve Generator
Integrating over the sky-projected brightness distribution of stars and planets, `ephesus.retr_rflxtranmodl()` generates the relative flux light curve of a system of stars, compact objects, and planets. You can see an example model evaluation ![in this plot](https://github.com/tdaylan/ephesus/visuals/lcur.png).

