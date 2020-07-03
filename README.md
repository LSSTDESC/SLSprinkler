# LSST DESC SL Sprinkler

The DESC SL (Strong Lensing) Sprinkler adds strongly lensed AGN and SNe to simulated catalogs, generates postage stamps for these systems, and computes the lensing observables given the DC2 cosmology. 
It originated as part of the [LSST DESC Twinkles project](https://www.github.com/LSSTDESC/Twinkles).

The code for generating the postage stamps and truth catalogs uses the Lenstronomy lens modeling package ([Birrer & Amara 2018](https://arxiv.org/abs/1803.09746v1)), available [here](https://github.com/sibirrer/lenstronomy).

See `SCHEMA.md` for the truth table schema. Please contact the following contributors with questions or feedback: 

- Bryce (@jbkalmbach) about the Sprinkler matching
- Ji Won (@jiwoncpark) about the postage stamp generation and truth tables
