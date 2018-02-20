The current implemented BRDF model was developed by Bruce Hapke in the serie of articles "Bidirectional Reflectance Spectroscopy", with the correction for macroscopic roughness.

Different BRDFs related to different Hapke parameters are available:

Hapke-Johnson, based on the parameters derived from the Apollo 11 soil sample 10084, published by Johnson et.al. in "Spectrogoniometry and modeling of martian and lunar analog samples
and Apollo soils".

Hapke-Shevchenko, based on the parameters proposed by Pugacheva and Shevchenko in "The parameters involved in hapke's model for estimation of the composition of the ejecta lunar terrains"

Hapke-LRO, based on the parameters of the Apollo 11 landing site derived by the LRO observations and published in Sato et.al. "Resolved Hapke parameter maps of the Moon".

Hapke-Helfenstein, based on the parameters for dark terrains proposed by Helfenstein and Veverka in "Photometric Properties of Lunar Terrains Derived from Hapke's Equation"

All the BRDFs are scaled by a constant to match a normal albedo of ~0.09 of the Apollo 11 soil sample 10084 as measured by Piatek et.al in "Scattering properties of lunar regolith samples determined by mimsa fits".

The BRDF based on the parameters proposed by Pugacheva and Shevchenko is included for comparison purposes only. Those parameters are in contraddiction with the parameters obtained from the analysis of the Apollo soil samples proposed by Johnson et.al., resulting, moreover, in a apparently incositent global behaviour of the Moon's disk (brightness becomes strongly dipendent from both the incident and observation angles). Therefore, Hapke-Johnson model is at the moment the most accurate model available for this project.

Additionally, a model based on the paper published in 1979 by Shevchenko "A new three-dimensional scattering indicatrix and the photometric properties of the moon"

More BRDF models will come hopefully later on.
