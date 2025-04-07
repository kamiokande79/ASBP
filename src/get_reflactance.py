#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:37:16 2025

@author: gianluca
"""
import numpy as np
import scipy
from lro_to_tensor import read_parameters
from brdflib.lro.brdf import get_val

spec_parameters = read_parameters("WAC/WAC_HAPKEPARAMMAP_Apollo11.txt")

alpha = 15.
theta_i = (90. - alpha)*np.pi/180.
phi_i = (0.0)*np.pi/180.

int_thetas_o = list(range(0, 89, 2))
int_thetas_o.append(int(90. - alpha))
int_thetas_o = sorted(int_thetas_o)

i_alpha = int_thetas_o.index(int(90. - alpha))

int_phis_o = list(range(0, 181, 10))

thetas_o = [float(angle)*np.pi/180. for angle in int_thetas_o]
phis_o = [float(angle)*np.pi/180. for angle in int_phis_o]

wavelengths = list(spec_parameters.keys())

ref = {int_theta_o: {} for int_theta_o in int_thetas_o}

for ic, theta_o in enumerate(thetas_o):
    for jc, phi_o in enumerate(phis_o):
        brdf = []
        for wavelength, paramters in spec_parameters.items():
            brdf.append(get_val(theta_in=theta_i, theta_out=theta_o,
                        phi_in=phi_i, phi_out=phi_o, parameters=paramters))
        spd_fun = scipy.interpolate.interp1d(wavelengths, brdf)
        ref[int_thetas_o[ic]][int_phis_o[jc]] = scipy.integrate.quad(
            spd_fun, wavelengths[0], wavelengths[-1])[0] / (wavelengths[-1] - wavelengths[0])

result = {}
for phi_o in [0, 90, 180]:
    values = []
    if phi_o == 0:
        start_i = i_alpha + 1
    else:
        start_i = 1
    for theta_o in int_thetas_o[start_i:]:
        values.append(ref[theta_o][phi_o])
    result[phi_o] = sum(values) / len(values)
