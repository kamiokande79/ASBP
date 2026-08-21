import csv
from pathlib import Path
from collections import OrderedDict
import numpy as np
from brdflib.coordinates import u2theta
from brdflib.ndf import eval_ndf, eval_vndf, projected_area, weighted_vndf
from brdflib.lro.brdf import get_backscatter, get_spectra


# Symbols | Name
# ------------------------------------------------------------------------------------
#   w     | Single scattering albedo
#   b     | Henyey-Greenstein double-lobed single particle phase function parameter
#   c     | Henyey-Greenstein double-lobed single particle phase function parameter
#   Bc0   | Amplitude of Coherent Backscatter Opposition Effect (CBOE) - fixed at 0.0
#   hc    | Angular width of CBOE - fixed at 1.0
#   Bs0   | Amplitude of Shadow Hiding Opposition Effect (SHOE)
#   hs    | Angular width of SHOE
#   theta | Effective value of the photometric roughness - fixed at 23.657
#   phi   | Filling factor - fixed at 0.0

# Product Name            |      Wavelength      | # of Bands
# ------------------------------------------------------------
# WAC_HAPKEPARAMMAP_321NM | 321 nm (WAC band 1) |      9
# WAC_HAPKEPARAMMAP_360NM | 360 nm (WAC band 2) |      9
# WAC_HAPKEPARAMMAP_415NM | 415 nm (WAC band 3) |      9
# WAC_HAPKEPARAMMAP_566NM | 566 nm (WAC band 4) |      9
# WAC_HAPKEPARAMMAP_604NM | 604 nm (WAC band 5) |      9
# WAC_HAPKEPARAMMAP_643NM | 643 nm (WAC band 6) |      9
# WAC_HAPKEPARAMMAP_689NM | 689 nm (WAC band 7) |      9
# WAC_HAPKEPARAMMAP_7BAND |  all 7 wavelengths  |     63


def read_parameters(filename: Path) -> dict[int, np.ndarray]:
    """
    Read the WAC Hapke paramters text file.

    Parameters
    ----------
    filename : Path
        DESCRIPTION.

    Returns
    -------
    parameters : dict[int, np.ndarray]
        WAC Hapke measured parameters.

    """
    wac_wavelengths = [321, 360, 415, 566, 604, 643, 689]
    n_wl = len(wac_wavelengths)
    n_params = 9
    parameters = OrderedDict(zip(wac_wavelengths, [[0.0] * n_params for ic in range(n_wl)]))
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        wc = 0  # wavelength counter
        pc = 0  # parameter counter
        for ic, row in enumerate(spamreader):
            if ic == 0:
                # skip the first line
                continue
            if pc < n_params:
                parameters[wac_wavelengths[wc]][pc] = float(row[1])
            else:
                # reset parameter counter
                pc = 0
                # and increase the wavelength counter
                wc += 1
                parameters[wac_wavelengths[wc]][pc] = float(row[1])
            pc += 1
    return parameters


if __name__ == "__main__":
    import argparse
    from matplotlib import pyplot as plt
    from visualize import write_tensor, plot_tensor

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="text file with the LRO WAC 7 band resolved Hapke parameters", type=Path)
    args = parser.parse_args()

    if not args.filename.exists():
        raise FileExistsError(" file {str(args.filename)} not found")

    wac_parameters = read_parameters(args.filename)

    tensor = {}
    tensor["version"] = np.array([1, 0], dtype=np.uint8)
    tensor["description"] = np.ones(30, dtype=np.uint8)

    n_angles = 128
    theta = u2theta(np.linspace(0.0, 1.0, n_angles))
    theta_i = np.deg2rad(np.array(range(0,91,5), dtype=np.float64))
    phi_i = np.array([0.0], dtype=np.float32)

    tensor["phi_i"] = phi_i
    tensor["theta_i"] = theta_i.astype(np.float32)

    print("Getting the backscatter reflectance distribution function ...")
    retro = get_backscatter(theta, wac_parameters)
    print("Evaluating the microfacet normal distribution function ...")
    ndf = eval_ndf(theta=theta, backscatter=retro * np.cos(theta))

    def forward(x):
        return x ** (1 / 2)

    def inverse(x):
        return x**2

    plt.xscale("function", functions=(forward, inverse))
    plt.xlim(min(np.rad2deg(theta)), max(np.rad2deg(theta)))
    plt.plot(np.rad2deg(theta), ndf)
    plt.show()
    print("Evaluating the microfacet projected areas ...")
    sigma = projected_area(np.stack([ndf, ndf]), isotropic=True)
    plt.plot(np.rad2deg(theta), sigma[0])
    plt.show()
    print("Evaluating the microfacet visible normal distribution function ...")
    vndf = weighted_vndf(eval_vndf(ndf, sigma[0], theta_i, phi_i))
    plot_tensor(vndf)

    tensor["sigma"] = sigma.astype(dtype=np.float32)
    plt.plot(np.rad2deg(theta), sigma[0])
    plt.show()
    tensor["ndf"] = np.stack([ndf.astype(dtype=np.float32), ndf.astype(dtype=np.float32)])
    tensor["vndf"] = vndf.astype(dtype=np.float32)

    theta_max = np.deg2rad(89.0)
    jacobian = 1
    sample_size = 32

    print("Evaluating the material spectra and luminance...")
    min_wavelength = min(wac_parameters.keys())
    max_wavelength = max(wac_parameters.keys())
    wavelengths_i = np.array(range(min_wavelength, max_wavelength + 1, 5))
    tensor["wavelengths"] = wavelengths_i.astype(dtype=np.float32)
    valid, spectra, luminance = get_spectra(
        wac_parameters,
        wavelengths_i,
        tensor["ndf"],
        vndf,
        tensor["sigma"],
        sample_size,
        theta_i,
        phi_i,
        theta_max,
        jacobian,
    )

    tensor["spectra"] = spectra.astype(dtype=np.float32)
    tensor["luminance"] = luminance.astype(dtype=np.float32)
    tensor["jacobian"] = np.array([jacobian], dtype=np.uint8)
    tensor["valid"] = valid.astype(dtype=np.uint8)
    print("Saving the bsdf file ...")
    out_filename = args.filename.parent / (args.filename.stem + "_spec.bsdf")
    write_tensor(str(out_filename.absolute()), **tensor)
