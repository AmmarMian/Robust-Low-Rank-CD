
##############################################################################
# Functions useful for decomposing SAR using Bell-shaped wavelets
# Authored by Ammar Mian, 11/12/2018
# e-mail: ammar.mian@centralesupelec.fr
##############################################################################
# Copyright 2018 @CentraleSupelec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import numpy as np

def gbellmf(x, a, b, c):
    """
    Generalized Bell function fuzzy membership generator.

    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : float
        Bell function parameter controlling width. See Note for definition.
    b : float
        Bell function parameter controlling slope. See Note for definition.
    c : float
        Bell function parameter controlling center. See Note for definition.

    Returns
    -------
    y : 1d array
        Generalized Bell fuzzy membership function.

    Notes
    -----
    Definition of Generalized Bell function is:

        y(x) = 1 / (1 + abs([x - c] / a) ** [2 * b])

    """
    return 1. / (1. + np.abs((x - c) / a) ** (2 * b))



def decompose_image_wavelet(image, bandwith, range_resolution, azimuth_resolution, center_frequency,
                            R, L, d_1, d_2):
    # Physical parameters
    number_pixels_range = image.shape[1]
    number_pixels_azimuth = image.shape[0]
    c = 3e8 # Speed of light in m/s
    κ_0 = 2*center_frequency/c

    # Construct k_range, k_azimuth vectors
    k_range_vec = κ_0 + (2*bandwith/c)*np.linspace(-0.5,0.5,number_pixels_range)
    k_azimuth_vec = np.linspace(-1/(2*azimuth_resolution),1/(2*azimuth_resolution)-1/(2*number_pixels_azimuth*azimuth_resolution),number_pixels_azimuth)
    KX, KY = np.meshgrid(k_range_vec, k_azimuth_vec)
    𝚱 = np.sqrt(KX**2 + KY**2)
    𝚯 = np.arctan2(KY, KX)

    # Doing decomposition
    spectre_to_decompose = np.fft.fft2(image)
    𝐂 = np.zeros((number_pixels_azimuth, number_pixels_range, R*L), dtype=complex)
    κ_B = 𝚱.max() - 𝚱.min()
    θ_B = 𝚯.max() - 𝚯.min()
    width_κ = κ_B/R
    width_θ = θ_B/L
    for m in range(R):
        for n in range(L):
            center_κ = 𝚱.min() + m*κ_B
            center_θ = 𝚯.min() + n*θ_B

            H_mn_d1d2 = gbellmf(𝚱, width_κ, d_1, center_κ) * \
                        gbellmf(𝚯, width_θ, d_2, center_θ)
            Ψ_mn_d1d2 = spectre_to_decompose * H_mn_d1d2
            𝐂[:,:,m*L + n] = np.fft.ifft2(Ψ_mn_d1d2)

    return 𝐂


