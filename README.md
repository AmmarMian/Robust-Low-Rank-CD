# Wavelet packet adapted to High-resolution SAR images

The work here correspond to a code for change detection for SAr images exploiting Low-rank structure and non-Gaussian assumptions.
It corresponds to a simulation done for a proposed IGARSS paper.

## Files' organisation

This folder is organised as follows:
- **change_detection_functions.py**: Contain the codes for every detector compared.
- **compute_distribution_eigenvalues_image_wevelet.py**: Code used for plotting the mean of eigenvalues over the image Time Series.
- **plot_roc_uavsar_dataset_wavelet.py**: ROC on real dataset by exploiting wavelet transform and statistics. Warning: Will download 28Go of data is not present.
- **plot_roc_uavsar_dataset_wavelet_sizofwindow.py**: Same as **plot_roc_uavsar_dataset_wavelet.py** but vary the size of window


## Requirements for Python
	The code provided was developped and tested using Python 3.7. The following packages must be installed 
	in order to run everything smoothly:
	- Scipy/numpy
	- matplotlib
	- seaborn
	- wget (if you do not have the UAVSAR data yet)
	- tqdm

The code use parallel processing which can be disabled by putting the boolean enable_multi to False in each file.
The figures can be plotted using Latex formatting by putting the boolean latex_in_figures to True in each file (must have a latex distribution installed).

Dataset available at https://uavsar.jpl.nasa.gov/.

## Credits
**Author:** Ammar Mian, Ph.d student at SONDRA, CentraleSupélec
 - **E-mail:** ammar.mian@centralesupelec.fr
 - **Web:** https://ammarmian.github.io/
 
 Acknowledgements to:
 - [**Arnaud Breloy**](https://www.researchgate.net/profile/Arnaud_Breloy), LEME, Université Paris Nanterre
 - [**Guillaume Ginolhac**](https://www.listic.univ-smb.fr/presentation/membres/enseignants-chercheurs/guillaume-ginolhac/), LISTIC, Université Savoie Mont-Blanc
 - [**Jean-Philippe Ovarlez**](http://www.jeanphilippeovarlez.com/), DEMR, ONERA , Université Paris-Saclay  & SONDRA, CentraleSupélec

 
## Copyright
 
 Copyright 2019 @CentraleSupelec

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.