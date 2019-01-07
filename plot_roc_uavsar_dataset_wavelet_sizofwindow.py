##############################################################################
# ROC plots on real data UAVSAR
# WARNING: will download about 28 Go of data
# Authored by Ammar Mian, 12/11/2018
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
from generic_functions import *
import matplotlib.pyplot as plt
from monte_carlo_tools import *
from multivariate_images_tools import *
from change_detection_functions import *
from read_sar_data import *
from wavelet_functions import *
import os
import time
import seaborn as sns
sns.set_style("darkgrid")
import warnings

def download_uavsar_cd_dataset(path='./Data/'):

    # if directory exists just catch error
    try:
        os.mkdir(path)
    except:
        path

    # Links to UAVSAR datasets
    list_of_files = ['http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090HH_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090HV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090VV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090HH_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090HV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090VV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090HH_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090HV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090VV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090HH_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090HV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090VV_03_BC.ann']

    for file_url in list_of_files:
        if not os.path.exists(path + file_url.split('/')[-1]):
            import wget
            print("File %s not found, downloading it" % file_url.split('/')[-1])
            wget.download(url=file_url, out=path+file_url.split('/')[-1])



if __name__ == '__main__':

    # Activate latex in figures (or not)
    latex_in_figures = True
    if latex_in_figures:
      enable_latex_infigures()

    # Enable parallel processing (or not)
    enable_multi = True
    # These two variables serves to split the original image into sub-images to be treated in parallel
    # In general the optimal parameters are obtained for 
    # number_of_threads_rows*number_of_threads_columns = number of cores on the machine
    number_of_threads_rows = 4
    number_of_threads_columns = 4


    # Parameters
    number_of_points = 100
    windows_size_vector = [3,5,7,9]
    ground_truth_original = np.load('./Data/ground_truth_uavsar_scene1.npy')
    function_to_compute = compute_several_statistics
    # function_args = [[covariance_equality_glrt_gaussian_statistic,
    #                     covariance_equality_glrt_gaussian_statistic_low_rank],
    #                  ['log', (3, 'log')]]
    # statistic_names = ['$\hat{\Lambda}_{\mathcal{G}}$', '$\hat{\Lambda}_{\mathcal{G},\mathrm{LR}}$']
    function_args = [[covariance_equality_glrt_gaussian_statistic,
                        covariance_equality_glrt_gaussian_statistic_low_rank,
                        scale_and_shape_equality_robust_statistic,
                        scale_and_shape_equality_robust_statistic_low_rank],
                     ['log', (3, 'log'), (0.01,10,'log'), (0.01,10,3,'log')]]
    if latex_in_figures:
        statistic_names = ['$\hat{\Lambda}_{\mathcal{G}}$', '$\hat{\Lambda}_{\mathcal{G},\mathrm{LR}}$', '$\hat{\Lambda}_{\mathcal{R}}$','$\hat{\Lambda}_{\mathcal{R},\mathrm{LR}}$']
    else:
        statistic_names = ['Gaussian GLRT', 'Gaussian GLRT Low rank','Robust scale and shape', 'Low rank robust scale and shape']


    # Downloading data if needed
    download_uavsar_cd_dataset(path='../../../../../Data/UAVSAR/')
    
    # Reading data using the class
    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   READING     |') 
    print( '|   dataset     |')
    print( '|               |' )  
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    data_class = uavsar_slc_stack_1x1('../../../../../Data/UAVSAR/')
    data_class.read_data(polarisation=['HH', 'HV', 'VV'], segment=4, crop_indexes=[28891,31251,2891,3491])
    print('Done')


    # Spatial vectors
    center_frequency = 1.26e+9 # GHz, for L Band
    bandwith = float(data_class.meta_data['SanAnd_26524_09014_007_090423_L090HH_03_BC']['Bandwidth']) * 10**6 # Hz
    range_resolution = float(data_class.meta_data['SanAnd_26524_09014_007_090423_L090HH_03_BC']['1x1 SLC Range Pixel Spacing']) # m, for 1x1 slc data
    azimuth_resolution = float(data_class.meta_data['SanAnd_26524_09014_007_090423_L090HH_03_BC']['1x1 SLC Azimuth Pixel Spacing']) # m, for 1x1 slc data
    number_pixels_azimuth, number_pixels_range, p, T = data_class.data.shape
    range_vec = np.linspace(-0.5,0.5,number_pixels_range) * range_resolution * number_pixels_range
    azimuth_vec = np.linspace(-0.5,0.5,number_pixels_azimuth) * azimuth_resolution * number_pixels_azimuth
    Y, X = np.meshgrid(range_vec,azimuth_vec)

    # Decomposition parameters (j=1 always because reasons)
    R = 2
    L = 2
    d_1 = 10
    d_2 = 10

    # Wavelet decomposition of the time series
    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   Wavelet     |') 
    print( '| decomposition |')
    print( '|               |' )  
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    image = np.zeros((number_pixels_azimuth, number_pixels_range, p*R*L, T), dtype=complex)
    for t in range(T):
        for i_p in range(p):
            image_temp = decompose_image_wavelet(data_class.data[:,:,i_p,t], bandwith, range_resolution, azimuth_resolution, center_frequency,
                                    R, L, d_1, d_2)
            image[:,:,i_p*R*L:(i_p+1)*R*L, t] = image_temp
    print('Done')
    n_r, n_rc, p, T = image.shape


    # Computing statistics on both images
    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   COMPUTING   |') 
    print( '|   in progress |')
    print( '|               |' )  
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    t_beginning = time.time()

    pfa_array = np.zeros((number_of_points, len(function_args[0]), len(windows_size_vector)))
    pd_array = np.zeros((number_of_points, len(function_args[0]), len(windows_size_vector)))
    t = time.time()
    for i_w, w in enumerate(windows_size_vector):
        print('Computing simulation %d of %d'%(i_w+1, len(windows_size_vector)))
        windows_mask = np.ones((w,w))
        m_r, m_c = windows_mask.shape
        results = sliding_windows_treatment_image_time_series_parallel(image, windows_mask, function_to_compute, 
                        function_args, multi=enable_multi, number_of_threads_rows=number_of_threads_rows,
                        number_of_threads_columns=number_of_threads_columns)

        # Computing ROC curves
        ground_truth = ground_truth_original[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)]
        for i_s, statistic in enumerate(statistic_names):

            # Sorting values of statistic
            λ_vec = np.sort(vec(results[:,:,i_s]), axis=0)
            λ_vec = λ_vec[np.logical_not(np.isinf(λ_vec))]

            # Selectionning number_of_points values from beginning to end
            indices_λ = np.floor(np.logspace(0, np.log10(len(λ_vec)-1), num=number_of_points))
            λ_vec = np.flip(λ_vec, axis=0)
            λ_vec = λ_vec[indices_λ.astype(int)]

            # Thresholding and summing for each value
            for i_λ, λ in enumerate(λ_vec):
                good_detection = (results[:,:,i_s] >= λ) * ground_truth
                false_alarms = (results[:,:,i_s] >= λ) * np.logical_not(ground_truth)
                pd_array[i_λ, i_s, i_w] = good_detection.sum() / (ground_truth==1).sum()
                pfa_array[i_λ, i_s, i_w] = false_alarms.sum() / (ground_truth==0).sum()

        print('Time on simulation: %d s'  %(time.time()-t) )
        t = time.time() 

    print('Done')

    markers = ['o', 's', 'd', '*', '+']
    pfa = 0.05
    plt.figure(figsize=(6, 4), dpi=80, facecolor='w')
    for i_s, statistic in enumerate(statistic_names):
        pd_statistic = np.zeros(len(windows_size_vector))
        for i_w, w in enumerate(windows_size_vector):
            index = np.argmin(np.abs(pfa_array[:,i_s,i_w]-pfa))
            pd_statistic[i_w] = pd_array[index,i_s,i_w]
        plt.plot(windows_size_vector, pd_statistic, linestyle='--', label=statistic,
        marker=markers[i_s])
    plt.legend()
    plt.show()


    print("Elpased time: %d s" %(time.time()-t_beginning))
    




