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


def compute_eig(𝐗, args):
    p, N, T = 𝐗.shape
    M = SCM(𝐗.reshape((p,T*N)))
    u, s, vh = np.linalg.svd(M)
    return list(s)

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
    number_of_threads_rows = 8
    number_of_threads_columns = 6

    # Downloading data if needed
    download_uavsar_cd_dataset(path='C:/Users/mian_amm/Dropbox/Thèse/Data/UAVSAR/')
    
    # Reading data using the class
    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   READING     |') 
    print( '|   dataset     |')
    print( '|               |' )  
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    data_class = uavsar_slc_stack_1x1('C:/Users/mian_amm/Dropbox/Thèse/Data/UAVSAR/')
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


    # Parameters
    n_r, n_rc, p, T = image.shape
    windows_mask = np.ones((7,7))
    m_r, m_c = windows_mask.shape
    function_to_compute = compute_eig
    function_args = None

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
    results = sliding_windows_treatment_image_time_series_parallel(image, windows_mask, function_to_compute, 
                    function_args, multi=enable_multi, number_of_threads_rows=number_of_threads_rows,
                    number_of_threads_columns=number_of_threads_columns)
    print("Elpased time: %d s" %(time.time()-t_beginning))
    print('Done')




    # Showing images 
    Amax = 20*np.log10((np.sum(np.abs(image[:,:,:,:])**2, axis=2)/p).max())
    Amin = 20*np.log10((np.sum(np.abs(image[:,:,:,:])**2, axis=2)/p).min())
    for t in range(T):
        Span = np.sum(np.abs(image[:,:,:,t])**2, axis=2)/p
        plt.figure(figsize=(15, 10), dpi=80, facecolor='w')
        plt.pcolormesh(X,Y,20*np.log10(Span), cmap='bone', vmin=Amin, vmax=Amax)
        plt.title(r'Image at $t_%d$' %(t+1))
        plt.xlabel(r'Azimuth (m)')
        plt.ylabel(r'Range (m)')
        plt.colorbar()






