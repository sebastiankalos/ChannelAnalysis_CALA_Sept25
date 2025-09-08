# this script loads a set of interferograms from a specified directory,
# and processes them to extract the average phase shift using Fourier techniques

##########################################################################################
# LIBRARIES:

import os
import PngIntLoader_SK_CALAsept25 as PngIntLoader_SK_CALAsept25
import PhaseExtractor_SK as PhaseExtractor_SK
import numpy as np


##########################################################################################
# GLOBAL VARIABLES (need to be set manually)

# Default interferogram parameters
sigHeader='tif' # a string common in the names of all interferograms
numShots=list(np.arange(1)) # a list containing indices of images to actually consider (i.e. [0,2,3] will load first, third and fourth image
loadFull=True #load the full interferogram? (or do you want a selected region...?)

show_plots_flag = True
FFT_window_size = 50

diag_angle = 34 # angle (degs) of the main noise diagonal in the 2D FFT
diag_dist = 50 # diagonal distance to FFT corner cut-out

#boundary_points_sig = [(1675, 913), (1714, 952)]#IR
boundary_points_sig = None


# Relative path to interferogram directories (signal and background)
common_path = r"D:\\OXFORD\\" #portion of path common to both raw AND processed data
unique_path = r"Data\2025-01-30_HOFI_channel_Seb\EnergyScan_t0\IR\100" #unique bit of the path to the signal folder for a specific processing run
sigPath = common_path + unique_path #rest of the path to the folder containing signal-background pairs of images
saveloc = common_path + r"Processed" + unique_path #location to save the processed data of this specific processing run
print(saveloc)
if not os.path.exists(saveloc):
    os.makedirs(saveloc)

#########################################################
## LOADING TIFF FILES
RawInterferograms = PngIntLoader_SK_CALAsept25.PngIntLoader(sigPath,
                                                sigHeader,
                                                numShots,
                                                loadFull)

#########################################################
# PHASE EXTRACTION
AvgPhase = PhaseExtractor_SK.PhaseExtractor(RawInterferograms,
                                            numShots,
                                            saveloc,
                                            boundary_points_sig,
                                            show_plots_flag,
                                            diag_angle,
                                            diag_dist,
                                            fourier_window_size=FFT_window_size)
