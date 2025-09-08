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
numShots=list(np.arange(25)) # a list containing indices of images to actually consider (i.e. [0,2,3] will load first, third and fourth image
loadFull=True #load the full interferogram? (or do you want a selected region...?)

show_plots_flag = False
FFT_window_size = 60

diag_angle = 34 # angle (degs) of the main noise diagonal in the 2D FFT
diag_dist = 50 # diagonal distance to FFT corner cut-out

boundary_points_sig = [(1639, 804), (1698, 863)]#IR
#boundary_points_sig = None

common_path = r"D:\OXFORD\\" #portion of path common to both raw AND processed data

for mbar in [100,50]:
    for t_ns in [1,2,3,4]: #loop over all the time delays 
        for attenuator_setting in [35,50,70,85,100]: #loop over all the attenuator settings

            unique_path = r"Data\2025-02-24_HOFI_channel_Seb\\"+str(mbar)+"mbar\\"+str(t_ns)+"ns\\"+str(attenuator_setting) #unique bit of the path to the signal folder for a specific processing run
            sigPath = common_path + unique_path #rest of the path to the folder containing signal-background pairs of images
            saveloc = common_path + r"Processed" + unique_path #location to save the processed data of this specific processing run
            print(saveloc) #print the save location to the console
            if not os.path.exists(saveloc):
                os.makedirs(saveloc) #create the folder if it doesn't exist 

            ## LOADING TIFF FILES
            RawInterferograms = PngIntLoader_SK_CALAsept25.PngIntLoader(sigPath,
                                                            sigHeader,
                                                            numShots,
                                                            loadFull)

            # PHASE EXTRACTION
            AvgPhase = PhaseExtractor_SK.PhaseExtractor(RawInterferograms,
                                                        numShots,
                                                        saveloc,
                                                        boundary_points_sig,
                                                        show_plots_flag,
                                                        diag_angle,
                                                        diag_dist,
                                                        fourier_window_size=FFT_window_size)
