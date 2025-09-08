
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
numShots=list(np.arange(40)) # a list containing indices of images to actually consider (i.e. [0,2,3] will load first, third and fourth image
loadFull=True #load the full interferogram? (or do you want a selected region...?)

show_plots_flag = True #set to True to see plots of the processing steps, set False to run without interruption
FFT_window_size = 110 # size of the square window in the 2D FFT to isolate the sideband; set carefully - noise vs resolution trade-off!

# settings for fft filtering, legacy from time of small Wollaston aperture in the CALA interferogram required filtering parasitic frequencies
diag_angle = 34 # angle (degs) of the main noise diagonal in the 2D FFT
diag_dist = 300 # diagonal distance to FFT corner cut-out; if set to a number much higher than FFT_window_size, no cut-out will be applied

# manual boundary points for the signal interferogram region (y1,x1),(y2,x2); set to None to select region with mouse initially, and then use same region for all images during an automated run
#boundary_points_sig = [(1639, 804), (1698, 863)]#IR
boundary_points_sig = None

common_path = r"D:\kPAC\2025_09_05PMOPA" #portion of path common to both raw AND processed data
run_path = r'\80mbar_Bdel604_t0' #unique bit of the path to the folder for a specific processing run
sigPath_IR = common_path + run_path + r'\Interferometry2' #rest of the path to the folder containing signal-background pairs of images: 1030 nm (IR)
sigPath_Green = common_path + run_path + r'\Interferometry1' #rest of the path to the folder containing signal-background pairs of images: 515 nm (Green)
saveloc_IR = r'C:\Users\kalos\Documents\kPAC\ChannelAnalysis_CALA_Sept25\channel_analysis_sept25\phase_maps_sept25' + run_path + r'\Interferometry2' #location to save the processed data of this specific processing run
saveloc_Green = r'C:\Users\kalos\Documents\kPAC\ChannelAnalysis_CALA_Sept25\channel_analysis_sept25\phase_maps_sept25' + run_path + r'\Interferometry1' #location to save the processed data of this specific processing run

for sigPath,saveloc in zip([sigPath_IR, sigPath_Green],[saveloc_IR,saveloc_Green]): #loop over both wavelengths
    print('Now processing interferograms in folder: ' + sigPath)
    if not os.path.exists(saveloc):
        os.makedirs(saveloc) #create the folder if it doesn't exist 
    ## LOADING TIFF FILES
    RawInterferograms = PngIntLoader_SK_CALAsept25.PngIntLoader(sigPath,
                                                                sigHeader,
                                                                numShots,
                                                                loadFull)
    # PHASE EXTRACTION
    AvgPhase, DiffCube, DiffCubeNorm, DiffCubeDetr, BgCube = PhaseExtractor_SK.PhaseExtractor(RawInterferograms=RawInterferograms,   # (sig_stack, bg_stack)
                                                                        numShots=numShots,              # or any number
                                                                        saveloc=saveloc,
                                                                        boundary_points_sig=boundary_points_sig,              # or pass saved points to skip clicking
                                                                        show_plots_flag=show_plots_flag,
                                                                        diag_angle_deg=diag_angle,
                                                                        diag_dist=diag_dist,
                                                                        fourier_window_size=FFT_window_size,
                                                                        # pagination + layout
                                                                        pairs_per_fig=10,
                                                                        grid_ncols=5,                          # -> 5 columns x (2*rows) = 10 pairs per page
                                                                        # normalization
                                                                        flip_strategy="median",
                                                                        offset_strategy="median",
                                                                        # background removal
                                                                        bg_method="gaussian",                  # or "median" / "poly"
                                                                        bg_sigma=50.0,
                                                                        # color scaling
                                                                        clim_mode="percentile",
                                                                        clim_value=99.0,
                                                                        # outputs
                                                                        save_avg_plot=True
                                                                    )
