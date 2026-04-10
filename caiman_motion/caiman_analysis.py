import cv2
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import os.path
import logging
import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo

# set CPU threads
try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

# Create Log-File if needed
logfile = None # Replace with a path if you want to log to a file
logger = logging.getLogger('caiman')
# Set to logging.INFO if you want much output, potentially much more output
logger.setLevel(logging.WARNING)
logfmt = logging.Formatter('%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s')
if logfile is not None:
    handler = logging.FileHandler(logfile)
else:
    handler = logging.StreamHandler()
handler.setFormatter(logfmt)
logger.addHandler(handler)

# Function to load video and if needed downsample and play it
def load_video(fname, fframe, downsample_ratio = 0.2, resize=False, play_movies=True, subindices = None):
    # load video, important to set the framerate (fframe) to the correct value, otherwise it will resize the video. 
    # If subindices is set, only the specified frames will be loaded.
    m_orig = cm.load(fname, subindices=subindices, fr=fframe)

    # Downsampling by 0.2 of time to enhance motion correction and its speed.
    # If set to False or if downsample_ratio is set to 1, no downsampling will be performed.
    if resize is True:
        m_orig.resize(1, 1, downsample_ratio) 
    
    # Set play_movies to false if you want to disable play of movies, e.g. for remote-hosted Jupyter environments
    if play_movies:
        m_orig.play(q_max=99.5, fr=fframe, magnification=2)   # play movie (press q to exit)
    return m_orig

# Function to run motion correction on the video
# pw_rigid is messing with the intensity values of the video, needs to be fixed first!
def run_motioncorrect(fname, max_shifts=(12, 12), strides=(96, 96), overlaps=(48, 48),
                         max_deviation_rigid=6, shifts_opencv=True, border_nan='copy', downsample_ratio = 1, nonneg_movie=True, 
                         save_movie=True, pw_rigid = False):
    # start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)
    
    # If pw_rigid is set to False, only rigid motion correction is performed
    if pw_rigid is False:
        print("Running rigid motion correction.")
        # Creation of the MotionCorrect object with the specified parameters
        mc = MotionCorrect(fname, dview=dview, max_shifts=max_shifts,
                    strides=strides, overlaps=overlaps,
                    max_deviation_rigid=max_deviation_rigid, 
                    shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie,
                    border_nan=border_nan, pw_rigid= False)

        # Run motion correction and save the corrected movie in memory if save_movie is set to True
        mc.motion_correct(save_movie=save_movie)

        # Load the corrected movie into memory
        m_corr = cm.load(mc.mmap_file)

        # Stop the cluster server after motion correction is done to free up resources
        cm.stop_server(dview=dview)

        print("Motion correction done. Returning the corrected movie.")
        # Return the corrected movie
        return m_corr
    
    # If pw_rigid is set to True, first rigid motion correction is performed and then piecewise-rigid motion correction is performed 
    # using the rigid template.
    else:
        print("Running both rigid motion correction and piecewise-rigid motion correction using the rigid template.")
        # Creation of the MotionCorrect object with the specified parameters for rigid motion correction.
        # The pw_rigid parameter is set to False for the first rigid motion correction step.
        mc = MotionCorrect(fname, dview=dview, max_shifts=max_shifts,
                    strides=strides, overlaps=overlaps,
                    max_deviation_rigid=max_deviation_rigid, 
                    shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie,
                    border_nan=border_nan, pw_rigid= False)
        
        # Run rigid motion correction and save the corrected movie in memory if save_movie is set to True
        mc.motion_correct(save_movie=save_movie)

        # Load the rigid motion corrected movie into memory
        m_corr = cm.load(mc.mmap_file)

        print("Rigid motion correction done. Now running piecewise-rigid motion correction using the rigid template.")

        # After the first rigid motion correction step, the pw_rigid parameter is set to True for the piecewise-rigid motion correction
        mc.pw_rigid = True

        # Run piecewise-rigid motion correction using the rigid template and save the corrected movie in memory if save_movie is set to True.
        mc.motion_correct(save_movie=True, template=mc.total_template_rig)

        # Load the piecewise-rigid motion corrected movie into memory
        m_els = cm.load(mc.fname_tot_els)

        # OPTIONAL: Downsampling by 0.2 of time to enhance motion correction and its speed
        m_els.resize(1, 1, downsample_ratio)

        # Stop the cluster server after motion correction is done to free up resources
        cm.stop_server(dview=dview)

        print("Piecewise-rigid motion correction done. Returning the rigid motion corrected movie and the piecewise-rigid motion corrected movie.")

        # Return both the rigid motion corrected movie and the piecewise-rigid motion corrected movie
        return m_corr, m_els

# Function to run alignment of two different videos to each other using the template 
# of the first video.
def run_alignment(orig, ref, max_shifts=(12, 12), strides=(96, 96), 
                  overlaps=(48, 48), max_deviation_rigid=6, shifts_opencv=True, 
                  border_nan='copy', nonneg_movie=True, save_movie=True):
    # start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    # Aliging of two different videos to each other using the template of the first video. 
    # This is useful if you have two different videos of the same field of view and want to 
    # align them to each other for further analysis.

    # Set the mean of the ref as template for the alignment of the two videos
    template = ref.mean(axis=0)

    # Creation of the MotionCorrect object with the specified parameters for alignment
    mc = MotionCorrect(orig, dview=dview, max_shifts=max_shifts,
                strides=strides, overlaps=overlaps,
                max_deviation_rigid=max_deviation_rigid, 
                shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie,
                border_nan=border_nan, pw_rigid= False)

    # Run motion correction and save the corrected movie in memory if save_movie is set to True, using the template of the ref video for alignment
    mc.motion_correct(save_movie=save_movie, template=template)

    # Load the aligned movie into memory
    aligned = cm.load(mc.mmap_file)

    # Stop the cluster server after motion correction is done to free up resources
    cm.stop_server(dview=dview)

    print("Alignment of the two videos done. Returning the aligned movie, the motion corrected reference video and the motion corrected original video.")

    # Return the aligned movie
    return aligned
