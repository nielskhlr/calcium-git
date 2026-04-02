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

# Create Log-File
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

# Function to load video, downsample it and play it (if play_movies is set to True)
def load_video(fname, fframe, downsample_ratio = 0.2, resize=False, play_movies=True, subindices = None):
    m_orig = cm.load(fname, subindices=subindices, fr=fframe) # load video
    if resize is True:
        m_orig.resize(1, 1, downsample_ratio) # Downsampling by 0.2 of time to enhance motion correction
    # Set play_movies to false if you want to disable play of movies, e.g. for remote-hosted Jupyter environments
    if play_movies:
        m_orig.play(q_max=99.5, fr=fframe, magnification=2)   # play movie (press q to exit)
    return m_orig

# Function to run motion correction on the video. If pw_rigid is set to False, only rigid motion correction is performed. 
# If pw_rigid is set to True, first rigid motion correction is performed and then piecewise-rigid motion correction is performed 
# using the rigid template.
def run_motioncorrect(fname, max_shifts=(12, 12), strides=(96, 96), overlaps=(48, 48),
                         max_deviation_rigid=6, shifts_opencv=True, border_nan='copy', downsample_ratio = 1, nonneg_movie=True, save_movie=True, pw_rigid = False):
    # start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)
    
    # run motion correction
    if pw_rigid is False:
        mc = MotionCorrect(fname, dview=dview, max_shifts=max_shifts,
                    strides=strides, overlaps=overlaps,
                    max_deviation_rigid=max_deviation_rigid, 
                    shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie,
                    border_nan=border_nan, pw_rigid= False)

        mc.motion_correct(save_movie=save_movie)
        m_corr = cm.load(mc.mmap_file)

        cm.stop_server(dview=dview) # stop the server

        return m_corr
    # If pw_rigid is set to True, first rigid motion correction is performed and then piecewise-rigid motion correction is performed
    else:
        mc = MotionCorrect(fname, dview=dview, max_shifts=max_shifts,
                    strides=strides, overlaps=overlaps,
                    max_deviation_rigid=max_deviation_rigid, 
                    shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie,
                    border_nan=border_nan, pw_rigid= False)
        
        mc.motion_correct(save_movie=save_movie)
        m_corr = cm.load(mc.mmap_file)

        mc.pw_rigid = True

        mc.motion_correct(save_movie=True, template=mc.total_template_rig)
        m_els = cm.load(mc.fname_tot_els)
        m_els.resize(1, 1, downsample_ratio)

        cm.stop_server(dview=dview) # stop the server

        return m_corr, m_els

