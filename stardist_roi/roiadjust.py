import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.signal import find_peaks, peak_prominences
from scipy import signal
#from tqdm.auto import tqdm
from scipy import stats
#import seaborn as sns
from stardist.models import StarDist2D 
#from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
from stardist import export_imagej_rois
from pathlib import Path
from skimage import measure

# Global list to store the IDs of selected ROIs, which can be accessed and modified across different functions in the script
selected_rois = []

# Function for reading video frames with OpenCV and converting them to a numpy array.
# Here only the first channel is used, as the videos are single channel grayscale.
# By Annemarie Sodmann
def read_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_list = []
    for i in range(frame_count):
        ret, frame = cap.read()
        video_list.append(frame[:,:,0])
    video = np.array(video_list)
    return video

# Function to extract mean fluorescence traces for each ROI across all frames of the video
# By Annemarie Sodmann
def get_traces(video, labels):
    rois = pd.DataFrame()
    for i in np.unique(labels):
        if i > 0:
            trace = video[:,labels==i].mean(axis=1)
            rois['roi_'+str(i)] = trace
    return rois

# Function to normalize the fluorescence traces by calculating the change in fluorescence (DeltaF) 
# relative to the baseline fluorescence (F0).
# By Annemarie Sodmann
def deltanorm(cells):
    labels = cells.columns.values.tolist()  
    F0 = cells.iloc[0:10,:].describe().iloc[1,:].values
    DeltaF = cells.values - F0
    norm = DeltaF/F0
    cells = pd.DataFrame(norm, columns = labels)
    return cells

# Function to apply a Butterworth low-pass filter to the fluorescence traces, which helps to reduce noise and smooth the signal
# By Annemarie Sodmann
def butter_lowpass_filter(data, cutOff, fs, order=4):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    par = signal.butter(order, normalCutoff, btype='low', output="sos")
    y = signal.sosfiltfilt(par, data)
    return y

# Function to load .avi video files from a specified directory, read the video frames into a numpy array, 
# and return both the video data and the path to the video file.
def load_file(path, multi_files=False, file_id=None):
    path = Path(path) # Convert string path to Path object
    path = path.as_posix() # Convert to POSIX format (using forward slashes)
    parent_path = Path(path) # Create a Path object for the parent directory
    
    # Find all .avi files in directory, return a list of Path objects for each file found
    avi_files = list(parent_path.glob("*.avi"))

    # Check if any .avi files were found, if not raise an error
    if len(avi_files) == 0:
        raise ValueError("Warning: No .avi files found in the specified directory.")
        return
    # If multiple .avi files are found, print a warning and use the first one
    elif len(avi_files) > 1 and multi_files == False:
        print("Warning: Multiple .avi files found, using: " + str(avi_files[0]))
        file_id = 0

    # If multiple .avi files are found and multi_files is set to True, print a warning and use file with file_id
    elif len(avi_files) > 1 and multi_files == True:
        if file_id is None:
            raise ValueError("Error: Multiple .avi files found and multi_files is set to True, but no file_id provided.")
            return
        elif file_id < 0 or file_id >= len(avi_files):
            raise ValueError("Error: file_id is out of range. Please provide a valid file_id between 0 and " + str(len(avi_files)-1))
            return
        else:
            print("Warning: Multiple .avi files found, using: " + str(avi_files[file_id]))
            file_id = file_id

    # Use the first .avi file found in the directory as the video path
    video_path = avi_files[file_id]

    # Read the video using the read_video function defined above, which returns a numpy array of video frames
    video = read_video(str(avi_files[file_id]))

    # Print the length of the video in frames for user information
    print("Lenght of video: "+str(video.shape[0])+" Frames")

    # Return the video as a numpy array and the path to the video file for further processing
    return video, video_path

# Function to predict neurons in the video using a pretrained StarDist2D model
def predict_neurons(video_mean, video_path, export=True, prob=0.7):
    # Pretrained Model for Fluorescence stainings, takes 2D single channel pictures
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Preditction of possible cells in the normalized, averaged image "video_mean", here with a probability threshold of 70%
    labels, polygons = model.predict_instances(normalize(video_mean), prob_thresh=prob)

    # Path of ROI export to the same directory as the video, with the same name and suffix "_rois-unfiltered.zip"
    # This is done because NA3 needs them im the same directory for analysis 
    roi_path = video_path.parent / (video_path.stem + "_rois-unfiltered.zip")

    # If export is set to True, all the predicted ROIs are exported in the ImageJ format as a .zip file
    if export == True:
        export_imagej_rois(roi_path, polygons['coord'])
    
    # Return the predicted labels and polygons for further processing
    return labels, polygons

# Function to append the ID of a selected ROI to the global list
def append_roi_selection(roi_id):
    # Access the global list of selected ROIs to modify it within the function
    global selected_rois
    
    # Append the ROI ID to the list of selected ROIs only if it is not already in the list, preventing duplicates
    if roi_id not in selected_rois:
        selected_rois.append(roi_id)

# Function to visualize the selected ROIs on the mean image of the video, highlighting them in a scatter plot
def show_roi_selection(video_mean, labels):
    # Create a figure with a specified size and display the mean image of the video in grayscale
    plt.figure(figsize=(6,6))
    plt.imshow(video_mean, cmap="gray")

    # Plot the selected ROIs on top of the mean image by iterating through the list of selected ROI IDs,
    # finding their coordinates in the labels, and plotting them as scatter points
    for rid in selected_rois:
        y, x = np.where(labels == rid)
        plt.scatter(x, y, s=1)

    plt.title(f"Selected ROIs: \n{selected_rois}")
    plt.axis("off")
    plt.show()

    # Return the list of selected ROIs for further processing if needed
    return selected_rois

# Function to plot the contours of the ROIs on the mean image of the video, 
# with options to specify which ROIs to plot and the color of the contours.
# Also possible to use for subplots by providing an ax argument, if no ax is provided, a new figure and ax will be created for plotting
def plot_roi_contours(video_mean, labels, roi_ids=None, color="red", ax=None):
    # If no specific subfigure is provided for plotting, create a new figure and ax subplot
    if ax is None:
        fig, ax = plt.subplots()

    # Display the mean image of the video in grayscale on the specified ax, with a maximum intensity value for better contrast
    ax.imshow(video_mean, cmap="gray", vmax=20)

    # If no specific ROI IDs are provided, use all unique labels from the labels array that are greater than 0
    if roi_ids is None:
        roi_ids = np.unique(labels)
        roi_ids = roi_ids[roi_ids > 0]

    # Iterate through the specified ROI IDs, create a mask for each ROI, find the contours of the masked area, 
    # and plot the contours on the ax with the specified color and line width
    for rid in roi_ids:
        mask = labels == rid
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:,1], contour[:,0], color=color, linewidth=1.5)

    ax.axis("off")
    # return the subplot with the plotted contours for further use in if needed
    return ax

# Function to plot the contours of the active neurons (selected ROIs) on the mean image of the video.
# Also possible to use for subplots by providing an ax argument, if no ax is provided, a new figure and ax will be created for plotting
def plot_active_neurons(video_mean, labels, ax=None):
    ax = plot_roi_contours(
        video_mean,
        labels,
        roi_ids=selected_rois,
        color="red",
        ax=ax
    )

    # Set the title of the plot to indicate that these are the active neurons, and list the selected ROIs in the title for clarity
    ax.set_title(f"K+ Active Neurons:\n{selected_rois}")

    return ax

# Function to plot the contours of the predicted neurons (all ROIs) on the mean image of the video.
# Also possible to use for subplots by providing an ax argument, if no ax is provided
def plot_predicted_neurons(video_mean, labels, ax=None):
    ax = plot_roi_contours(
        video_mean,
        labels,
        roi_ids=None,   # alle ROIs
        color="yellow",
        ax=ax
    )

    ax.set_title("Predicted Neurons")
    return ax

# Function to export the selected ROIs as ImageJ ROIs in a .zip file, using the coordinates of the predicted polygons for the ROIs
# Running a new prediction with the masked image to get the polygons for the selected ROIs, and then exporting only those polygons 
# as ROIs in ImageJ format (otherwise hard to get them in ImageJ format)
def export_roi_selection(video_mean, video_path, labels, export):
    # Pretrained Model for Fluorescence stainings, takes 2D single channel pictures
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Masking of the mean image to only include the selected ROIs
    mask = np.isin(labels, selected_rois)
    masked_image = video_mean.copy()

    masked_image[~mask] = np.median(video_mean)

    # New prediction with the masked image to get the polygons for the selected ROIs
    labels_new, polygons_new = model.predict_instances(normalize(masked_image))

    # Exporting the polygons of the selected ROIs as ImageJ ROIs in a .zip file, 
    # using the same path and name as the original video with the suffix "_rois_filtered.zip"
    if export == True:
        roi_path = video_path.parent / (video_path.stem + "_rois_filtered.zip")
        export_imagej_rois(roi_path, polygons_new['coord'])

    # Return the new labels and polygons for the selected ROIs, which can be used for further analysis or visualization if needed
    return labels_new, polygons_new

# Function to reset the list of selected ROIs, clearing all previously selected ROIs from the global list
def reset_roi_selection():
    global selected_rois
    selected_rois = []

# Function to compare the original mean image of the video with the masked ROIs and the predicted neurons,
def compare_roi_selection(video_mean, labels):
    # Creating a mask for the selected ROIs by checking which labels in the labels array correspond to the selected ROIs
    mask1 = np.isin(labels, selected_rois)
    
    # Applying the mask to the mean image of the video to create a new image that shows the selected ROIs
    masked_roi = video_mean * mask1

    # Creating a mask for all predicted neurons (all ROIs) by checking which labels in the labels array are greater than 0
    mask2 = labels > 0

    # Applying the mask for all predicted neurons to the mean image of the video to create a new image that shows all predicted neurons
    prediction = video_mean * mask2

    # Creating a figure with 4 subplots to compare the original mean image, the predicted neurons, 
    # the active neurons (selected ROIs), and the masked ROIs
    fig, axes = plt.subplots(1,4, figsize=(10,10))

    # Original Image Mean
    axes[0].imshow(video_mean, cmap="gray", vmax=20)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Predicted Neurons
    plot_predicted_neurons(video_mean, labels, ax=axes[1])

    # Active Neurons (Selected ROIs)
    plot_active_neurons(video_mean, labels, ax=axes[2])

    # Masked ROIs
    axes[3].imshow(masked_roi, cmap="gray", vmax=20)
    axes[3].set_title("Masked ROIs")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
    return

# Main function to analyze the fluorescence traces of the ROIs, identify active ROIs based on significant peaks in the signal, 
# and adding the IDs of the active ROIs to the global list of selected ROIs.
# This function integrates all the previous functions to perform a comprehensive analysis of the video data.
def analyze_roi_traces(video, video_mean, video_path, labels, video_fps, show_graphs=False, prom=10, cutoff=0.1):
    # Extracts the mean fluorescence traces for each ROI across all frames of the video, 
    # resulting in a DataFrame where each column corresponds to an ROI and each row corresponds to a frame
    roi_traces_mean = get_traces(video, labels)

    # Normalizes the extracted mean fluorescence traces
    roi_traces_mean_norm = deltanorm(roi_traces_mean)

    # Counter for the total number of ROIs that are identified as active 
    # based on the presence of significant peaks in their fluorescence traces
    counter = 0

    # Setting the time axis for plotting and analyzing the signal, where the length of the video in seconds is used as the stop value,
    # the starting point of the analysis is set to 0.2 seconds to avoid initial artifacts, and the signal resolution in seconds is 
    # determined by the video FPS (frames per second)
    video_step = 1 / video_fps # Resolution of the signal in seconds, calculated as the inverse of the video FPS
    video_length = (video.shape[0]/video_fps)+video_step # Sum of the video length in seconds and the video step
    time = np.arange(0.2,video_length,video_step) # np.arange(start, stop, step)

    for e,i in enumerate(roi_traces_mean_norm.columns):
        # Normalized mean fluorescence trace for the current ROI i
        signal = roi_traces_mean_norm[i]

        # Applying a Butterworth low-pass filter to the normalized mean fluorescence trace of the current ROI, 
        # with a cutoff frequency of 0.1 Hz and the video FPS as the sampling frequency
        filtered = butter_lowpass_filter(signal,cutoff,video_fps)
        #filtered = wavelet_transform(signal) # optional wavelet-transformation
        
        # Signal differentiation to analyze the rate of change in the fluorescence signal, 
        # which can help to identify significant peaks that indicate neuronal activity.
        dF = np.diff(filtered)/signal[0:50].var()

        # Findet signifikante Peaks im abgeleiteten Signal, hier mit prominence = 10
        # prominence bezieht sich nicht auf die Höhe des Peaks, sondern auf dessen Höhe im Vergleich zu angrenzenden "Tälern"
        
        # find_peaks function from scipy.signal is used to identify peaks in the differentiated signal (dF),
        # with a specified prominence threshold (prom) to filter out insignificant peaks
        peaks, peak_heights = find_peaks(dF, prominence=prom)
        
        # If at least 1 significant peak is found in the differentiated signal of the current ROI, 
        # it is considered active and the following steps are performed:
        if len(peaks) > 0:
            counter += 1

            # The ROI ID is extracted from the column name of the DataFrame, 
            # which is in the format "roi_X", where X is the number of the ROI.
            roi_id = int(i.split("_")[1])
            # Appending the ID of the active ROI to the global list of selected ROIs 
            # using the append_roi_selection function defined earlier
            append_roi_selection(roi_id)

            if show_graphs == True:
                ######## Show graph only, if peaks are found
                # Highlights the current ROI in red
                plt.figure(figsize=(8, 2.5))
                plt.title(i)
                plt.axis("off")
                plt.imshow(render_label((labels==e+1)*1, cmap=(1, 0, 0), img=video_mean, alpha=1))
                plt.show()
            
                # Creates a double plot, on the left the graph of the fluorescence signal and the low-pass filtered signal,
                plt.figure(figsize=(8,2.5))
                plt.subplot(1,2,1)
                plt.plot(time, signal)
                plt.plot(time, filtered)
                plt.ylim(-0.5,1)
                plt.ylabel('\u0394F/F0', fontsize = 16)
                plt.xlabel('Time(s)', fontsize=16)
                plt.legend(['signal', 'low-pass filtered'], fontsize=12)
            
                # on the right the graph of the differentiated signal with the identified peaks highlighted in red Xs
                plt.subplot(1,2,2)
                plt.plot(dF)
                plt.plot(peaks, dF[peaks], "x", c='red')
                plt.ylim(-15,30)
                plt.show()
                ######## Show graph only, if peaks are found
        # If there is no peak found in the differentiated signal of the current ROI, it is not considered active 
        # and not appended to the list of selected ROIs
    print(str(counter)+" counted, positive ROIs")
    return labels_new, polygons_new