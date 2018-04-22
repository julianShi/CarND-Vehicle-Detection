import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from moviepy.editor import VideoFileClip
import cv2
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.externals import joblib


svc=joblib.load(open('svc.pkl', 'rb'))
X_scaler=joblib.load(open('X_scaler.pkl', 'rb'))

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

def pipeline_window_test(image):
	return pipeline_window(image,svc,X_scaler)

def pipeline_label_test(image):
	return pipeline_label(image,svc,X_scaler)

video_file = 'project_video.mp4'
# clip1 = VideoFileClip(video_file).get_frame(10)
# window_img = pipeline_window_test(clip1)
# cv2.imwrite('output/project_video.jpg', window_img)

subclip = VideoFileClip(video_file).subclip(10,11)
white_clip = subclip.fl_image(pipeline_label_test)
white_clip.write_videofile('output/'+video_file.split('/')[-1], audio=False)

