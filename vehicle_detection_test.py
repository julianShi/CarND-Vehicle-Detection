import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from moviepy.editor import VideoFileClip
import cv2
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
# import pickle
from sklearn.externals import joblib

svc=joblib.load(open('svc.pkl', 'rb'))
X_scaler=joblib.load(open('X_scaler.pkl', 'rb'))

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 1 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (8, 8) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

def pipeline_image_test(image):
    return pipeline_image(image,svc,X_scaler,color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

# buffer to use the labels detected in the previous image
hot_windows_previous=[]
def pipeline_video(image):
    global hot_windows_previous
    draw_image = image
    hot_windows = pipeline_window(image,svc,X_scaler,color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows+hot_windows_previous)

    # Save the current hot windows for the use of the next frame
    hot_windows_previous = hot_windows

    # Apply threshold to help remove false positives
    heat_threshold = apply_threshold(heat,8)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat_threshold, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_image, labels)

    return draw_img

video_file = 'project_video.mp4'
# for i in range(20,40,5):
#     clip1 = VideoFileClip('project_video.mp4').get_frame(i)
#     window_img, heat, img_label = pipeline_image_test(clip1)
#     cv2.imwrite('examples/project_video_window_'+str(i)+'.jpg',window_img)
#     cv2.imwrite('examples/project_video_heat_'+str(i)+'.jpg',heat*20)
#     cv2.imwrite('examples/project_video_label_'+str(i)+'.jpg',img_label)

subclip = VideoFileClip(video_file)
white_clip = subclip.fl_image(pipeline_video)
white_clip.write_videofile('output/'+video_file.split('/')[-1], audio=False)

