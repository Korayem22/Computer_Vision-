import cv2
import numpy as np

# Load video file
video = cv2.VideoCapture("mouthwash.avi")

# Read first frame to initialize tracking
_, first_frame = video.read()

# Define initial bounding box coordinates (x, y) and dimensions (width, height)
x = 300       # Top-left x-coordinate of bounding box
y = 305       # Top-left y-coordinate of bounding box
width = 100   # Width of bounding box
height = 115  # Height of bounding box

# Student code : Extract Region of Interest (ROI) from first frame
# Hint: Use slicing with y, y+height, x, and x+width
roi = first_frame[y:y+height, x:x+width]
#convert ndarray to cv2 image
# Student code : Convert ROI from BGR to HSV color space (better for color-based tracking)
# Hint: Use cv2.cvtColor with cv2.COLOR_BGR2HSV
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

# Student code : Calculate histogram of Hue channel (channel 0) in ROI
# Hint: Use cv2.calcHist with channels=[0], histSize=[180], and ranges=[0, 180]
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

# Student code : Normalize histogram to 0-255 range for better comparison
# Hint: Use cv2.normalize with norm_type=cv2.NORM_MINMAX and ranges 0 to 255
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Student code : Define termination criteria for meanShift algorithm:
# Hint: Use (criteria_type, max_iter, min_shift)
# - Stop after 10 iterations (cv2.TERM_CRITERIA_COUNT)
# - Or if the centroid moves less than 1 pixel (cv2.TERM_CRITERIA_EPS)
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Main tracking loop
while True:
    # Read next frame from video
    _, frame = video.read()
    
    # Student code : Convert current frame to HSV color space
    # Hint: Use cv2.cvtColor as before
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate back projection using ROI histogram
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    
    # Student code : Apply meanShift to find the new location of the ROI
    # Hint: Use cv2.meanShift with mask and term_criteria
    _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
    x, y, w, h = track_window
    
    # Student code : Draw bounding box around tracked object
    # Hint: Use cv2.rectangle with points (x, y) and (x+w, y+h)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the probability mask and tracking result
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    
    # Exit if ESC key is pressed
    key = cv2.waitKey(60)
    if key == 27:
        break

# Clean up
video.release()
cv2.destroyAllWindows()
