#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


# Define the lower and upper bounds for each color (in HSV color space)
color_ranges = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'green': [(35, 100, 100), (85, 255, 255)],
    'blue': [(100, 100, 100), (140, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'orange': [(10, 100, 100), (25, 255, 255)],
}


# In[3]:


# Define colors and corresponding BGR values
color_map = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
}


# In[4]:


def recognize_colors(frame):
    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Initialize an empty list to store recognized colors
    recognized_colors = []

    # Iterate over the defined color ranges
    for color_name, (lower, upper) in color_ranges.items():
        # Create a mask to extract the color from the frame
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))

        # Count the number of non-zero pixels (colored pixels) in the mask
        pixel_count = cv2.countNonZero(mask)

        # If a sufficient number of pixels are detected, consider it as the color
        if pixel_count > 1000:
            recognized_colors.append(color_name)
            # Find the center of the colored region and label it
            moments = cv2.moments(mask)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                cv2.putText(
                    frame, color_name, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )

    return frame, recognized_colors



# In[5]:


# Capture video from the webcam (change 0 to the video file path if using a file)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Recognize colors and label objects
    labeled_frame, recognized_colors = recognize_colors(frame)

    # Display the original frame with recognized colors
    cv2.imshow('Color Recognition', labeled_frame)

    # Print recognized colors
    if recognized_colors:
        print("Recognized Colors:", recognized_colors)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




