import cv2
import numpy as np
# import matplotlib.pyplot as plt

def canny(image):
  # Step1: convert image to grayscale (processing 1 channel in GS is faster than 3 channels in color)
  gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  # Step2: Reduce Noise (Gaussian Filter -> filter out noise that affects edge detection)
  blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0) # 5x5 kernel with deviation of 0

  # Step 3: Apply Canny method to simplify edge detection, select strongest gradient in the image
  canny_img = cv2.Canny(blur_img, 50, 150) # with low_threshold and high_threshold. Recommend to use 1,2 or 1,3 means 50,100 or 50, 150
  return canny_img

def region_of_interest(image):
  """
  This function is to create a cropped image that focus on the region of interest, which is lanes/lines
  """
  height = image.shape[0]

  # Step 1: plot 3 dots that make a lane area
  polygons = np.array([[(200, height), (1100, height), (550, 250)]])

  # Step 2: create a same image size with black (0s)
  mask = np.zeros_like(image)

  # Step 3: fill the mask image with polygon in white (255)
  cv2.fillPoly(mask, polygons, 255)

  # Step 4: apply bitwise AND on image with mask image
  masked_image = cv2.bitwise_and(image, mask)
  return masked_image

def make_points(image, line):
  slope, intercept = line
  y1 = int(image.shape[0]) # bottom of the image
  y2 = int(y1 * 3 / 5) # slightly lower than the middle
  x1 = int((y1 - intercept)/slope)
  x2 = int((y2 - intercept)/slope)
  return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
  """
  This function is to draw a optimal single lines based on averaged left and right lines, apply on image
  """
  left_fit = []
  right_fit = []
  if lines is None:
    return None
  for line in lines:
    for x1, y1, x2, y2 in line:
      fit = np.polyfit((x1, x2), (y1, y2), 1)
      slope = fit[0]
      intercept = fit[1]
      if slope < 0: # this is left lane because x increase and y decrease which make slope negative
        left_fit.append((slope, intercept))
      else: # this is right lane because x and y both decrease which make slope positive
        right_fit.append((slope, intercept))

  # create single lines for both left and right lines based on averages
  if left_fit:
    left_line = make_points(image, np.average(left_fit, axis=0))
  else:
    left_line = np.array([[0, 0, 0, 0]])
  if right_fit:
    right_line = make_points(image, np.average(right_fit, axis=0))
  else:
    right_line = np.array([[0, 0, 0, 0]])

  # return averaged lines together as array
  return np.array([left_line, right_line])

def display_lines(image, lines):
  """
  This funtion is to draw lines based on the cropped_image that focuses on region of interest
  """
  mask_image = np.zeros_like(image)
  if lines is None:
    return None
  if lines is not None:
    for line in lines:
    #   x1, y1, x2, y2 = line.reshape(4) # don not need to reshape because averaged_lines method already unpacked params into 1D array
      for x1, y1, x2, y2 in line:
        cv2.line(mask_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
  return mask_image

# ori_image = cv2.imread('../media/test_image.jpg')
# copy_image = np.copy(ori_image)
# canny = canny(copy_image)
# cropped_canny = region_of_interest(canny)
# created_lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# line_image = display_lines(copy_image, created_lines)
# combo_image = cv2.addWeighted(copy_image, 0.8, line_image, 1, 1)
# cv2.imshow('result', combo_image)
# cv2.waitKey(0)

# plt.imshow(canny)
# plt.show()

cap = cv2.VideoCapture("../media/test_video.mp4")
copy_image = None
while cap.isOpened():
  _, frame = cap.read()
  # print(frame)
  if frame is not None:
    copy_image = np.copy(frame)
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    created_lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, created_lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    print("Invalid frame")
    # cv2.imshow("result", copy_image)
    # cap.grab()
    # continue
    break
cap.release()
cv2.destroyAllWindows()
