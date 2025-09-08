#this script allows the user to select rectangular regions in an interferogram image
#by Sebastian Kalos, University of Oxford, September 2025

import cv2
import matplotlib.pyplot as plt 
import glob

SIG_COLOR = (222,0,222) #RGB color of the signal rectangle (magenta)
BG_COLOR = (0,222,0) #RGB color of the background rectangle (green)
WINDOW_NAME = "Select region with a mouse. Draw as many times as you wish; only the last take will be saved. Hit enter when happy with the selection"

# function for recording user input through keyboard, and saving coordinates of regions of interest
def click_select(event, x, y, flags, data): 
    image, points = data
    if event == cv2.EVENT_LBUTTONDOWN: # record coordinates after pressing left mouse button
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP: # record coordinates after releasing left mouse button
        points.append((x, y))
        if (int(len(points)/2) % 2) == 0:
            current_color = BG_COLOR
        else:
            current_color = SIG_COLOR
        cv2.imshow(WINDOW_NAME, image)
        cv2.rectangle(image, points[-2], points[-1], current_color, 2)

def show_mouse_select(sigPath,sigHeader):
    filename=glob.glob(sigPath + r"/*" + sigHeader + r"*")[-1] #path to the first image within signal folder
    print(filename)
    orig = cv2.imread(filename,-1) #load the image using OpenCV
    image = orig.copy()

    # first, the image needs to get scaled (12 bit to 16 bit type of thing)
    img_scaled = cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(sigPath+'/scaled_interferogram.tiff', img_scaled)
    cv2.imwrite(sigPath+'/scaled_interferogram.png', img_scaled)
    orig = cv2.imread(sigPath+'/scaled_interferogram.tiff') #load the scaled image using OpenCV
    image = orig.copy()

    cv2.namedWindow(WINDOW_NAME) #create a window for plotting the image
    points = []
    cv2.setMouseCallback(WINDOW_NAME, click_select, (image, points))

    while True: #keep waiting for user entry through keyboard
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(1)
        if key == ord('\r'): break  # pressing enter causes exit


    picked_points=[points[-1],points[-2]] #the last set of saved coordinates denote the BACKGROUND rectangle
    cv2.destroyAllWindows()
    return picked_points

