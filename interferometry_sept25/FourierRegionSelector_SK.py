#This is a script to allow user to select a rectangular region in a Fourier image
#The script will then find the peak in that region and center a square around it
#by Sebastian Kalos, University of Oxford, September 2025

import cv2
import matplotlib.pyplot as plt 
from scipy.ndimage import maximum_filter
import numpy as np

REC_COLOR = (222,0,222) #RGB color of the drawn rectangle (magenta)
GREEN_COLOR=(170,255,0) #RGB color of the automatically centered square (bright green)
WINDOW_NAME = "Select Fourier region with a mouse. Draw as many times as you wish; only the last drawn rectangle will be saved. Hit enter when happy with the selection"

def find_peak_and_construct_square(image, coordinates, fourier_window_size):
    # Extract the specified rectangular region from the image
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    region = image[y1:y2, x1:x2]

    # Find the coordinates of the peak in the specified region
    peak_coords = np.unravel_index(np.argmax(region), region.shape)
    peak_coords = (peak_coords[1] + x1, peak_coords[0] + y1)  # Adjust to global coordinates

    # Determine the side length of the square (odd number)
    side_length = fourier_window_size
    side_length = side_length if side_length % 2 == 1 else side_length - 1

    # Calculate the coordinates of the square's top-left corner
    square_x1 = peak_coords[0] - side_length // 2
    square_y1 = peak_coords[1] - side_length // 2

    # Ensure the square is within the image bounds
    square_x1 = max(0, square_x1)
    square_y1 = max(0, square_y1)

    # Calculate the coordinates of the square's bottom-right corner
    square_x2 = square_x1 + side_length
    square_y2 = square_y1 + side_length

    # Draw a red dot at the identified peak
    cv2.circle(image, peak_coords, radius=3, color=(10,255,255), thickness=-1)

    # Return the coordinates of the four corners of the shifted square
    return (square_x1, square_y1), (square_x2, square_y2)

# function for recording user input through keyboard, and saving coordinates of regions of interest
def click_select(event, x, y, flags, data): 
    image, points, fourier_window_size = data
    image_copy = image.copy()
    if event == cv2.EVENT_LBUTTONDOWN: # record coordinates after pressing left mouse button
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP: # record coordinates after releasing left mouse button
        points.append((x, y))

        # Get image dimensions
        height, width = image.shape[:2]
        # Define zooming window size (adjust based on screen resolution)
        zoom_width, zoom_height = 1500, 1500  # Size of the display window
        # Define the center region to focus on
        center_x, center_y = width // 2, height // 2
        # Create a window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # Set window size (simulating zoom)
        cv2.resizeWindow(WINDOW_NAME, zoom_width, zoom_height)
        # Move the window to focus on the center
        #cv2.moveWindow(WINDOW_NAME, center_x - zoom_width // 2, center_y - zoom_height // 2)
        # Display the original image (zoomed in naturally)
        cv2.imshow(WINDOW_NAME, image)

        draw_points = [points[-2], points[-1]] # the last set of saved coordinates denote the rectangle

        cv2.rectangle(image, draw_points[0], draw_points[1], REC_COLOR, 2) # draw a rectangle around the picked points
        shifted_points = find_peak_and_construct_square(image_copy, draw_points, fourier_window_size)
        cv2.rectangle(image, shifted_points[0], shifted_points[1], GREEN_COLOR, 2) # draw a rectangle around the shifted points

def show_mouse_select(fourier_image, fourier_window_size):
    orig = cv2.imread(fourier_image)  # load the image using OpenCV
    image = orig.copy()

    # Get image dimensions
    height, width = image.shape[:2]

    # Define zooming window size (adjust based on screen resolution)
    zoom_width, zoom_height = 1500, 1500  # Size of the display window

    # Create a window but don't forcefully move it
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, zoom_width, zoom_height)

    points = []  # initialize list for saving cursor coordinates in the image
    cv2.setMouseCallback(WINDOW_NAME, click_select, (image, points, fourier_window_size))

    while True:  # keep waiting for user entry through keyboard/mouse
        cv2.imshow(WINDOW_NAME, image)  # keep showing the image
        key = cv2.waitKey(1)
        if key == ord('\r'): 
            break  # press enter to exit

    cv2.destroyAllWindows()

    draw_points = [points[-2], points[-1]]  # the last set of saved coordinates denote the rectangle
    shifted_points = find_peak_and_construct_square(orig, draw_points, fourier_window_size)

    return shifted_points

