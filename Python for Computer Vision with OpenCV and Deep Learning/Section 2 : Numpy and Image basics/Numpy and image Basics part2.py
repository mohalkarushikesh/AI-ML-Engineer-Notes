# Imports 
import cv2
import numpy as np 

# Global variables
drawing = False 
ix, iy = -1, -1
temp_img = None

def draw_shapes(event, x, y, flags, param):
    global drawing, ix, iy, temp_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        # Create a copy of the original image for temporary drawing
        temp_img = img.copy()
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # Restore the original image and draw the current rectangle
            img[:] = temp_img[:]
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
                
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Draw final rectangle
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        temp_img = None

# Load the puppy image
img = cv2.imread('puppy.jpg')

# Check if image was loaded successfully
if img is None:
    print('Creating a black canvas instead...')
    img = np.zeros((512, 512, 3), dtype=np.uint8)
else:
    print(f'Image dimensions: {img.shape[1]}x{img.shape[0]}')

cv2.namedWindow('mydrawing')
cv2.setMouseCallback('mydrawing', draw_shapes)

while True:
    cv2.imshow('mydrawing', img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
