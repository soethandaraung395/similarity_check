import os
import cv2

def crop_black_border(image_path, output_folder):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image
    cropped_image = image[y:y+h, x:x+w]
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save cropped image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, cropped_image)

def crop_black_borders_in_folder(input_folder, output_folder):
    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            crop_black_border(image_path, output_folder)

# Example usage
input_folder = '/Users/soethandara/Desktop/Flutter/UITest/similarity_check/Original/'
output_folder = '/Users/soethandara/Desktop/Flutter/UITest/exercise/exercise1/Crop/'
crop_black_borders_in_folder(input_folder, output_folder)
