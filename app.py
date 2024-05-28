from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import csv
import cv2
from PIL import Image
import numpy as np
import pytesseract

app = Flask(__name__)

# Define the path to exercise and answer folders
exercise_folder = "exercise_folder"
answer_folder = "answer_folder"
result_folder = "result_folder"
difference_folder = "difference_folder"

def highlight_image_difference(image1, image2_path, output_path):
    org_img = image1
    correct_image = Image.open(image1)
    correct_width, correct_height = correct_image.size
    # Read images
    original_image1 = cv2.imread(image1)
    
    student_image_files = [f for f in os.listdir(image2_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    print (image2_path)
    for student_image_name in student_image_files:
        student_image = Image.open(image2_path+"/"+student_image_name)
        student_width, student_height = student_image.size
        
        # Compare image sizes
        if (student_width, student_height) == (correct_width, correct_height):
            print ("same resolution")
                # Construct full path to the student image
            student_image_path = os.path.join(image2_path, student_image_name)
            
            print ('Student Images after comparing the image sizes: ' + student_image_path)
            image2 = cv2.imread(student_image_path)
            
            # Compute absolute difference between the images
            difference = cv2.absdiff(original_image1, image2)
            
        else:
             # Read images
            image1 = cv2.imread(org_img, cv2.IMREAD_COLOR)
            image2 = cv2.imread(image2_path+"/"+student_image_name, cv2.IMREAD_COLOR)
            target_width = 700
            target_height = 550
            
            # Resize both images to the specified resolution
            resize_image1 = resize_image(image1, target_width, target_height)
            resize_student_image_path = resize_image(image2, target_width, target_height)
            cv2.imwrite(image2_path+"/"+student_image_name, resize_student_image_path)
            #cv2.imwrite(os.path.join(os.path.dirname(image2_path), image2_path, student_image_name), student_image_path)
            #print (student_image_name)
            #print (image2_path)
        
            # Construct full path to the student image
            resize_student_image_path = os.path.join(image2_path, student_image_name)
            
            print (resize_student_image_path)
            image2 = cv2.imread(resize_student_image_path)
            
            # Compute absolute difference between the images
            height, width, channels = resize_image1.shape

            # Print the dimensions
            print("Correct Image size: {} x {} pixels".format(width, height))
            print("Number of channels: ", channels)
            
            # Compute absolute difference between the images
            height, width, channels = image2.shape

            # Print the dimensions
            print("Student Image size: {} x {} pixels".format(width, height))
            print("Number of channels: ", channels)
            
            difference = cv2.absdiff(resize_image1, image2)

        # Convert difference image to grayscale
        gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

        # Threshold the difference image to obtain binary mask
        _, mask = cv2.threshold(gray_difference, 30, 255, cv2.THRESH_BINARY)

        # Dilate the mask to make the highlighted regions more visible
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Apply the mask to the original image
        highlighted_image = image2.copy()
        highlighted_image[mask != 0] = [0, 0, 255]  # Highlight differences in red (BGR format)

        # Blend the images to create a semi-transparent overlay
        alpha = 0.3  # Adjust the transparency level as needed
        result = cv2.addWeighted(image2, 1-alpha, highlighted_image, alpha, 0)

        print (student_image_name)
        print (output_path)
        output_path_name = output_path+student_image_name
        # Write the resulting image with highlighted differences to output path
        cv2.imwrite(output_path_name, result)
            

def check_image_size_and_similarity(correct_image_path, student_images_folder, output_csv_path):
    # Open the correct image
    correct_image = Image.open(correct_image_path)
    correct_width, correct_height = correct_image.size

    # Get a list of image files in the folder
    student_image_files = [f for f in os.listdir(student_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Initialize list to store similarity results
    similarity_results = []

    for student_image_name in student_image_files:
        # Construct full path to the student image
        student_image_path = os.path.join(student_images_folder, student_image_name)

        # Open the student image
        student_image = Image.open(student_image_path)
        student_width, student_height = student_image.size
        threshold = 95.00
        # Compare image sizes
        if (student_width, student_height) == (correct_width, correct_height):
            # Calculate similarity using SIFT
            similarity_percentage = calculate_similarity_by_SIFT(correct_image_path, student_image_path)
            similarity_results.append({'Image': student_image_name, 'Similarity (%)': similarity_percentage, 'Remark': similarity_percentage})
        else:
            # Calculate similarity using ORB
            similarity_percentage = calculate_similarity_by_ORB(correct_image_path, student_image_path)
            print(f"{student_image_name}: Incorrect size")
            if similarity_percentage>=threshold:
                similarity_results.append({'Image': student_image_name, 'Similarity (%)': similarity_percentage, 'Remark': 100.00})
            else:
                similarity_results.append({'Image': student_image_name, 'Similarity (%)': similarity_percentage, 'Remark': similarity_percentage})
    # Sort similarity results in descending order based on similarity percentage
    similarity_results.sort(key=lambda x: x['Similarity (%)'], reverse=True)

    # Output CSV file path
    #output_csv_path = os.path.join(student_images_folder, 'similarity_results.csv')

    # Write sorted results to CSV file with two decimal places
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'Similarity (%)', "Remark"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in similarity_results:
            writer.writerow({'Image': result['Image'], 'Similarity (%)': f"{result['Similarity (%)']:.2f}", 'Remark': f"{result['Remark']:.2f}"})

    print("Similarity comparison completed. Results saved in:", output_csv_path)

def calculate_similarity_by_SIFT(image1_path, image2_path):
    # Read images
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    
    #avg_height = (image1.shape[0] + image2.shape[0]) // 2
    #avg_width = (image1.shape[1] + image2.shape[1]) // 2

    # Resize both images to the average dimensions
    #image1_resized = cv2.resize(image1, (avg_width, avg_height))
    #image2_resized = cv2.resize(image2, (avg_width, avg_height))
    
    # Save the resized images
    #cv2.imwrite(os.path.join(os.path.dirname(image1_path), 'resize', 'resized_image1.jpg'), image1_resized)
    #cv2.imwrite(os.path.join(os.path.dirname(image2_path), 'resize', 'resized_image2.jpg'), image2_resized)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher_create()

    # Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Calculate similarity percentage
    similarity = len(good_matches) / max(len(keypoints1), len(keypoints2)) * 100

    return similarity

def resize_image(image, target_width, target_height):
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def calculate_similarity_by_ORB(image1_path, image2_path):
     # Read images
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    target_width = 700
    target_height = 550
    
    # Resize both images to the specified resolution
    image1_resized = resize_image(image1, target_width, target_height)
    image2_resized = resize_image(image2, target_width, target_height)
    
    # Save the resized images
    #cv2.imwrite('/Users/soethandara/Desktop/Flutter/UITest/exercise/exercise1/resize/'+count1+'.jpg', image1_resized)
    #cv2.imwrite('/Users/soethandara/Desktop/Flutter/UITest/exercise/exercise1/resize/'+count2+'.jpg', image2_resized)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(image1_resized, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2_resized, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity percentage
    similarity = len(matches) / max(len(keypoints1), len(keypoints2)) * 100

    return similarity

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print ("Soe")
        global exercis_nam
        selected_exercise = exercis_nam
        print (selected_exercise)
        similar = "Soe"
        exercise_folders = os.listdir(exercise_folder)
        
        exercise_path = os.path.join(exercise_folder, selected_exercise)
        answer_path = os.path.join(answer_folder, f'answer_{selected_exercise.split("_")[1]}')
        print (exercise_path)
        print (answer_path)
        exercise_image = os.path.join(exercise_path, f'{selected_exercise}.png')
        # Get answer images
        answer_images = [os.path.join(answer_path, image) for image in os.listdir(answer_path)]
    
    else:
        # Get the list of exercise folders
        exercise_folders = os.listdir(exercise_folder)

        # Get the currently selected exercise from the combo box
        selected_exercise = request.args.get('exercise')

        # If no exercise is selected, default to the first one in the list
        if not selected_exercise and exercise_folders:
            selected_exercise = exercise_folders[0]

        # Fetch images for the selected exercise
        exercise_path = os.path.join(exercise_folder, selected_exercise)
        answer_path = os.path.join(answer_folder, f'answer_{selected_exercise.split("_")[1]}')

        # Get exercise image
        exercise_image = os.path.join(exercise_path, f'{selected_exercise}.png')

        # Get answer images
        answer_images = [os.path.join(answer_path, image) for image in os.listdir(answer_path)]
    
    return render_template('index.html', exercise_folders=exercise_folders, selected_exercise=selected_exercise,
                           exercise_image=exercise_image, answer_images=answer_images)

@app.route('/get_images', methods=['POST'])
def get_images():
    print ("Received image")
    exercise_name = request.json['exercise_name']
    global exercis_nam
    exercis_nam = request.json['exercise_name']
    #print (exercis_nam)
    exercise_path = os.path.join(exercise_folder, exercis_nam)
    answer_path = os.path.join(answer_folder, f'answer_{exercis_nam.split("_")[1]}')
    result_path = os.path.join(result_folder, f'answer_{exercis_nam.split("_")[1]}')
    path = result_path+'/'+f'answer_{exercis_nam.split("_")[1]}'+'.csv'
    
    diff_result_path = os.path.join(difference_folder, f'answer_{exercis_nam.split("_")[1]}')
    diff_path = diff_result_path+'/'
    print (diff_path)
    if os.path.exists(path):
        print('Yes')
    else:
        print('No')
        correct_image_path = exercise_path+'/'+exercis_nam+'.png'
        print (correct_image_path)
        print (answer_path)
        check_image_size_and_similarity(correct_image_path, answer_path, path)
        highlight_image_difference(correct_image_path, answer_path, diff_path)
    # Get exercise image
    exercise_image = os.path.join(exercise_path, f'{exercis_nam}.png')

    # Get answer images
    answer_images = [os.path.join(answer_path, image) for image in os.listdir(answer_path)]
    diff_images = [os.path.join(diff_path, image) for image in os.listdir(diff_path)]
    #print (exercise_path)
    #print (answer_path)

    return jsonify({'exercise_image': exercise_image, 'answer_images': answer_images, 'diff_images': diff_images})


@app.route('/exercise_folder/<exercise_name>/<image_name>')
def get_exercise_image(exercise_name, image_name):
    return send_from_directory(os.path.join(exercise_folder, exercise_name), image_name)

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    exercise_me = request.json['exercise_me']
    result_path = os.path.join(result_folder, f'answer_{exercise_me.split("_")[1]}')
    path = result_path+'/'+f'answer_{exercise_me.split("_")[1]}'+'.csv'
    print (path)
    # Read CSV data
    csv_data = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_data.append(row)

    return jsonify({'csv_data': csv_data})

@app.route('/answer_folder/<exercise_name>/<image_name>')
def get_answer_image(exercise_name, image_name):
    answer_folder_name = f'answer_{exercise_name.split("_")[1]}'
    return send_from_directory(os.path.join(answer_folder, answer_folder_name), image_name)

@app.route('/difference_folder/<exercise_name>/<image_name>')
def get_difference_image(exercise_name, image_name):
    answer_folder_name = f'answer_{exercise_name.split("_")[1]}'
    return send_from_directory(os.path.join(difference_folder, answer_folder_name), image_name)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
