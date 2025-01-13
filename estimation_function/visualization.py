import cv2
from ultralytics import YOLO
import numpy as np
import euclidean_distance as eu
import pickle

# Load the model
model = YOLO('C:\\Users\\acer\\Desktop\\Pig_Weight_Estimation_Alpha_Testing\\Trained_Model\\yolov8m-tuning-transfered-learning-model\\weights\\best.pt')

# Constants
KNOWN_DISTANCE = 89  
PIG_WIDTH = 22.5  
CONFIDENCE_THRESHOLD = 0.7
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
PINK = (255,51, 255)
FONTS = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.4

# Getting class names from classes.txt file
class_names = []
with open("C:\\Users\\acer\\Desktop\\Pig_Weight_Estimation_Alpha_Testing\\Estimation_Function\\classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

def focal_length_finder(measured_distance, real_width, width_in_rf):
    return (width_in_rf * measured_distance) / real_width

def distance_finder(focal_length, real_object_width, width_in_frame):
    return (real_object_width * focal_length) / width_in_frame

def combined_detection(image):
    results = model(image)
    distances = []
    pig_width_in_rf = 0 
    estimated_distance = 0
    back_length = 0
    focal_pig = None

    # Load regression model for predictions
    with open('C:\\Users\\acer\\Desktop\\Pig_Weight_Estimation_Alpha_Testing\\Estimation_Function\\multi_output_regressor.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            first_keypoint_set = result.keypoints.xy[0]
            for keypoint in first_keypoint_set:
                x, y = int(keypoint[0].item()), int(keypoint[1].item())
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            for i in range(len(first_keypoint_set) - 1):
                point1 = np.array([first_keypoint_set[i][0].item(), first_keypoint_set[i][1].item()])
                point2 = np.array([first_keypoint_set[i + 1][0].item(), first_keypoint_set[i + 1][1].item()])
                distance = eu.euclidean_distance(point1, point2)
                distances.append(distance)
                cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 0, 255), 2)

    results = model(image)
    for result in results:
        for box in result.boxes:
            classid = int(box.cls)
            score = float(box.conf)
            box = box.xyxy[0].cpu().numpy()

            if score > CONFIDENCE_THRESHOLD and classid == 0:
                color = COLORS[classid % len(COLORS)]
                label = f"{class_names[classid]} : {score:.2f}"
                width_in_pixels = int(box[2] - box[0])

                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(image, label, (int(box[0]), int(box[1]) - 14), FONTS, 0.5, color, 2)

                pig_width_in_rf = width_in_pixels

                if focal_pig is None:
                    focal_pig = focal_length_finder(KNOWN_DISTANCE, PIG_WIDTH, pig_width_in_rf)

                estimated_distance = distance_finder(focal_pig, PIG_WIDTH, width_in_pixels)
                x, y = int(box[0]), int(box[1])
                cv2.rectangle(image, (x, y - 3), (x + 150, y + 23), BLACK, -1)
                cv2.putText(image, f'Distance: {round(estimated_distance)} inches', (x + 5, y + 13), FONTS, 0.3, GREEN)

                # Calculate length and girth based on regression model
                new_X = [[float(estimated_distance), float(back_length), float(pig_width_in_rf)]]
                predictions = loaded_model.predict(new_X)

                length = round(predictions[0][0])
                girth = round(predictions[0][1])

                estimated_weight = ((girth * girth) * length) / 400
                # weight_in_lbs = round(estimated_weight)
                converting_to_kg = estimated_weight / 2.2046
                # weight_in_kg = round(converting_to_kg)

                # Display length, girth, and weight
                cv2.rectangle(image, (x, y + 28), (x + 150, y + 100), BLACK, -1)  # Add rectangle behind text
                cv2.putText(image, f'Length: {length} inches', (x + 5, y + 33), FONTS, FONT_SCALE, PINK)
                cv2.putText(image, f'Girth: {girth} inches', (x + 5, y + 53), FONTS, FONT_SCALE, PINK)
                cv2.putText(image, f'Weight: {converting_to_kg} KG', (x + 5, y + 73), FONTS, FONT_SCALE, PINK)


    return image

