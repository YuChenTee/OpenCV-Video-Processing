import cv2
import numpy as np

# --Variables--

# Open the video file for processing
vid = cv2.VideoCapture("alley.mp4")

# Create an output video file
out = cv2.VideoWriter('processed_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (1280, 720))

# Get the total number of frames in the input video
total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Open the overlay video file for watermarking
overlay_vid = cv2.VideoCapture("talking.mp4")

# Get the total number of frames in the overlay video
total_no_overlay_frames = int(overlay_vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Load watermark images
watermark1 = cv2.imread("watermark1.png", 1)
watermark2 = cv2.imread("watermark2.png", 1)

# Open the end screen video
end_vid = cv2.VideoCapture("endscreen.mp4")

# Get the total number of frames in the end screen video
total_no_end_frames = int(end_vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize the active watermark flag
watermark1_active = True

# Initialize an empty list to store detected faces in each frame
face_buffer = []

# Initialize an empty list to store video frames
frame_list = []

# --Functions--

# Function to determine whether it's day or night based on frame brightness
def determine_day_night():
    # Initialize a list to store mean brightness values for each frame
    mean_brightness_list = []

    for frame in frame_list:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the mean brightness of the grayscale frame
        mean_brightness = np.mean(gray_frame)

        # Append the mean brightness to the list
        mean_brightness_list.append(mean_brightness)

    # Calculate the overall mean brightness for all frames
    overall_mean_brightness = np.mean(mean_brightness_list)

    mean_threshold = 100
    print("Mean brightness = ", overall_mean_brightness)

    if overall_mean_brightness < mean_threshold:
        return "night"
    else:
        return "day"

# Function to increase brightness if it's night
def raise_brightness(frame):
    
    mask = np.zeros((frame.shape), np.uint8)
    mask +=30
    frame = cv2.add(frame, mask)
    
    return frame

# Function to check if two faces are in the same position
def is_same_position(frame, face1, face2, threshold_percent=5):
    # Calculate the center of the bounding boxes
    center1 = (face1[0] + face1[2] / 2, face1[1] + face1[3] / 2)
    center2 = (face2[0] + face2[2] / 2, face2[1] + face2[3] / 2)

    # Calculate the Euclidean distance between the centers
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    # Define a threshold for considering faces in the same position
    image_width = frame.shape[1]  # Assuming frame is the current frame
    threshold_distance = (threshold_percent / 100) * image_width

    return distance < threshold_distance

# Function to fix faces that disappear and reappear between frames
def fix_disappearing_face(frame_count):
    earlier_faces = face_buffer[0]
    previous_faces = face_buffer[1]
    current_faces = face_buffer[2]

    for face3 in current_faces:
        for face1 in earlier_faces:
            if is_same_position(frame_list[frame_count - 1], face1, face3):
                face_appeared = False
                for face2 in previous_faces:
                    if is_same_position(frame_list[frame_count - 1], face1, face2):
                        face_appeared = True
                        break
                if face_appeared == False: 
                    if len(face_buffer[1]) == 0 :# to prevent append error to empty array
                        face_buffer[1] = [face1]
                    else:
                        face_buffer[1] = np.append(face_buffer[1], [face1], axis=0)
                    x, y, w, h = face1
                    kernel_size = (int(w/2), int(w/2))
                    cv2.rectangle(frame_list[frame_count - 1], (x, y), (x+w, y+h), (255, 255, 0), 2)
                    frame_list[frame_count - 1][y:y+h, x:x+w] = cv2.blur(frame_list[frame_count - 1][y:y+h, x:x+w], kernel_size)

# Function to blur detected faces
def blur_face(frame):
    face_cascade = cv2.CascadeClassifier("face_detector.xml")
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        # kernel size depends on the width of the face
        kernel_size = (int(w/2), int(w/2))
        frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], kernel_size)

    # Append the detected faces to the buffer
    face_buffer.append(faces)
    return frame

# Function to resize the overlay video frame
def resize_overlay_video(overlay_frame):
    scale_percent = 20  # percent of original size
    width = int(overlay_frame.shape[1] * scale_percent / 100)
    height = int(overlay_frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    overlay_frame = cv2.resize(overlay_frame, dim)

    return overlay_frame

# Function to locate and add the overlay video on the frame
def locate_overlay_video(frame, overlay_frame):
    [h, w, n] = overlay_frame.shape
    x, y = 100, 100  # position to place the overlay video
    frame[y:y+h, x:x+w] = overlay_frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
    return frame

# Function to add a watermark to the frame
def add_watermark(frame, watermark):
    frame = cv2.add(frame, watermark)
    return frame

# Function to add an end screen to the output video
def add_end_screen(end_vid, n_frame, out):
    for frame_count in range(n_frame):
        success, end_frame = end_vid.read()
        out.write(end_frame)

# Function to switch between watermarks
def switch_watermark(frame_count, watermark1_active):
    # switch every 5 seconds (150/30=5)
    if frame_count % 150 == 0:
        return not watermark1_active
    else:
        return watermark1_active

# --Main Loop--

# Load all frames into the list
for frame_count in range(total_no_frames):
    print("Saving frames into list... ", round(frame_count/total_no_frames*100, 1), "%")
    success, frame = vid.read()
    if not success:
        break
    frame_list.append(frame)

# Determine whether it's day or night
day_or_night = determine_day_night()

# Process frames from the list

# If it's night, increase the brightness
if day_or_night == "night":
    for frame_count in range(len(frame_list)):
        print("Adjusting brightness... ", round(frame_count/total_no_frames*100, 1), "%")
        frame_list[frame_count] = raise_brightness(frame_list[frame_count])

for frame_count in range(len(frame_list)):
    print("Blurring faces... ", round(frame_count/total_no_frames*100, 1), "%")
    frame = frame_list[frame_count]

    # Blur detected faces
    frame = blur_face(frame)

    # If there are enough frames in the buffer, fix disappearing faces
    if frame_count > 2:
        fix_disappearing_face(frame_count)
    if len(face_buffer) > 3:
        face_buffer.pop(0)

    frame_list[frame_count] = frame

for frame_count in range(len(frame_list)):
    print("Adding watermarks and overlay video... ", round(frame_count/total_no_frames*100, 1), "%")
    frame = frame_list[frame_count]
    # Switch between watermarks
    watermark1_active = switch_watermark(frame_count, watermark1_active)
    if watermark1_active:
        frame = add_watermark(frame, watermark1)
    else:
        frame = add_watermark(frame, watermark2)

    # If there are overlay frames available, add the overlay
    if frame_count < total_no_overlay_frames:
        success, overlay_frame = overlay_vid.read()
        overlay_frame = resize_overlay_video(overlay_frame)
        frame = locate_overlay_video(frame, overlay_frame)

    frame_list[frame_count] = frame

# Write the processed frames to the output video
for frame_count in range(len(frame_list)):
    print("Writing output video... ", round(frame_count/total_no_frames*100, 1), "%")
    out.write(frame_list[frame_count])

# Add the end screen to the output video
add_end_screen(end_vid, total_no_end_frames, out)

print("Finished")