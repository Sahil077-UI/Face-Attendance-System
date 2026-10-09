import cv2
from deepface import DeepFace
import os
import json
import numpy as np
import threading
import time

# --------------------------------
# Load registered face embeddings
# --------------------------------

known_faces = []

for file_name in os.listdir("embeddings"):
    if file_name.endswith(".json"):

        file_path = os.path.join("embeddings", file_name)

        with open(file_path, "r") as file:
            data = json.load(file)

        known_faces.append({
            "name": data["name"],
            "student_id": data["student_id"],
            "embedding": np.array(data["embedding"])
        })

print(f"Loaded {len(known_faces)} registered face(s)")


# --------------------------------
# Cosine similarity
# --------------------------------

def cosine_similarity(embedding1, embedding2):

    denominator = (
        np.linalg.norm(embedding1) *
        np.linalg.norm(embedding2)
    )

    if denominator == 0:
        return 0

    return np.dot(embedding1, embedding2) / denominator


# --------------------------------
# Shared recognition results
# --------------------------------

current_faces = []
processing = False
lock = threading.Lock()


# --------------------------------
# AI Recognition function
# --------------------------------

def recognize_faces(frame):

    global current_faces
    global processing

    try:

        # Resize frame for faster AI processing
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.5,
            fy=0.5
        )

        results = DeepFace.represent(
            img_path=small_frame,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False
        )

        detected_faces = []

        for result in results:

            face_embedding = np.array(result["embedding"])

            area = result.get("facial_area", {})

            # Scale coordinates back to original frame
            x = area.get("x", 0) * 2
            y = area.get("y", 0) * 2
            w = area.get("w", 0) * 2
            h = area.get("h", 0) * 2

            best_match = None
            highest_similarity = -1

            # Compare with known faces
            for known_face in known_faces:

                similarity = cosine_similarity(
                    face_embedding,
                    known_face["embedding"]
                )

                if similarity > highest_similarity:

                    highest_similarity = similarity
                    best_match = known_face


            # Recognition threshold
            if highest_similarity > 0.68:

                label = (
                    f"{best_match['name']} "
                    f"| ID: {best_match['student_id']}"
                )

                box_color = (0, 255, 0)

            else:

                label = "UNKNOWN"

                box_color = (0, 0, 255)


            detected_faces.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": label,
                "color": box_color
            })


        # Safely update recognition results
        with lock:

            current_faces = detected_faces


    except Exception as e:

        print("Recognition error:", e)


    finally:

        processing = False


# --------------------------------
# Open webcam
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Could not open webcam")
    exit()


print("Live face recognition started")
print("Press Q to quit")


last_process_time = 0
PROCESS_INTERVAL = 0.5


# --------------------------------
# Main live camera loop
# --------------------------------

while True:

    ret, frame = cap.read()

    if not ret:

        break


    display_frame = frame.copy()


    # Start AI processing in background
    current_time = time.time()

    if (
        not processing
        and current_time - last_process_time > PROCESS_INTERVAL
    ):

        processing = True
        last_process_time = current_time

        # Send frame to background thread
        frame_copy = frame.copy()

        thread = threading.Thread(
            target=recognize_faces,
            args=(frame_copy,),
            daemon=True
        )

        thread.start()


    # --------------------------------
    # Draw latest recognition results
    # --------------------------------

    with lock:
        faces_to_draw = current_faces.copy()


    for face in faces_to_draw:

        x = face["x"]
        y = face["y"]
        w = face["w"]
        h = face["h"]

        label = face["label"]
        box_color = face["color"]


        # Draw bounding box
        cv2.rectangle(
            display_frame,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )


        # Draw label
        text_y = y - 10

        if text_y < 20:
            text_y = y + h + 25


        cv2.putText(
            display_frame,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )


    # Show live camera continuously
    cv2.imshow(
        "Live AI Face Recognition - Press Q to Quit",
        display_frame
    )


    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()