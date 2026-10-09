import cv2
from deepface import DeepFace
import os
import json
import numpy as np


# -------------------------------
# Load registered face embeddings
# -------------------------------

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


# -------------------------------
# Calculate cosine similarity
# -------------------------------

def cosine_similarity(embedding1, embedding2):

    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) *
        np.linalg.norm(embedding2)
    )


# -------------------------------
# Open webcam
# -------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()


print("Face recognition started")
print("Press Q to quit")


# -------------------------------
# Real-time recognition
# -------------------------------

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # Show webcam
    display_frame = frame.copy()


    # Process every 15th frame
    frame_count += 1

    if frame_count % 15 == 0:

        try:

            # Save current frame temporarily
            temp_image = "temp_recognition.jpg"

            cv2.imwrite(temp_image, frame)


            # Generate embedding
            results = DeepFace.represent(
                img_path=temp_image,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )


            for result in results:

                face_embedding = np.array(result["embedding"])


                best_match = None
                highest_similarity = -1


                # Compare with registered faces
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

                    text = (
                        f"{best_match['name']} "
                        f"({highest_similarity:.2f})"
                    )

                    color = (0, 255, 0)

                else:

                    text = (
                        f"UNKNOWN "
                        f"({highest_similarity:.2f})"
                    )

                    color = (0, 0, 255)


                print(text)


                # Display result
                cv2.putText(
                    display_frame,
                    text,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )


        except Exception as e:

            print("Error:", e)


    cv2.imshow("AI Face Recognition - Press Q to Quit", display_frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


# Remove temporary image
if os.path.exists("temp_recognition.jpg"):
    os.remove("temp_recognition.jpg")