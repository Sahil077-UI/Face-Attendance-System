import cv2
from deepface import DeepFace
import os
import json
import numpy as np
from database import get_connection

# Create embeddings folder
os.makedirs("embeddings", exist_ok=True)

name = input("Enter your name: ")
student_id = input("Enter your student ID: ")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("\nMultiple Face Registration Started")
print("Press SPACE to capture a sample")
print("Capture faces from slightly different angles")
print("Press Q to quit")

embeddings = []
sample_number = 0
TOTAL_SAMPLES = 5

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam")
        break

    display_frame = frame.copy()

    # Display sample counter
    cv2.putText(
        display_frame,
        f"Samples: {sample_number}/{TOTAL_SAMPLES}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        "Press SPACE to capture",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Face Registration", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture face
    if key == 32:

        print(f"\nProcessing sample {sample_number + 1}...")

        try:
            results = DeepFace.represent(
                img_path=frame,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=True
            )

            # Make sure exactly one face is captured
            if len(results) != 1:
                print("Please make sure only one face is visible.")
                continue

            embedding = np.array(results[0]["embedding"])

            embeddings.append(embedding)

            sample_number += 1

            print(f"Sample {sample_number}/{TOTAL_SAMPLES} captured successfully!")

            # Stop after enough samples
            if sample_number >= TOTAL_SAMPLES:
                break

        except Exception as e:
            print("Face detection failed:", e)

    elif key == ord("q"):
        print("Registration cancelled")
        break


cap.release()
cv2.destroyAllWindows()


# --------------------------------
# Create average face embedding
# --------------------------------

if len(embeddings) > 0:

    # Calculate average embedding
    average_embedding = np.mean(
        embeddings,
        axis=0
    )

    # Normalize the embedding
    average_embedding = (
        average_embedding /
        np.linalg.norm(average_embedding)
    )

    # Save person information
    data = {
        "name": name,
        "student_id": student_id,
        "samples": len(embeddings),
        "embedding": average_embedding.tolist()
    }

    file_path = f"embeddings/{student_id}.json"

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    print("\nRegistration completed successfully!")
    print("Name:", name)
    print("Student ID:", student_id)
    print("Samples captured:", len(embeddings))
    print("Embedding length:", len(average_embedding))

else:
    print("No face samples were captured.")


# --------------------------------
# SAVE STUDENT TO MYSQL
# --------------------------------

connection = get_connection()

if connection and connection.is_connected():

    try:

        cursor = connection.cursor()

        query = """
        INSERT INTO students (student_id, name)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
        name = VALUES(name)
        """

        cursor.execute(
            query,
            (
                student_id,
                name
            )
        )

        connection.commit()

        print("Student saved to MySQL successfully!")

    except Exception as error:

        print("MySQL error:", error)

    finally:

        if connection.is_connected():

            cursor.close()

            connection.close()