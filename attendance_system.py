import cv2
from deepface import DeepFace
import os
import json
import numpy as np
import threading
import time
from datetime import datetime

from database import get_connection


# ==========================================
# SETTINGS
# ==========================================

SIMILARITY_THRESHOLD = 0.68
PROCESS_INTERVAL = 0.5

# Face must disappear for this many seconds
# before the same student can scan again
FACE_ABSENCE_TIME = 5


# ==========================================
# LOAD REGISTERED FACE EMBEDDINGS
# ==========================================

known_faces = []

if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

for file_name in os.listdir("embeddings"):

    if file_name.endswith(".json"):

        file_path = os.path.join(
            "embeddings",
            file_name
        )

        try:

            with open(file_path, "r") as file:

                data = json.load(file)

            known_faces.append({
                "name": data["name"],
                "student_id": data["student_id"],
                "embedding": np.array(
                    data["embedding"]
                )
            })

        except Exception as error:

            print(
                f"Error loading {file_name}:",
                error
            )


print(
    f"Loaded {len(known_faces)} registered face(s)"
)


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(
    embedding1,
    embedding2
):

    denominator = (
        np.linalg.norm(embedding1)
        *
        np.linalg.norm(embedding2)
    )

    if denominator == 0:
        return 0

    return (
        np.dot(
            embedding1,
            embedding2
        )
        /
        denominator
    )


# ==========================================
# MARK ATTENDANCE
# ==========================================

def mark_attendance(
    name,
    student_id
):

    connection = get_connection()

    if not connection:

        return "DATABASE ERROR"


    cursor = None


    try:

        cursor = connection.cursor()

        now = datetime.now()

        today = now.strftime(
            "%Y-%m-%d"
        )

        current_time = now.strftime(
            "%H:%M:%S"
        )


        # Check today's attendance

        check_query = """
        SELECT id, in_time, out_time
        FROM attendance
        WHERE student_id = %s
        AND date = %s
        """

        cursor.execute(
            check_query,
            (
                student_id,
                today
            )
        )

        record = cursor.fetchone()


        # ----------------------------------
        # FIRST SCAN = IN TIME
        # ----------------------------------

        if record is None:

            insert_query = """
            INSERT INTO attendance
            (
                student_id,
                name,
                date,
                in_time,
                status
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """

            cursor.execute(
                insert_query,
                (
                    student_id,
                    name,
                    today,
                    current_time,
                    "Present"
                )
            )

            connection.commit()

            print(
                f"\nIN TIME MARKED: "
                f"{name} "
                f"at {current_time}\n"
            )

            return "IN MARKED"


        # ----------------------------------
        # SECOND SCAN = OUT TIME
        # ----------------------------------

        elif record[2] is None:

            update_query = """
            UPDATE attendance
            SET out_time = %s
            WHERE id = %s
            """

            cursor.execute(
                update_query,
                (
                    current_time,
                    record[0]
                )
            )

            connection.commit()

            print(
                f"\nOUT TIME MARKED: "
                f"{name} "
                f"at {current_time}\n"
            )

            return "OUT MARKED"


        # ----------------------------------
        # BOTH ALREADY MARKED
        # ----------------------------------

        else:

            return "ATTENDANCE COMPLETED"


    except Exception as error:

        print(
            "Database error:",
            error
        )

        return "DATABASE ERROR"


    finally:

        if cursor:

            cursor.close()

        if connection and connection.is_connected():

            connection.close()


# ==========================================
# SHARED VARIABLES
# ==========================================

current_faces = []

processing = False

lock = threading.Lock()


# Student scan lock

scan_locked = {}

# Last time student was seen

last_seen = {}


# ==========================================
# AI FACE RECOGNITION
# ==========================================

def recognize_faces(
    frame
):

    global current_faces
    global processing


    try:

        # Resize for faster processing

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

        detected_student_ids = set()

        current_timestamp = time.time()


        for result in results:


            face_embedding = np.array(
                result["embedding"]
            )


            area = result.get(
                "facial_area",
                {}
            )


            # Scale coordinates back

            x = area.get(
                "x",
                0
            ) * 2

            y = area.get(
                "y",
                0
            ) * 2

            w = area.get(
                "w",
                0
            ) * 2

            h = area.get(
                "h",
                0
            ) * 2


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


            # ==================================
            # RECOGNIZED FACE
            # ==================================

            if (

                best_match is not None

                and

                highest_similarity
                >
                SIMILARITY_THRESHOLD

            ):


                name = best_match[
                    "name"
                ]

                student_id = best_match[
                    "student_id"
                ]


                # Remember student is visible

                detected_student_ids.add(
                    student_id
                )

                last_seen[
                    student_id
                ] = current_timestamp


                # ----------------------------------
                # CHECK SCAN LOCK
                # ----------------------------------

                if not scan_locked.get(
                    student_id,
                    False
                ):


                    attendance_status = mark_attendance(

                        name,

                        student_id

                    )


                    # Lock student after scan

                    scan_locked[
                        student_id
                    ] = True


                else:

                    attendance_status = (
                        "WAITING TO LEAVE"
                    )


                # ----------------------------------
                # DISPLAY STATUS
                # ----------------------------------

                if attendance_status == "IN MARKED":

                    label = (
                        f"{name} | IN MARKED"
                    )

                    box_color = (
                        0,
                        255,
                        0
                    )


                elif attendance_status == "OUT MARKED":

                    label = (
                        f"{name} | OUT MARKED"
                    )

                    box_color = (
                        255,
                        255,
                        0
                    )


                elif attendance_status == "WAITING TO LEAVE":

                    label = (
                        f"{name} | PLEASE LEAVE CAMERA"
                    )

                    box_color = (
                        0,
                        255,
                        255
                    )


                elif attendance_status == "ATTENDANCE COMPLETED":

                    label = (
                        f"{name} | COMPLETED"
                    )

                    box_color = (
                        255,
                        0,
                        255
                    )


                else:

                    label = (
                        f"{name} | ERROR"
                    )

                    box_color = (
                        0,
                        0,
                        255
                    )


            # ==================================
            # UNKNOWN FACE
            # ==================================

            else:


                label = "UNKNOWN"

                box_color = (
                    0,
                    0,
                    255
                )


            # Save detection

            detected_faces.append({

                "x": x,

                "y": y,

                "w": w,

                "h": h,

                "label": label,

                "similarity": highest_similarity,

                "color": box_color

            })


        # ==================================
        # CHECK WHO LEFT THE CAMERA
        # ==================================

        for student_id in list(
            scan_locked.keys()
        ):


            if student_id not in detected_student_ids:


                last_time = last_seen.get(

                    student_id,

                    current_timestamp

                )


                # If absent for 5 seconds

                if (

                    current_timestamp
                    -
                    last_time
                    >=
                    FACE_ABSENCE_TIME

                ):


                    scan_locked[
                        student_id
                    ] = False


                    print(

                        f"{student_id} can scan again"

                    )


        # Update display

        with lock:

            current_faces = (
                detected_faces
            )


    except Exception as error:

        print(
            "Recognition error:",
            error
        )


    finally:

        processing = False


# ==========================================
# START CAMERA
# ==========================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "Could not open webcam"
    )

    exit()


print("\n================================")
print("AI FACE ATTENDANCE SYSTEM")
print("================================")

print(
    f"Registered faces: "
    f"{len(known_faces)}"
)

print(
    f"Face must leave for "
    f"{FACE_ABSENCE_TIME} seconds "
    f"before scanning again."
)

print(
    "Press Q to quit\n"
)


last_process_time = 0


# ==========================================
# LIVE CAMERA LOOP
# ==========================================

while True:


    ret, frame = cap.read()


    if not ret:

        break


    display_frame = frame.copy()


    current_time = time.time()


    # Start background recognition

    if (

        not processing

        and

        current_time
        -
        last_process_time
        >
        PROCESS_INTERVAL

    ):


        processing = True

        last_process_time = current_time


        frame_copy = frame.copy()


        thread = threading.Thread(

            target=recognize_faces,

            args=(frame_copy,),

            daemon=True

        )


        thread.start()


    # Draw faces

    with lock:

        faces_to_draw = (
            current_faces.copy()
        )


    for face in faces_to_draw:


        x = face["x"]

        y = face["y"]

        w = face["w"]

        h = face["h"]

        label = face["label"]

        box_color = face["color"]


        cv2.rectangle(

            display_frame,

            (x, y),

            (x + w, y + h),

            box_color,

            2

        )


        text_y = y - 10


        if text_y < 20:

            text_y = (
                y
                +
                h
                +
                25
            )


        cv2.putText(

            display_frame,

            label,

            (x, text_y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            box_color,

            2

        )


    # Show camera

    cv2.imshow(

        "AI Face Attendance System - Press Q to Quit",

        display_frame

    )


    # Quit

    if (

        cv2.waitKey(1)
        &
        0xFF
        ==
        ord("q")

    ):

        break


cap.release()

cv2.destroyAllWindows()

print(
    "Attendance system stopped."
)