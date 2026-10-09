import cv2
import os
import json
import time
import threading
import numpy as np
import base64

from flask import (
    Flask,
    render_template,
    Response,
    redirect,
    url_for,
    request,
    jsonify
)

from deepface import DeepFace
from database import get_connection


app = Flask(__name__)


# ==========================================
# SETTINGS
# ==========================================

SIMILARITY_THRESHOLD = 0.68
PROCESS_INTERVAL = 0.5
FACE_ABSENCE_TIME = 5


# ==========================================
# LOAD REGISTERED FACES
# ==========================================

known_faces = []

if not os.path.exists("embeddings"):
    os.makedirs("embeddings")


for file_name in os.listdir("embeddings"):

    if not file_name.endswith(".json"):
        continue

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
# CAMERA VARIABLES
# ==========================================

camera = None

camera_lock = threading.Lock()

camera_running = False


# ==========================================
# RECOGNITION VARIABLES
# ==========================================

current_faces = []

processing = False

last_process_time = 0

lock = threading.Lock()


scan_locked = {}

last_seen = {}


# ==========================================
# START CAMERA
# ==========================================

def start_camera():

    global camera
    global camera_running

    with camera_lock:

        if camera_running:

            return


        print("Starting camera...")

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():

            print("ERROR: Could not open camera")

            camera.release()

            camera = None

            return


        camera_running = True

        print("Camera started")


# ==========================================
# STOP CAMERA
# ==========================================

def stop_camera():

    global camera
    global camera_running
    global current_faces
    global processing

    with camera_lock:

        if camera is not None:

            print("Stopping camera...")

            camera.release()

            camera = None


        camera_running = False

        current_faces = []

        processing = False

        print("Camera stopped")


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

        from datetime import datetime

        now = datetime.now()

        today = now.strftime(
            "%Y-%m-%d"
        )

        current_time = now.strftime(
            "%H:%M:%S"
        )


        # Check today's attendance

        query = """
        SELECT id, in_time, out_time
        FROM attendance
        WHERE student_id = %s
        AND date = %s
        """

        cursor.execute(
            query,
            (
                student_id,
                today
            )
        )

        record = cursor.fetchone()


        # ==================================
        # FIRST SCAN = IN
        # ==================================

        if record is None:

            query = """
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
                query,
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
                f"IN marked: {name} "
                f"at {current_time}"
            )

            return "IN MARKED"


        # ==================================
        # SECOND SCAN = OUT
        # ==================================

        elif record[2] is None:

            query = """
            UPDATE attendance
            SET out_time = %s
            WHERE id = %s
            """

            cursor.execute(
                query,
                (
                    current_time,
                    record[0]
                )
            )

            connection.commit()

            print(
                f"OUT marked: {name} "
                f"at {current_time}"
            )

            return "OUT MARKED"


        # ==================================
        # COMPLETED
        # ==================================

        else:

            return "COMPLETED"


    except Exception as error:

        print(
            "Database error:",
            error
        )

        return "DATABASE ERROR"


    finally:

        if cursor:

            cursor.close()

        if connection:

            connection.close()


# ==========================================
# FACE RECOGNITION
# ==========================================

def recognize_faces(frame):

    global current_faces
    global processing

    try:

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

        detected_ids = set()

        current_timestamp = time.time()


        for result in results:

            embedding = np.array(
                result["embedding"]
            )


            area = result.get(
                "facial_area",
                {}
            )


            x = int(
                area.get("x", 0) * 2
            )

            y = int(
                area.get("y", 0) * 2
            )

            w = int(
                area.get("w", 0) * 2
            )

            h = int(
                area.get("h", 0) * 2
            )


            best_match = None

            best_similarity = -1


            # Compare with registered faces

            for known_face in known_faces:

                similarity = cosine_similarity(

                    embedding,

                    known_face["embedding"]

                )


                if similarity > best_similarity:

                    best_similarity = similarity

                    best_match = known_face


            # ==================================
            # RECOGNIZED FACE
            # ==================================

            if (

                best_match is not None

                and

                best_similarity
                >=
                SIMILARITY_THRESHOLD

            ):

                name = best_match["name"]

                student_id = best_match[
                    "student_id"
                ]


                detected_ids.add(
                    student_id
                )


                last_seen[
                    student_id
                ] = current_timestamp


                # ==================================
                # ATTENDANCE
                # ==================================

                if not scan_locked.get(
                    student_id,
                    False
                ):

                    status = mark_attendance(

                        name,

                        student_id

                    )

                    scan_locked[
                        student_id
                    ] = True

                else:

                    status = "WAITING"


                # ==================================
                # LABEL
                # ==================================

                if status == "IN MARKED":

                    label = (
                        f"{name} | IN"
                    )

                    color = (
                        0,
                        255,
                        0
                    )


                elif status == "OUT MARKED":

                    label = (
                        f"{name} | OUT"
                    )

                    color = (
                        255,
                        255,
                        0
                    )


                elif status == "COMPLETED":

                    label = (
                        f"{name} | COMPLETED"
                    )

                    color = (
                        255,
                        0,
                        255
                    )


                elif status == "WAITING":

                    label = (
                        f"{name} | SCANNED"
                    )

                    color = (
                        0,
                        255,
                        255
                    )


                else:

                    label = (
                        f"{name} | ERROR"
                    )

                    color = (
                        0,
                        0,
                        255
                    )


            # ==================================
            # UNKNOWN
            # ==================================

            else:

                label = "UNKNOWN"

                color = (
                    0,
                    0,
                    255
                )


            detected_faces.append({

                "x": x,

                "y": y,

                "w": w,

                "h": h,

                "label": label,

                "color": color

            })


        # ==================================
        # UNLOCK AFTER LEAVING
        # ==================================

        for student_id in list(
            scan_locked.keys()
        ):

            if student_id not in detected_ids:

                last_time = last_seen.get(
                    student_id,
                    current_timestamp
                )


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
                        f"{student_id} "
                        "ready for next scan"
                    )


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
# VIDEO GENERATOR
# ==========================================

def generate_frames():

    global processing
    global last_process_time


    while camera_running:

        with camera_lock:

            if camera is None:

                break

            success, frame = camera.read()


        if not success:

            break


        current_time = time.time()


        # Run AI in background

        if (

            not processing

            and

            current_time
            -
            last_process_time
            >=
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


        # Draw recognition

        with lock:

            faces = current_faces.copy()


        for face in faces:

            x = face["x"]

            y = face["y"]

            w = face["w"]

            h = face["h"]

            label = face["label"]

            color = face["color"]


            cv2.rectangle(

                frame,

                (x, y),

                (x + w, y + h),

                color,

                2

            )


            cv2.putText(

                frame,

                label,

                (
                    x,
                    max(y - 10, 20)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                color,

                2

            )


        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )


        if not ret:

            continue


        frame_bytes = buffer.tobytes()


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            +
            frame_bytes
            +
            b"\r\n"

        )


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/")
def dashboard():

    connection = get_connection()

    total_students = 0

    present_today = 0

    attendance_today = []


    if connection:

        cursor = connection.cursor(
            dictionary=True
        )


        cursor.execute(
            "SELECT COUNT(*) AS total FROM students"
        )

        total_students = (
            cursor.fetchone()["total"]
        )


        cursor.execute(
            """
            SELECT
                student_id,
                name,
                date,
                in_time,
                out_time,
                status
            FROM attendance
            WHERE date = CURDATE()
            ORDER BY in_time DESC
            """
        )


        attendance_today = (
            cursor.fetchall()
        )


        present_today = len(
            attendance_today
        )


        cursor.close()

        connection.close()


    return render_template(

        "index.html",

        total_students=total_students,

        present_today=present_today,

        attendance_today=attendance_today

    )


# ==========================================
# START ATTENDANCE PAGE
# ==========================================

@app.route("/attendance")
def attendance():

    start_camera()

    return render_template(
        "attendance.html"
    )


# ==========================================
# STOP CAMERA
# ==========================================

@app.route("/stop_camera")
def stop_camera_route():

    stop_camera()

    return redirect(url_for("dashboard"))


# ==========================================
# VIDEO FEED
# ==========================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )

    )

@app.route("/students")
def students():

    connection = get_connection()

    student_list = []

    if connection:

        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT
                student_id,
                name
            FROM students
            ORDER BY student_id
            """
        )

        student_list = cursor.fetchall()

        cursor.close()
        connection.close()

    return render_template(
        "students.html",
        students=student_list
    )

@app.route("/register")
def register():

    return render_template(
        "register.html"
    )


@app.route("/register_student", methods=["POST"])
def register_student():

    try:

        data = request.get_json()

        name = data.get("name")
        student_id = data.get("student_id")
        samples = data.get("samples", [])


        # ==================================
        # VALIDATION
        # ==================================

        if not name or not student_id:

            return jsonify({
                "success": False,
                "message": "Name and Student ID are required."
            })


        if len(samples) != 5:

            return jsonify({
                "success": False,
                "message": "Exactly 5 face samples are required."
            })


        # ==================================
        # CHECK DUPLICATE STUDENT
        # ==================================

        connection = get_connection()

        if not connection:

            return jsonify({
                "success": False,
                "message": "Database connection failed."
            })


        cursor = connection.cursor()


        cursor.execute(
            """
            SELECT student_id
            FROM students
            WHERE student_id = %s
            """,
            (student_id,)
        )


        existing_student = cursor.fetchone()


        if existing_student:

            cursor.close()
            connection.close()

            return jsonify({
                "success": False,
                "message": "Student ID already exists."
            })


        # ==================================
        # GENERATE EMBEDDINGS
        # ==================================

        embeddings = []


        for sample in samples:

            # Remove data URL prefix

            image_data = sample.split(
                ",",
                1
            )[1]


            image_bytes = base64.b64decode(
                image_data
            )


            # Convert to numpy image

            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )


            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )


            if image is None:

                cursor.close()
                connection.close()

                return jsonify({
                    "success": False,
                    "message": "Invalid face image."
                })


            # ==================================
            # ARC FACE
            # ==================================

            result = DeepFace.represent(

                img_path=image,

                model_name="ArcFace",

                detector_backend="retinaface",

                enforce_detection=True

            )


            if not result:

                continue


            embeddings.append(
                result[0]["embedding"]
            )


        # ==================================
        # CHECK EMBEDDINGS
        # ==================================

        if len(embeddings) == 0:

            cursor.close()
            connection.close()

            return jsonify({
                "success": False,
                "message": "No face detected in the samples."
            })


        # ==================================
        # AVERAGE EMBEDDING
        # ==================================

        final_embedding = np.mean(
            np.array(embeddings),
            axis=0
        )


        # Normalize

        final_embedding = (
            final_embedding
            /
            np.linalg.norm(
                final_embedding
            )
        )


        # ==================================
        # SAVE STUDENT TO MYSQL
        # ==================================

        cursor.execute(

            """
            INSERT INTO students
            (
                student_id,
                name
            )
            VALUES
            (
                %s,
                %s
            )
            """,

            (
                student_id,
                name
            )

        )


        connection.commit()


        cursor.close()

        connection.close()


        # ==================================
        # SAVE EMBEDDING
        # ==================================

        embedding_file = os.path.join(

            "embeddings",

            f"{student_id}.json"

        )


        with open(
            embedding_file,
            "w"
        ) as file:

            json.dump(

                {

                    "name": name,

                    "student_id": student_id,

                    "embedding":
                        final_embedding.tolist()

                },

                file

            )


        # Reload known faces

        known_faces.append({

            "name": name,

            "student_id": student_id,

            "embedding":
                final_embedding

        })


        print(
            f"Registered student: "
            f"{name} ({student_id})"
        )


        return jsonify({

            "success": True,

            "message":
                "Student registered successfully."

        })


    except Exception as error:

        print(
            "Registration error:",
            error
        )


        return jsonify({

            "success": False,

            "message":
                f"Registration failed: {error}"

        })
    
# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    print(
        "Starting AI Face Attendance System..."
    )

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True

    )