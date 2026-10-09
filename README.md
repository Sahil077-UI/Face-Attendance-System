# Face Attendance System

An automated attendance management system that uses facial recognition to identify registered students and record their attendance. The project is built with Python and DeepFace, with database storage for student records and attendance logs.

## Features

- **Face Recognition:** Identifies registered students using facial recognition.
- **Student Registration:** Stores student information and face data for identification.
- **Automated Attendance:** Records attendance when a registered student is recognized.
- **Attendance Tracking:** Maintains attendance records with timestamps.
- **Database Integration:** Uses MySQL to store student details and attendance information.
- **Real-Time Camera Recognition:** Uses a camera to capture faces and identify registered students.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| DeepFace | Facial recognition |
| RetinaFace | Face detection |
| OpenCV | Camera access and image processing |
| MySQL | Student and attendance data storage |
| NumPy | Numerical and image-related operations |

## Project Structure

```text
Face-Attendance-System/
├── app.py                  # Main application (if applicable)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── database/               # Database scripts (if applicable)
├── static/                 # Static assets (if applicable)
└── templates/              # HTML templates (if applicable)
```

*Adjust the structure above to match your actual project files.*

## Prerequisites

- Python 3.9 or a compatible version
- MySQL Server
- A webcam or compatible camera
- Git

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Face-Attendance-System.git
cd Face-Attendance-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not included, create it with the dependencies required by your project.

### 4. Configure the database

1. Start your MySQL server.
2. Create the database required by the application.
3. Configure your database connection settings.
4. Create the necessary tables for student records and attendance.

Keep database credentials in environment variables or a local `.env` file. Never commit passwords or other secrets to GitHub.

### 5. Run the application

Run the appropriate entry-point script for your project. For example:

```bash
python app.py
```

Replace `app.py` with your actual script name if it differs.

## How It Works

1. Register student information in the system.
2. Capture or register facial data for each student.
3. Start the camera-based recognition process.
4. Detect faces and compare them with registered students.
5. Record attendance for recognized students in the database.
6. Review stored attendance records.

## Database

The system uses MySQL to maintain its records. The database may include tables for:

- **Students:** Student information and identifiers.
- **Attendance:** Student attendance records and timestamps.

The exact schema depends on the implementation in this repository.

## Future Improvements

- Attendance reports with date filters.
- Export attendance records to CSV or Excel.
- Admin dashboard for managing students.
- Improved recognition performance in different lighting conditions.
- Duplicate attendance prevention and improved error handling.

## Security and Privacy

- Do not upload real student face images, biometric data, or personal records to a public repository.
- Keep database credentials and API keys out of source control.
- Obtain appropriate consent before collecting or processing facial data.

## Author

**Sahil Pratap**

GitHub: [@Sahil077-UI](https://github.com/Sahil077-UI)

---

If you find this project useful, consider giving the repository a star!
