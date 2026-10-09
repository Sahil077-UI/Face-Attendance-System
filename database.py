import mysql.connector


def get_connection():

    try:

        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="mysql123",
            database="face_attendance"
        )

        return connection

    except mysql.connector.Error as error:

        print("Database connection error:", error)

        return None