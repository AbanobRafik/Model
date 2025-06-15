import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ========== SETUP GOOGLE SHEETS ==========
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("faceattendance-462916-b4d11df32761.json", scope)
client = gspread.authorize(creds)
sheet = client.open("Face Attendance").sheet1

# Optional: Write header if empty
if sheet.row_count == 0 or sheet.cell(1, 1).value is None:
    sheet.append_row(["Name", "Time"])

# ========== LOAD AND ENCODE KNOWN FACES ==========
path = 'persons'
images = []
classNames = []

# Read images from 'persons' folder
personsList = os.listdir(path)
for cl in personsList:
    curImage = cv2.imread(f'{path}/{cl}')
    if curImage is not None:
        images.append(curImage)
        classNames.append(os.path.splitext(cl)[0])
print("Loaded classes:", classNames)

# Encode faces once
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete.')

# ========== FUNCTION TO RECOGNIZE AND LOG ==========
def recognize_and_log(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "No image found"

    img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(img_small)
    encodings = face_recognition.face_encodings(img_small, face_locations)

    for encodeFace in encodings:
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sheet.append_row([name, now])
            return name

    return "Unknown"
