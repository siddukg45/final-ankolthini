import os
import cv2
import time
import pickle
import mysql.connector
import numpy as np
import datetime
from threading import Thread
from queue import Queue
from zoneinfo import ZoneInfo
from sklearn.metrics.pairwise import cosine_similarity

from keras_facenet import FaceNet
from twilio.rest import Client

# ---------------- CONFIG ----------------
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "siddu@8276"
MYSQL_DB = "CSE"

EMBED_THRESHOLD = 0.60
COOLDOWN_SEC = 30
IMG_SIZE = (160, 160)

WORK_START = datetime.time(8, 0, 0)
WORK_END = datetime.time(21, 30, 59)
LOCAL_TZ = ZoneInfo("Asia/Kolkata")

TWILIO_SID = "AC87623090c881a78388110fc072677480"
TWILIO_TOKEN = "84068acc753b828ca6ac4632623113e2"
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

# FaceNet
embedder = FaceNet()

# Haarcascade
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Queues (ONLY latest frame kept)
detect_queue = Queue(maxsize=1)
recognition_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)


# ---------------- DB CONNECT ----------------
def get_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )


# ---------------- SEND WHATSAPP ----------------
def send_whatsapp_message(phone, text):
    if not phone.startswith("whatsapp:"):
        phone = "whatsapp:" + phone
    try:
        twilio_client.messages.create(
            body=text,
            from_=TWILIO_WHATSAPP_FROM,
            to=phone
        )
        return True
    except Exception as e:
        print("[ERROR] Twilio:", e)
        return False


def normalize_phone(num):
    s = str(num).strip()
    if len(s) == 10:
        return "+91" + s
    return s


# ---------------- LOAD EMBEDDINGS ----------------
def load_known_faces():
    known = []
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, embedding, contact_number FROM employees")
        rows = cur.fetchall()

        for emp_id, name, emb_blob, phone in rows:
            if emb_blob is None:
                continue
            emb = pickle.loads(emb_blob)
            known.append({
                "id": emp_id,
                "name": name,
                "embedding": np.asarray(emb, dtype=np.float32),
                "contact": phone
            })

        cur.close()
        conn.close()
    except Exception as e:
        print("[ERROR] load_known_faces:", e)

    return known


# ---------------- EMBEDDING ----------------
def compute_embedding_from_crop(face_crop_bgr):
    try:
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, IMG_SIZE)
        emb = embedder.embeddings([face_resized])[0]
        return np.asarray(emb, dtype=np.float32)
    except:
        return None


# ---------------- MATCHING ----------------
def find_best_match(embed, known_list):
    best = None
    best_score = -1
    for rec in known_list:
        score = float(cosine_similarity([embed], [rec["embedding"]])[0][0])
        if score > best_score:
            best_score = score
            best = rec
    return best, best_score


# ---------------- ATTENDANCE ----------------
def ensure_attendance_row(emp_id, date):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id,in1,out1,in2,out2 FROM attendance WHERE emp_id=%s AND date=%s",
                (emp_id, date))
    row = cur.fetchone()
    if row is None:
        cur.execute("INSERT INTO attendance (emp_id,date) VALUES (%s,%s)", (emp_id, date))
        conn.commit()
        cur.execute("SELECT id,in1,out1,in2,out2 FROM attendance WHERE emp_id=%s AND date=%s",
                    (emp_id, date))
        row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def update_field(att_id, field, time_str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"UPDATE attendance SET {field}=%s WHERE id=%s", (time_str, att_id))
    conn.commit()
    cur.close()
    conn.close()


def now_local():
    return datetime.datetime.now(LOCAL_TZ)


def is_within_attendance_window(dt):
    t = dt.time()
    return WORK_START <= t <= WORK_END


def mask_phone(phone):
    p = phone.replace("whatsapp:", "")
    length = len(p)
    if length <= 5:
        return p
    first = p[:2]
    last = p[-3:]
    middle = "X" * (length - 5)
    return first + middle + last

def notify_student(emp_id, event, time_str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name,contact_number FROM employees WHERE id=%s", (emp_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return

    name, phone = row
    phone = normalize_phone(phone)

    # Get today's date in local timezone
    today_date = now_local().strftime("%Y-%m-%d")

    # Session label
    session = "morning" if event in ("IN1", "OUT1") else "afternoon"
    action = "checked in" if event.startswith("IN") else "checked out"

    # Message with DATE added
    msg = (
        f"Dear {name},\n"
        f"You have {action} for the {session} session at {time_str} on {today_date}."
    )

    status = send_whatsapp_message(phone, msg)
    masked = mask_phone(phone)

    print(f"[WHATSAPP] {emp_id} → {masked} → sent={status}")

def mark_attendance(emp_id):
    now = now_local()
    today = now.date()
    ts = now.strftime("%H:%M:%S")
    t = now.time()

    att_id, in1, out1, in2, out2 = ensure_attendance_row(emp_id, today)

    LUNCH_OUT = datetime.time(13, 45)
    LUNCH_IN  = datetime.time(14, 30)

    # --- CASE 1: MORNING SESSION ---
    if t < LUNCH_OUT:
        if in1 is None:
            update_field(att_id, "in1", ts)
            notify_student(emp_id, "IN1", ts)
            print(f"[IN1]  {emp_id} logged in at {ts}")

        else:
            update_field(att_id, "out1", ts)
            notify_student(emp_id, "OUT1", ts)
            print(f"[OUT1] {emp_id} logged out at {ts}")

        return

    # --- CASE 2: AFTERNOON SESSION (t >= 13:45) ---
    # SPECIAL CASE: Person never logged OUT1 & never logged IN2
    if out1 is None and in1 is not None and in2 is None:
        # Auto-fill the missing values
        update_field(att_id, "out1", LUNCH_OUT.strftime("%H:%M:%S"))
        update_field(att_id, "in2",  LUNCH_IN.strftime("%H:%M:%S"))

        print(f"[AUTO] Filled OUT1={LUNCH_OUT}, IN2={LUNCH_IN} for {emp_id}")

        # Now continue to treat this as a normal OUT2 event
        update_field(att_id, "out2", ts)
        notify_student(emp_id, "OUT2", ts)
        print(f"[OUT2] {emp_id} logged out at {ts}")
        return

    # --- NORMAL AFTERNOON BEHAVIOR ---
    if in2 is None:
        update_field(att_id, "in2", ts)
        notify_student(emp_id, "IN2", ts)
        print(f"[IN2]  {emp_id} logged in at {ts}")

    else:
        update_field(att_id, "out2", ts)
        notify_student(emp_id, "OUT2", ts)
        print(f"[OUT2] {emp_id} logged out at {ts}")


# def mark_attendance(emp_id):
#     today = now_local().date()
#     ts = now_local().strftime("%H:%M:%S")
#     t = now_local().time()

#     att_id, in1, out1, in2, out2 = ensure_attendance_row(emp_id, today)

#     if t < datetime.time(13, 45):

#         if in1 is None:
#             update_field(att_id, "in1", ts)
#             notify_student(emp_id, "IN1", ts)
#             print(f"[IN1]  {emp_id} logged in at {ts}")

#         else:
#             update_field(att_id, "out1", ts)
#             notify_student(emp_id, "OUT1", ts)
#             print(f"[OUT1] {emp_id} logged out at {ts}")

#     else:

#         if in2 is None:
#             update_field(att_id, "in2", ts)
#             notify_student(emp_id, "IN2", ts)
#             print(f"[IN2]  {emp_id} logged in at {ts}")

#         else:
#             update_field(att_id, "out2", ts)
#             notify_student(emp_id, "OUT2", ts)
#             print(f"[OUT2] {emp_id} logged out at {ts}")


# ==============================================================  
#               REAL-TIME PIPELINE (NO FRAME SKIP)
# ==============================================================

known_faces = []
last_seen = {}

# -------- CAMERA THREAD --------
def camera_thread():
    cap = cv2.VideoCapture("rtsp://10.154.79.119:1945/")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    while True:
        cap.grab()         # flush buffer
        ret, frame = cap.read()
        if not ret:
            continue

        if detect_queue.full():
            detect_queue.get()

        detect_queue.put(frame)


# -------- DETECTION THREAD --------
def detection_thread():
    while True:
        frame = detect_queue.get()

        # Resized detection frame (FAST)
        det_frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 1.1, 5)

        detections = []
        for (x, y, w, h) in faces:
            # Adjust for resized frame
            sx = int(x * (frame.shape[1] / 640))
            sy = int(y * (frame.shape[0] / 480))
            sw = int(w * (frame.shape[1] / 640))
            sh = int(h * (frame.shape[0] / 480))

            crop = frame[sy:sy+sh, sx:sx+sw]
            detections.append((sx, sy, sw, sh, crop))

        if recognition_queue.full():
            recognition_queue.get()

        recognition_queue.put((frame, detections))


# -------- ASYNC ATTENDANCE THREAD --------
def async_mark(emp_id):
    Thread(target=mark_attendance, args=(emp_id,), daemon=True).start()


# -------- RECOGNITION THREAD --------
def recognition_thread():
    global last_seen

    while True:
        frame, detections = recognition_queue.get()
        results = []

        for (x, y, w, h, crop) in detections:

            emb = compute_embedding_from_crop(crop)
            if emb is None:
                results.append((x, y, w, h, None, None))
                continue

            rec, score = find_best_match(emb, known_faces)

            if rec and score >= EMBED_THRESHOLD:

                emp_id = rec["id"]
                name = rec["name"]
                now_ts = time.time()

                if now_ts - last_seen.get(emp_id, 0) > COOLDOWN_SEC:
                    if is_within_attendance_window(now_local()):
                        async_mark(emp_id)
                    last_seen[emp_id] = now_ts

                results.append((x, y, w, h, name, score))
            else:
                results.append((x, y, w, h, None, None))

        if result_queue.full():
            result_queue.get()

        result_queue.put((frame, results))


# -------- DRAW RESULTS --------
def draw_results(frame, results):
    for (x, y, w, h, name, score) in results:
        if name:
            color = (0, 255, 0)
            label = f"{name}"
        else:
            color = (0, 0, 255)
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# -------- UI LOOP --------
def ui_loop():
    while True:
        if not result_queue.empty():
            frame, results = result_queue.get()
            draw_results(frame, results)
            cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# -------- START SYSTEM --------
def start_system():
    global known_faces
    known_faces = load_known_faces()
    print("[INFO] Loaded", len(known_faces), "faces")

    Thread(target=camera_thread, daemon=True).start()
    Thread(target=detection_thread, daemon=True).start()
    Thread(target=recognition_thread, daemon=True).start()

    ui_loop()


if __name__ == "__main__":
    start_system()
