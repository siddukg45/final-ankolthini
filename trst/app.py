# --------------------- PART 1 ---------------------
# app.py (Final Merged Version)

import os
import pickle
import random
import numpy as np
import mysql.connector
from flask import Flask, render_template, request, jsonify, url_for, session, redirect, send_file, abort
from datetime import datetime, date, timedelta, time as dtime
import io
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# --- For face capture / embeddings ---
import cv2
import cv2.data
from keras_facenet import FaceNet

# ---------------- CONFIG ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_strong_secret")

# DB_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "siddu@8276",
#     "database": "CSE"
# }

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Tejas@sql1",
    "database": "face_db"
}

def get_db_conn():
    return mysql.connector.connect(**DB_CONFIG)

# ---------------- ensure tables exist ----------------
def ensure_tables():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(100),
        embedding LONGBLOB,
        password_hash VARCHAR(255),
        contact_number VARCHAR(20)
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INT AUTO_INCREMENT PRIMARY KEY,
        emp_id VARCHAR(50),
        date DATE,
        in1 TIME,
        out1 TIME,
        in2 TIME,
        out2 TIME,
        FOREIGN KEY (emp_id) REFERENCES employees(id)
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

    # ---------------- Admins table ----------------
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        admin_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(100),
        password_hash VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()

    # Auto-create default admin only if missing
    cur.execute("SELECT admin_id FROM admins WHERE admin_id='admin'")
    if not cur.fetchone():
        pw_hash = generate_password_hash("admin123")
        cur.execute("""
            INSERT INTO admins (admin_id, name, password_hash)
            VALUES ('admin', 'Super Admin', %s)
        """, (pw_hash,))
        conn.commit()

    cur.close()
    conn.close()

ensure_tables()

# ---------------- utilities ----------------
import datetime as _dt

def time_to_str_safe(t):
    if t is None:
        return None
    if isinstance(t, str):
        return t
    if isinstance(t, _dt.time):
        try:
            return t.strftime("%H:%M:%S")
        except:
            return str(t)
    if isinstance(t, _dt.timedelta):
        total_seconds = int(t.total_seconds())
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return str(t)

def parse_time_str(t):
    if t is None:
        return None
    if isinstance(t, _dt.time):
        return t
    if isinstance(t, str):
        try:
            parts = [int(x) for x in t.split(":")]
            if len(parts) == 3:
                return _dt.time(parts[0], parts[1], parts[2])
            if len(parts) == 2:
                return _dt.time(parts[0], parts[1], 0)
        except:
            return None
    return None

def seconds_between(start_t, end_t):
    if not start_t or not end_t:
        return 0
    today = _dt.date.today()
    try:
        s_dt = _dt.datetime.combine(today, start_t)
        e_dt = _dt.datetime.combine(today, end_t)
        delta = (e_dt - s_dt).total_seconds()
        return int(delta) if delta > 0 else 0
    except:
        return 0

# ---------------- FACE MODEL (Lazy Loading) ----------------
facenet_model = None

def get_facenet():
    global facenet_model
    if facenet_model is None:
        facenet_model = FaceNet()
    return facenet_model

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def face_distance(a, b):
    a = a.astype("float32")
    b = b.astype("float32")
    return float(np.linalg.norm(a - b))

def capture_face_embedding(num_images=50):
    model = get_facenet()
    if face_cascade.empty():
        return False, "Face detection model not loaded."

    cap = cv2.VideoCapture("rtsp://10.154.79.119:1945/")
    if not cap.isOpened():
        return False, "Unable to access camera."

    collected = []
    try:
        while len(collected) < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad = 10
                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + w + pad, frame.shape[1])
                y2 = min(y + h + pad, frame.shape[0])
                face_img = frame[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_rgb = cv2.resize(face_rgb, (160, 160))
                emb = model.embeddings([face_rgb])[0]
                collected.append(emb)

            cv2.imshow("Face Capture - Press q to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False, "Face capture cancelled."

        cap.release()
        cv2.destroyAllWindows()
        return True, np.mean(collected, axis=0)

    except Exception as e:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        return False, f"Error: {e}"

# ---------------- ROLE SELECTION PAGE ----------------
@app.route("/", methods=["GET"])
def role_select():
    return render_template("role_select.html")

# ---------------- ADMIN LOGIN ----------------
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        admin_id = request.form.get("admin_id", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT password_hash, name FROM admins WHERE admin_id=%s", (admin_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return render_template("admin_login.html", error="Admin not found.")

        pw_hash, name = row
        if not check_password_hash(pw_hash, password):
            return render_template("admin_login.html", error="Incorrect password.")

        session.clear()
        session["is_admin"] = True
        session["admin_id"] = admin_id
        session["admin_name"] = name

        return redirect(url_for("admin_dashboard"))

    return render_template("admin_login.html")

@app.route("/admin_logout")
def admin_logout():
    session.pop("is_admin", None)
    session.pop("admin_id", None)
    session.pop("admin_name", None)
    return redirect(url_for("role_select"))

# --------------------- END OF PART 1 ---------------------
# --------------------- PART 2 ---------------------
# ADMIN DASHBOARD
@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    return render_template("admin_dashboard.html", admin_name=session.get("admin_name"))

# --------------------- ADMIN: EMPLOYEE LIST API ---------------------
@app.route("/admin/employees")
def admin_employees():
    if not session.get("is_admin"):
        return jsonify({"error": "unauthorized"}), 403

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, contact_number FROM employees ORDER BY id")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    arr = [{"id": r[0], "name": r[1], "contact": r[2]} for r in rows]
    return jsonify(arr)

# --------------------- ADMIN: VIEW EMPLOYEE ATTENDANCE ---------------------
@app.route("/admin/attendance/<emp_id>")
def admin_attendance(emp_id):
    if not session.get("is_admin"):
        return jsonify({"error": "unauthorized"}), 403

    conn = get_db_conn()
    cur = conn.cursor()

    # Get employee name
    cur.execute("SELECT name FROM employees WHERE id=%s", (emp_id,))
    r = cur.fetchone()
    if not r:
        cur.close()
        conn.close()
        return jsonify({"error": "not found"}), 404

    name = r[0]

    # Get attendance rows
    cur.execute("""
        SELECT date, in1, out1, in2, out2 
        FROM attendance 
        WHERE emp_id=%s ORDER BY date DESC LIMIT 365
    """, (emp_id,))
    raw_rows = cur.fetchall()
    cur.close()
    conn.close()

    formatted = []
    for (d, in1, out1, in2, out2) in raw_rows:
        s_in1 = time_to_str_safe(in1)
        s_out1 = time_to_str_safe(out1)
        s_in2 = time_to_str_safe(in2)
        s_out2 = time_to_str_safe(out2)

        t_in1 = parse_time_str(s_in1)
        t_out1 = parse_time_str(s_out1)
        t_in2 = parse_time_str(s_in2)
        t_out2 = parse_time_str(s_out2)

        secs1 = seconds_between(t_in1, t_out1)
        secs2 = seconds_between(t_in2, t_out2)
        total_secs = secs1 + secs2
        total_hours = round(total_secs / 3600, 2)

        formatted.append({
            "date": d.isoformat(),
            "in1": s_in1,
            "out1": s_out1,
            "in2": s_in2,
            "out2": s_out2,
            "present_seconds": total_secs,
            "present_hours": float(total_hours)
        })

    return jsonify({"name": name, "rows": formatted})

# --------------------- ADMIN: DELETE EMPLOYEE ---------------------
@app.route("/admin/delete_employee/<emp_id>", methods=["POST"])
def admin_delete_employee(emp_id):
    if not session.get("is_admin"):
        return jsonify({"error": "unauthorized"}), 403

    conn = get_db_conn()
    cur = conn.cursor()

    # Delete attendance first
    cur.execute("DELETE FROM attendance WHERE emp_id=%s", (emp_id,))
    cur.execute("DELETE FROM employees WHERE id=%s", (emp_id,))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"ok": True})

# --------------------- ADMIN: UPDATE EMPLOYEE ---------------------
@app.route("/admin/update_employee/<emp_id>", methods=["POST"])
def admin_update_employee(emp_id):
    if not session.get("is_admin"):
        return jsonify({"error": "unauthorized"}), 403

    name = request.form.get("name", "").strip()
    contact = request.form.get("contact", "").strip()

    if not name:
        return jsonify({"error": "Name required"}), 400

    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE employees 
        SET name=%s, contact_number=%s 
        WHERE id=%s
    """, (name, contact, emp_id))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"ok": True})

# --------------------- ADMIN: DOWNLOAD ANY EMPLOYEE ATTENDANCE ---------------------
@app.route("/admin/download/<emp_id>")
def admin_download(emp_id):

    if not session.get("is_admin"):
        return abort(403)

    # Fetch employee name from DB
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM employees WHERE id=%s", (emp_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    emp_name = row[0] if row else "Unknown"

    # Read date range
    start = request.args.get("start")
    end = request.args.get("end")

    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else date.today() - timedelta(days=6)
        end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
    except:
        return "Invalid date format", 400

    # Fetch attendance rows
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT date, in1, out1, in2, out2
        FROM attendance
        WHERE emp_id=%s AND date BETWEEN %s AND %s
        ORDER BY date
    """, (emp_id, start_date, end_date))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Convert to DataFrame
    df_rows = []
    for (d, in1, out1, in2, out2) in rows:
        df_rows.append({
            "date": d.isoformat(),
            "in1": time_to_str_safe(in1) or "",
            "out1": time_to_str_safe(out1) or "",
            "in2": time_to_str_safe(in2) or "",
            "out2": time_to_str_safe(out2) or ""
        })

    df = pd.DataFrame(df_rows)
    output = io.BytesIO()

    # Excel writer with formatting
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Attendance")
        writer.sheets["Attendance"] = worksheet

        # Styles
        header_format = workbook.add_format({
            "bold": True, "font_color": "white",
            "bg_color": "#4F81BD", "border": 1, "align": "center"
        })
        label_format = workbook.add_format({"bold": True, "bg_color": "#DCE6F1", "border": 1})
        value_format = workbook.add_format({"border": 1})

        # Title
        worksheet.merge_range("A1:F1", "Employee Attendance Report", header_format)

        # Employee details
        worksheet.write("A3", "Employee Name", label_format)
        worksheet.write("B3", emp_name, value_format)

        worksheet.write("A4", "Employee ID", label_format)
        worksheet.write("B4", emp_id, value_format)

        worksheet.write("A5", "From Date", label_format)
        worksheet.write("B5", str(start_date), value_format)

        worksheet.write("A6", "To Date", label_format)
        worksheet.write("B6", str(end_date), value_format)

        # Insert DataFrame starting row 8
        df.to_excel(writer, index=False, startrow=7, sheet_name="Attendance")

        # Auto column widths
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_len)

    output.seek(0)
    filename = f"{emp_id}attendance{start_date}_{end_date}.xlsx"

    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --------------------- END OF PART 2 ---------------------
# --------------------- PART 3 ---------------------

# ---------------- EMPLOYEE LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        emp_id = request.form.get("emp_id", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT name, password_hash FROM employees WHERE id=%s", (emp_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return render_template("login.html", error="Employee ID not found.")

        emp_name, password_hash = row

        if not check_password_hash(password_hash, password):
            return render_template("login.html", error="Incorrect password.")

        session.clear()
        session["emp_id"] = emp_id
        session["emp_name"] = emp_name

        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("role_select"))

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "emp_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        emp_id=session["emp_id"],
        emp_name=session["emp_name"],
        current_date=date.today().isoformat()
    )

# ---------------- TWILIO OTP (SMS) ----------------
from twilio.rest import Client

TWILIO_SID = "AC6f8a011f27b34f08f88a00f380d9c54d"
TWILIO_AUTH_TOKEN = "7824ebc957a4075e31f5ec8bf86a4a67"
TWILIO_SMS_FROM = "+13135133191"

def send_otp_sms(phone_number, otp):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=f"Your OTP is {otp}. It is valid for 5 minutes.",
        from_=TWILIO_SMS_FROM,
        to="+91" + phone_number[-10:]
    )
    return message.sid

def generate_otp(length=6):
    return "".join(random.choices("0123456789", k=length))

# ---------------- EMPLOYEE REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    try:
        emp_id = request.form.get("emp_id", "").strip()
        emp_name = request.form.get("emp_name", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm", "").strip()
        contact = request.form.get("contact", "").strip()

        if not emp_id or not emp_name or not password or not confirm or not contact:
            return jsonify({"error": "All fields are required."})

        if password != confirm:
            return jsonify({"error": "Passwords do not match."})

        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters."})

        import re
        if not re.match(r"^CS\d{3}$", emp_id):
            return jsonify({"error": "Employee ID format must be like CS001."})

        if not re.match(r"^[A-Za-z ]+$", emp_name):
            return jsonify({"error": "Name must contain only letters."})

        if not re.match(r"^[6-9]\d{9}$", contact):
            return jsonify({"error": "Invalid 10-digit mobile number."})

        normalized_contact = "+91" + contact

        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT id FROM employees WHERE id=%s", (emp_id,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return jsonify({"error": "Employee ID already registered."})

        # ------- Face capture (50 images) -------
        ok, result = capture_face_embedding(num_images=50)
        if not ok:
            cur.close()
            conn.close()
            return jsonify({"error": result})

        new_emb = result

        cur.execute("SELECT id, name, embedding FROM employees WHERE embedding IS NOT NULL")
        for other_id, other_name, emb_blob in cur.fetchall():
            try:
                existing_emb = pickle.loads(emb_blob)
                if face_distance(new_emb, existing_emb) <= 0.9:
                    cur.close()
                    conn.close()
                    return jsonify({
                        "duplicate": True,
                        "error": f"Face already registered for {other_name} ({other_id})."
                    })
            except:
                continue

        password_hash = generate_password_hash(password)
        emb_blob = pickle.dumps(new_emb.astype(np.float32))

        cur.execute("""
            INSERT INTO employees (id, name, embedding, password_hash, contact_number)
            VALUES (%s, %s, %s, %s, %s)
        """, (emp_id, emp_name, emb_blob, password_hash, normalized_contact))

        conn.commit()
        cur.close()
        conn.close()

        return render_template("success.html", emp_id=emp_id, emp_name=emp_name, count=50)

    except Exception as e:
        print("Registration error:", e)
        return jsonify({"error": "Server error. Please try again."}), 500

# ---------------- FORGOT PASSWORD ----------------
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html")

    emp_id = request.form.get("emp_id", "").strip()
    contact = request.form.get("contact", "").strip()

    if not emp_id or not contact:
        return render_template("forgot_password.html", error="All fields are required.")

    digits_only = "".join(ch for ch in contact if ch.isdigit())
    if len(digits_only) == 10:
        normalized_contact = "+91" + digits_only
    else:
        return render_template("forgot_password.html", error="Enter valid mobile number.")

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT contact_number FROM employees WHERE id=%s", (emp_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return render_template("forgot_password.html", error="ID not found.")

    db_contact = row[0]
    if db_contact != normalized_contact:
        return render_template("forgot_password.html", error="Wrong mobile number.")

    otp = generate_otp(6)
    expires_at = _dt.datetime.utcnow() + _dt.timedelta(minutes=5)

    session["fp_emp_id"] = emp_id
    session["fp_otp"] = otp
    session["fp_expires"] = expires_at.isoformat()

    send_otp_sms(db_contact, otp)
    masked = f"+91******{db_contact[-4:]}"

    return render_template("verify_otp.html", emp_id=emp_id, masked_contact=masked)

@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    emp_id = request.form.get("emp_id", "").strip()
    otp_input = request.form.get("otp", "").strip()

    session_emp = session.get("fp_emp_id")
    session_otp = session.get("fp_otp")
    exp_str = session.get("fp_expires")

    if not session_emp or not session_otp or not exp_str:
        return render_template("forgot_password.html", error="OTP expired.")

    try:
        exp = _dt.datetime.fromisoformat(exp_str)
    except:
        exp = None

    if exp and _dt.datetime.utcnow() > exp:
        return render_template("forgot_password.html", error="OTP expired.")

    if emp_id != session_emp:
        return render_template("verify_otp.html", error="Wrong ID.", emp_id=emp_id)

    if otp_input != session_otp:
        return render_template("verify_otp.html", error="Invalid OTP.", emp_id=emp_id)

    session["allow_reset_for"] = emp_id
    return render_template("reset_password.html", emp_id=emp_id)

@app.route("/reset_password", methods=["POST"])
def reset_password():
    emp_id = session.get("allow_reset_for")
    if not emp_id:
        return redirect(url_for("forgot_password"))

    password = request.form.get("password", "")
    confirm = request.form.get("confirm", "")

    if password != confirm:
        return render_template("reset_password.html", error="Passwords do not match.", emp_id=emp_id)

    if len(password) < 6:
        return render_template("reset_password.html", error="Minimum 6 characters.", emp_id=emp_id)

    password_hash = generate_password_hash(password)

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE employees SET password_hash=%s WHERE id=%s", (password_hash, emp_id))
    conn.commit()
    cur.close()
    conn.close()

    session.pop("allow_reset_for", None)

    return render_template("login.html", success="Password reset successfully.")

# ---------------- EMPLOYEE API: ATTENDANCE RANGE ----------------
@app.route("/api/attendance_range")
def attendance_range():
    # Ensure user is logged in
    emp_id = session.get("emp_id")
    is_admin = session.get("is_admin")

    # If neither admin nor employee is logged in
    if not emp_id and not is_admin:
        return jsonify({"error": "Unauthorized"}), 403

    # Determine whose data to show
    if is_admin:
        emp_id_to_fetch = request.args.get("emp_id")
        if not emp_id_to_fetch:
            return jsonify({"error": "Employee ID required for admin"}), 400
    else:
        emp_id_to_fetch = emp_id  # logged-in employee

    # Date range
    start = request.args.get("start")
    end = request.args.get("end")

    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else date.today() - timedelta(days=6)
        end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
    except Exception:
        return jsonify({"error": "Invalid date format"}), 400

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # Fetch attendance from database
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT date, in1, out1, in2, out2
        FROM attendance
        WHERE emp_id=%s AND date BETWEEN %s AND %s
        ORDER BY date
    """, (emp_id_to_fetch, start_date, end_date))

    rows = []
    for (d, in1, out1, in2, out2) in cur.fetchall():
        s_in1 = time_to_str_safe(in1)
        s_out1 = time_to_str_safe(out1)
        s_in2 = time_to_str_safe(in2)
        s_out2 = time_to_str_safe(out2)

        t_in1 = parse_time_str(s_in1)
        t_out1 = parse_time_str(s_out1)
        t_in2 = parse_time_str(s_in2)
        t_out2 = parse_time_str(s_out2)

        secs1 = seconds_between(t_in1, t_out1)
        secs2 = seconds_between(t_in2, t_out2)
        total_hours = round((secs1 + secs2) / 3600, 2)

        rows.append({
            "date": d.isoformat(),
            "in1": s_in1,
            "out1": s_out1,
            "in2": s_in2,
            "out2": s_out2,
            "present_hours": total_hours
        })

    cur.close()

    # Fetch employee name
    cur = conn.cursor()
    cur.execute("SELECT name FROM employees WHERE id=%s", (emp_id_to_fetch,))
    emp = cur.fetchone()
    emp_name = emp[0] if emp else ""
    cur.close()
    conn.close()

    return jsonify({
        "emp_id": emp_id_to_fetch,
        "name": emp_name,
        "rows": rows
    })


# ---------------- DOWNLOAD ATTENDANCE ----------------
@app.route("/download_attendance")
def download_attendance():
    if "emp_id" not in session and not session.get("is_admin"):
        return abort(401)

    emp_id = session.get("emp_id")
    if session.get("is_admin"):
        emp_id = request.args.get("emp_id") or emp_id

    # ✅ Fetch employee name
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM employees WHERE id=%s", (emp_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    emp_name = row[0] if row else "Unknown"

    # ✅ Get date range
    start = request.args.get("start")
    end = request.args.get("end")
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else date.today() - timedelta(days=6)
        end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
    except:
        return "Invalid date format", 400

    # ✅ Fetch attendance records
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT date, in1, out1, in2, out2
        FROM attendance
        WHERE emp_id=%s AND date BETWEEN %s AND %s
        ORDER BY date
    """, (emp_id, start_date, end_date))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return "No attendance records found", 404

    # ✅ Convert to DataFrame
    df_rows = []
    for (d, in1, out1, in2, out2) in rows:
        df_rows.append({
            "date": d.isoformat(),
            "in1": time_to_str_safe(in1) or "-",
            "out1": time_to_str_safe(out1) or "-",
            "in2": time_to_str_safe(in2) or "-",
            "out2": time_to_str_safe(out2) or "-"
        })

    df = pd.DataFrame(df_rows)
    output = io.BytesIO()

    # ✅ Write with proper formatting
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Attendance")
        writer.sheets["Attendance"] = worksheet

        # --- Formats ---
        title_format = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#4472C4",
            "align": "center", "valign": "vcenter", "font_size": 14
        })
        label_format = workbook.add_format({
            "bold": True, "bg_color": "#DCE6F1", "border": 1
        })
        value_format = workbook.add_format({"border": 1})
        header_format = workbook.add_format({
            "bold": True, "bg_color": "#4F81BD", "font_color": "white", "border": 1, "align": "center"
        })
        cell_format = workbook.add_format({"border": 1})

        # --- Title Row ---
        worksheet.merge_range("A1:E1", "Employee Attendance Report", title_format)

        # --- Details Section ---
        worksheet.write("A3", "Employee Name", label_format)
        worksheet.write("B3", emp_name, value_format)
        worksheet.write("A4", "Employee ID", label_format)
        worksheet.write("B4", emp_id, value_format)
        worksheet.write("A5", "From Date", label_format)
        worksheet.write("B5", str(start_date), value_format)
        worksheet.write("A6", "To Date", label_format)
        worksheet.write("B6", str(end_date), value_format)

        # --- Table Header ---
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(7, col_num, col_name, header_format)

        # --- Table Data ---
        for row_num, record in enumerate(df.values):
            for col_num, value in enumerate(record):
                worksheet.write(row_num + 8, col_num, value, cell_format)

        # --- Auto-adjust columns ---
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_len)

    output.seek(0)
    filename = f"attendance_{emp_name}_{emp_id}_{start_date}_{end_date}.xlsx"

    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/admin/register", methods=["GET", "POST"])
def admin_register():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))

    if request.method == "GET":
        # show same employee registration page
        return render_template("register.html")

    # POST => same registration logic as employee register
    try:
        emp_id = request.form.get("emp_id", "").strip()
        emp_name = request.form.get("emp_name", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm", "").strip()
        contact = request.form.get("contact", "").strip()

        if not emp_id or not emp_name or not password or not confirm or not contact:
            return jsonify({"error": "All fields are required."})

        if password != confirm:
            return jsonify({"error": "Passwords do not match."})

        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters."})

        import re
        if not re.match(r"^CS\d{3}$", emp_id):
            return jsonify({"error": "Employee ID format must be like CS001."})

        if not re.match(r"^[A-Za-z ]+$", emp_name):
            return jsonify({"error": "Name must contain only letters."})

        if not re.match(r"^[6-9]\d{9}$", contact):
            return jsonify({"error": "Invalid 10-digit mobile number."})

        normalized_contact = "+91" + contact

        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT id FROM employees WHERE id=%s", (emp_id,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return jsonify({"error": "Employee ID already registered."})

        ok, result = capture_face_embedding(num_images=50)
        if not ok:
            cur.close()
            conn.close()
            return jsonify({"error": result})

        new_emb = result

        cur.execute("SELECT id, name, embedding FROM employees WHERE embedding IS NOT NULL")
        for other_id, other_name, emb_blob in cur.fetchall():
            try:
                existing_emb = pickle.loads(emb_blob)
                if face_distance(new_emb, existing_emb) <= 0.9:
                    cur.close(); conn.close()
                    return jsonify({"duplicate": True, "error": f"Face already registered for {other_name} ({other_id})."})
            except:
                continue

        password_hash = generate_password_hash(password)
        emb_blob = pickle.dumps(new_emb.astype(np.float32))

        cur.execute(
            "INSERT INTO employees (id, name, embedding, password_hash, contact_number) "
            "VALUES (%s, %s, %s, %s, %s)",
            (emp_id, emp_name, emb_blob, password_hash, normalized_contact)
        )
        conn.commit()
        cur.close()
        conn.close()

        return render_template("success.html", emp_id=emp_id, emp_name=emp_name, count=50)

    except Exception as e:
        return jsonify({"error": "Server error. Please try again."}), 500
@app.route("/register_employee", methods=["POST"])
def register_employee():
    # call the original register() function
    return register()


import threading
import time
import os
from flask import request

@app.route("/shutdown", methods=["POST", "GET"])
def shutdown():
    """Robust shutdown: try werkzeug shutdown, otherwise exit the process."""
    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()
        return "Shutting down (werkzeug)..."
    else:
        # Spawn a thread to exit after returning a response so the client sees the message.
        def _exit_after_delay():
            time.sleep(0.5)
            try:
                # Try graceful exit
                os._exit(0)
            except SystemExit:
                pass

        threading.Thread(target=_exit_after_delay, daemon=True).start()
        return "Shutdown signal sent (fallback)."

# ---------------- START SERVER ----------------
if __name__ == "__main__":
    port = 5000
    try:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}/")
    except:
        pass

    app.run(debug=True, port=port)

# --------------------- END OF PART 3 ---------------------
