import os
import sys
import json
import argparse
import socket
import random
import time
import numpy as np
import smtplib
import hashlib
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
from email.message import EmailMessage

# --- 1. Dependency Check ---
try:
    import PIL
    import cv2
    import tensorflow as tf
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    import uvicorn
    from google import genai
    from google.genai import types
except ImportError as e:
    print(f"\n[ERROR] Missing library: {e}")
    print("Please run: pip install fastapi uvicorn tensorflow opencv-python-headless numpy python-multipart pillow google-genai pydantic mysql-connector-python")
    sys.exit(1)

# ==========================================
# 2. CONFIGURATION & RAW DATASET SETTINGS
# ==========================================
MODEL_PATH = "plant_model.tflite"
CLASSES_PATH = "classes.json"
RAW_DATASET_DIR = r"D:\Plant_AI_Project\dataset"

# >>> UPDATE YOUR GOOGLE GEMINI API KEYS HERE <<<
GEMINI_API_KEYS = [
    "YOUR_GEMINI_API_KEY_1",
    "YOUR_GEMINI_API_KEY_2"             
]

# ==========================================
# 2.5 DATABASE & OTP CONFIGURATION
# ==========================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',       # Default XAMPP username
    'password': '',       # Default XAMPP password is empty
    'database': 'plantcare_db'
}

SMTP_HOST = os.getenv("EMAIL_SERVER_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SERVER_PORT", 587))
SMTP_USER = os.getenv("EMAIL_SERVER_USER", "")
SMTP_PASS = os.getenv("EMAIL_SERVER_PASSWORD", "")
SMTP_FROM = os.getenv("EMAIL_FROM", SMTP_USER)

def get_db_connection():
    """Establishes connection to XAMPP MySQL Database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"[DB ERROR] Error connecting to MySQL: {e}")
    return None

def send_otp_email(to_email, otp_code):
    if not SMTP_USER or not SMTP_PASS:
        return False, "SMTP credentials not configured. Please set environment variables."

    msg = EmailMessage()
    msg['Subject'] = "Your PlantCare AI Login Code"
    msg['From'] = SMTP_FROM
    msg['To'] = to_email
    
    # HTML Email Body
    msg.set_content(f"Your OTP code is {otp_code}. It expires in 5 minutes.")
    msg.add_alternative(f"""
        <div style="font-family: sans-serif; padding: 20px;">
            <h2>Your Login Code</h2>
            <h1 style="color: #333; background: #f4f4f4; padding: 10px; display: inline-block; border-radius: 5px;">{otp_code}</h1>
            <p>This code will expire in 5 minutes.</p>
        </div>
    """, subtype='html')

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return True, "OTP Sent Successfully"
    except Exception as e:
        print(f"SMTP Error: {e}")
        return False, f"Failed to send email: {str(e)}"

def get_basic_treatment(raw_class_name: str) -> dict:
    """Fallback treatments in case all cloud AIs are unavailable."""
    name = raw_class_name.lower()
    res = {
        "why_it_happens": "Environmental factors or local pathogens.", 
        "how_to_control": "Consult a local agronomist."
    }
    
    if "background" in name: 
        res["why_it_happens"] = "No plant detected in the image."
        res["how_to_control"] = "Please upload a clear picture of a leaf."
    elif "healthy" in name or "normal" in name: 
        res["why_it_happens"] = "The plant is growing in optimal conditions without visible pathogen interference."
        res["how_to_control"] = "Continue maintaining your current watering, soil, and sunlight routines."
    elif "tomato" in name: 
        res["why_it_happens"] = "Often caused by fungal or bacterial spores thriving in high humidity."
        res["how_to_control"] = "Apply appropriate copper-based fungicides. Avoid overhead watering."
    
    return res

# ==========================================
# 3. DIRECT TRAINING PIPELINE (CNN)
# ==========================================
def train_direct(data_dir: str, epochs: int = 5):
    print(f"\n>>> Starting Direct CNN Training Pipeline from: {data_dir} <<<")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory '{data_dir}' not found!")
        sys.exit(1)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, validation_split=0.2 
    )

    print("Loading images directly from folders...")
    train_generator = datagen.flow_from_directory(
        data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training'
    )
    val_generator = datagen.flow_from_directory(
        data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation'
    )

    classes = list(train_generator.class_indices.keys())
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=1, restore_best_weights=True)
    
    print("\n--- Training Model... PLEASE DO NOT CLOSE THE WINDOW ---")
    model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stop])

    print("\n--- Saving Offline Mobile Model (TFLite) ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(MODEL_PATH, "wb") as f:
        f.write(converter.convert())
    print(f"\n>>> SUCCESS: Model saved directly to {MODEL_PATH}! <<<")

# ==========================================
# 4. FASTAPI HYBRID SERVER
# ==========================================
app = FastAPI(title="Hybrid Plant AI")

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open(CLASSES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False

@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...), 
    crop_name: str = Form(""),
    language: str = Form("English")
):
    contents = await file.read()
    img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    if img_bgr is None: 
        return {"status": "error", "message": "Invalid image."}
        
    if not crop_name or not crop_name.strip():
        return {"status": "error", "message": "Crop name is required. Please specify the plant name."}
    
    if not MODEL_LOADED: 
        return {
            "status": "demo", 
            "final_class": "Model Not Found", 
            "confidence": 0.0, 
            "why_it_happens": "-", 
            "how_to_control": "Run --train first.", 
            "is_healthy": False,
            "detected_plant": crop_name
        }

    # 1. Run Local CNN for Base Prediction
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    class_idx = np.argmax(preds)
    raw_class = CLASS_NAMES[class_idx]
    confidence = float(preds[class_idx])
    clean_class = raw_class.replace("___", " - ").replace("_", " ")
    
    basic_res = get_basic_treatment(raw_class)
    final_class = clean_class
    why_it_happens = basic_res["why_it_happens"]
    how_to_control = basic_res["how_to_control"]
    is_healthy = "healthy" in final_class.lower() or "normal" in final_class.lower()
    detected_plant = crop_name

    # 2. Extract and sanitize valid keys
    valid_keys = [k.strip() for k in GEMINI_API_KEYS if k.strip() and not k.startswith("YOUR_GEMINI")]
    
    if valid_keys:
        random.shuffle(valid_keys)
        
        # Optimize image to prevent rate limits
        max_dim = 800
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_optimized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_optimized = img_bgr.copy()
            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', img_optimized, encode_param)
        image_bytes = buffer.tobytes()
        
        prompt = (
            f"You are an expert agricultural pathologist. The user stated the plant in the image is a '{crop_name}'.\n"
            f"1. First, analyze the image to verify if it is actually a plant, leaf, or crop. If NOT, return EXACTLY this JSON: "
            f"{{\"is_healthy\": false, \"disease_name\": \"Invalid Image\", \"detected_plant\": \"Unknown\", \"why_it_happens\": \"This does not appear to be a plant.\", \"how_to_control\": \"Please upload a proper plant image.\"}}\n"
            f"2. If it IS a plant, identify the actual plant species/crop. If the user's '{crop_name}' is wildly incorrect, provide the correct plant name in the 'detected_plant' field.\n"
            f"3. Verify the health. An initial local CNN model suggested '{clean_class}'. Visually verify and determine the TRUE diagnosis.\n"
            f"4. Write your response entirely in {language} (except the JSON keys).\n"
            f"5. VERY IMPORTANT: Keep your summaries extremely short! Maximum 1 to 2 short sentences for 'why_it_happens' and 'how_to_control'. Do NOT write lengthy paragraphs.\n"
            f"Respond ONLY with a valid JSON object containing exactly these five keys: \"is_healthy\", \"disease_name\", \"detected_plant\", \"why_it_happens\", \"how_to_control\"."
        )

        for attempt, key in enumerate(valid_keys):
            try:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                
                if response.text:
                    json_str = response.text.strip()
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0].strip()
                        
                    ai_data = json.loads(json_str)
                    
                    final_class = ai_data.get("disease_name", final_class)
                    detected_plant = ai_data.get("detected_plant", crop_name)
                    why_it_happens = ai_data.get("why_it_happens", why_it_happens)
                    how_to_control = ai_data.get("how_to_control", how_to_control)
                    is_healthy = ai_data.get("is_healthy", is_healthy)
                    break
                    
            except Exception as e:
                err_msg = str(e).lower()
                print(f"\n[GEMINI API WARNING on Key {attempt+1}] {str(e)}")
                if "429" in err_msg or "503" in err_msg or "exhausted" in err_msg or "quota" in err_msg:
                    time.sleep(1.5)
                    continue
                else:
                    time.sleep(0.5)
                    continue

    return {
        "status": "success",
        "final_class": final_class,
        "detected_plant": detected_plant,
        "confidence": confidence,
        "why_it_happens": why_it_happens,
        "how_to_control": how_to_control,
        "is_healthy": is_healthy
    }

# ==========================================
# 4.5 AI ASSISTANT ENDPOINT (TEXT ONLY)
# ==========================================
class ChatRequest(BaseModel):
    message: str
    language: str = "English"

@app.post("/chat")
async def chat_assistant(request: ChatRequest):
    if not request.message or not request.message.strip():
        return {"status": "error", "message": "Message is empty."}
        
    valid_keys = [k.strip() for k in GEMINI_API_KEYS if k.strip() and not k.startswith("YOUR_GEMINI")]
    if not valid_keys:
        return {"status": "demo", "reply": "This is a demo response. Please configure your API keys."}

    random.shuffle(valid_keys)
    
    prompt = (
        f"You are a highly knowledgeable agricultural AI assistant helping farmers. "
        f"You MUST ONLY answer questions related to agriculture, farming, crops, plant care, plant diseases, soil, fertilizers, and weather impacting farming. "
        f"If the user asks about anything completely unrelated to agriculture (like programming, math, history, pop culture, etc), "
        f"politely decline and state that you are an agricultural assistant. "
        f"Respond entirely in {request.language} and keep your answers concise, clean, and helpful. "
        f"IMPORTANT: Do NOT use any Markdown formatting. Do not use asterisks (*), bolding (**), or bullet points. Provide plain text only. "
        f"User query: {request.message}"
    )

    for attempt, key in enumerate(valid_keys):
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
            if response.text:
                return {"status": "success", "reply": response.text.strip()}
        except Exception as e:
            err_msg = str(e).lower()
            print(f"\n[CHAT GEMINI API WARNING on Key {attempt+1}] {str(e)}")
            time.sleep(1.5 if "429" in err_msg or "503" in err_msg or "exhausted" in err_msg else 0.5)

    return {"status": "error", "message": "All AI nodes are currently busy. Please try again later."}

# ==========================================
# 4.6 DB INTEGRATION & OTP Endpoints
# ==========================================
class SignupRequest(BaseModel):
    firstName: str
    lastName: str
    email: str
    phone: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class OTPRequest(BaseModel):
    email: str

class VerifyOTPRequest(BaseModel):
    email: str
    otp: str

class ResetPasswordRequest(BaseModel):
    email: str
    otp: str
    new_password: str

class HistoryRequest(BaseModel):
    user_email: str
    crop_name: str
    disease_name: str
    status: str
    confidence_score: float

@app.post("/api/signup")
def api_signup(req: SignupRequest):
    conn = get_db_connection()
    if not conn: return {"success": False, "message": "Database connection failed. Is XAMPP running?"}
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM users WHERE email = %s", (req.email,))
        if cursor.fetchone():
            return {"success": False, "message": "Email already registered."}
            
        hashed_pw = hashlib.sha256(req.password.encode()).hexdigest()
        cursor.execute("""
            INSERT INTO users (first_name, last_name, email, phone, password) 
            VALUES (%s, %s, %s, %s, %s)
        """, (req.firstName, req.lastName, req.email, req.phone, hashed_pw))
        conn.commit()
        return {"success": True, "message": "Account created successfully!"}
    except Error as e:
        return {"success": False, "message": f"DB Error: {e}"}
    finally:
        if conn.is_connected(): conn.close()

@app.post("/api/login")
def api_login(req: LoginRequest):
    conn = get_db_connection()
    if not conn: return {"success": False, "message": "Database connection failed. Is XAMPP running?"}

    try:
        cursor = conn.cursor(dictionary=True)
        hashed_pw = hashlib.sha256(req.password.encode()).hexdigest()
        cursor.execute("SELECT first_name, last_name, email, phone FROM users WHERE email = %s AND password = %s", (req.email, hashed_pw))
        user = cursor.fetchone()
        
        if user:
            return {
                "success": True, 
                "user": {"name": f"{user['first_name']} {user['last_name']}", "email": user['email'], "phone": user['phone']}
            }
        return {"success": False, "message": "Invalid email or password."}
    except Error as e:
        return {"success": False, "message": f"DB Error: {e}"}
    finally:
        if conn.is_connected(): conn.close()

@app.post("/api/send-otp")
def api_send_otp(req: OTPRequest):
    if not req.email:
        return {"success": False, "message": "Email is required"}

    conn = get_db_connection()
    if not conn: return {"success": False, "message": "Database connection failed."}

    try:
        otp = str(random.randint(100000, 999999))
        expires_at = datetime.now() + timedelta(minutes=5)
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO otp_store (email, otp_code, expires_at) 
            VALUES (%s, %s, %s) 
            ON DUPLICATE KEY UPDATE otp_code = VALUES(otp_code), expires_at = VALUES(expires_at)
        """, (req.email, otp, expires_at))
        conn.commit()

        # Try to send email
        success, message = send_otp_email(req.email, otp)
        print(f"\n[DEV MODE] OTP for {req.email} is: {otp}\n")

        if success:
            return {"success": True, "message": "OTP sent to your email!"}
        else:
            return {"success": True, "message": f"Check your terminal for OTP (Dev Code: {otp}). Details: {message}"}
    except Error as e:
        return {"success": False, "message": f"DB Error: {e}"}
    finally:
        if conn.is_connected(): conn.close()

@app.post("/api/verify-otp")
def api_verify_otp(req: VerifyOTPRequest):
    conn = get_db_connection()
    if not conn: return {"success": False, "message": "Database connection failed."}

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT otp_code, expires_at FROM otp_store WHERE email = %s", (req.email,))
        record = cursor.fetchone()
        
        if not record:
            return {"success": False, "message": "No OTP requested for this email."}
        if datetime.now() > record["expires_at"]:
            cursor.execute("DELETE FROM otp_store WHERE email = %s", (req.email,))
            conn.commit()
            return {"success": False, "message": "OTP has expired."}
        if record["otp_code"] != req.otp:
            return {"success": False, "message": "Invalid OTP code."}

        # Clear OTP
        cursor.execute("DELETE FROM otp_store WHERE email = %s", (req.email,))
        
        # Check if user exists, auto-create if new
        cursor.execute("SELECT first_name, last_name, email, phone FROM users WHERE email = %s", (req.email,))
        user = cursor.fetchone()
        
        if not user:
            name_part = req.email.split('@')[0].capitalize()
            cursor.execute("INSERT INTO users (first_name, last_name, email, password) VALUES (%s, %s, %s, %s)", 
                           (name_part, "", req.email, "otp-login-no-pass"))
            conn.commit()
            user_payload = {"name": name_part, "email": req.email, "phone": ""}
        else:
            user_payload = {"name": f"{user['first_name']} {user['last_name']}".strip(), "email": user['email'], "phone": user['phone']}

        conn.commit()
        return {"success": True, "message": "Login successful!", "user": user_payload}
    except Error as e:
        return {"success": False, "message": f"DB Error: {e}"}
    finally:
        if conn.is_connected(): conn.close()

@app.post("/api/reset-password")
def api_reset_password(req: ResetPasswordRequest):
    conn = get_db_connection()
    if not conn: return {"success": False, "message": "Database connection failed."}

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT otp_code, expires_at FROM otp_store WHERE email = %s", (req.email,))
        record = cursor.fetchone()
        
        if not record:
            return {"success": False, "message": "No OTP requested for this email."}
        if datetime.now() > record["expires_at"]:
            cursor.execute("DELETE FROM otp_store WHERE email = %s", (req.email,))
            conn.commit()
            return {"success": False, "message": "OTP has expired."}
        if record["otp_code"] != req.otp:
            return {"success": False, "message": "Invalid OTP code."}

        # Update Password
        hashed_pw = hashlib.sha256(req.new_password.encode()).hexdigest()
        cursor.execute("UPDATE users SET password = %s WHERE email = %s", (hashed_pw, req.email))
        
        # Clear OTP
        cursor.execute("DELETE FROM otp_store WHERE email = %s", (req.email,))
        conn.commit()
        
        return {"success": True, "message": "Password updated successfully!"}
    except Error as e:
        return {"success": False, "message": f"DB Error: {e}"}
    finally:
        if conn.is_connected(): conn.close()

@app.post("/api/history")
def api_save_history(req: HistoryRequest):
    conn = get_db_connection()
    if not conn: return {"success": False}
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scan_history (user_email, crop_name, disease_name, status, confidence_score) 
            VALUES (%s, %s, %s, %s, %s)
        """, (req.user_email, req.crop_name, req.disease_name, req.status, req.confidence_score))
        conn.commit()
        return {"success": True}
    except Error as e:
        print(f"Error saving history: {e}")
        return {"success": False}
    finally:
        if conn.is_connected(): conn.close()

@app.get("/api/history/{email}")
def api_get_history(email: str):
    conn = get_db_connection()
    if not conn: return {"success": False, "history": []}
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT crop_name, disease_name, status, scan_date 
            FROM scan_history 
            WHERE user_email = %s 
            ORDER BY scan_date DESC LIMIT 15
        """, (email,))
        records = cursor.fetchall()
        for r in records:
            r['scan_date'] = r['scan_date'].strftime("%b %d, %Y %I:%M %p")
        return {"success": True, "history": records}
    except Error as e:
        print(f"Error reading history: {e}")
        return {"success": False, "history": []}
    finally:
        if conn.is_connected(): conn.close()

@app.get("/api/stats/{email}")
def api_get_stats(email: str):
    conn = get_db_connection()
    if not conn: return {"success": False, "stats": {"total":0, "diseased":0, "healthy":0}}
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as total FROM scan_history WHERE user_email = %s", (email,))
        total = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as healthy FROM scan_history WHERE user_email = %s AND status='healthy'", (email,))
        healthy = cursor.fetchone()['healthy']
        
        cursor.execute("SELECT COUNT(*) as diseased FROM scan_history WHERE user_email = %s AND status='diseased'", (email,))
        diseased = cursor.fetchone()['diseased']
        
        return {"success": True, "stats": {"total": total, "diseased": diseased, "healthy": healthy}}
    except Error as e:
        return {"success": False, "stats": {"total":0, "diseased":0, "healthy":0}}
    finally:
        if conn.is_connected(): conn.close()


# ==========================================
# 5. FRONTEND UI (HTML)
# ==========================================
HTML_CONTENT = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>PlantCare AI — Disease Diagnosis System</title>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>
  <style>
    :root {
      /* Beautiful New Color Palette */
      --green-dark: #0f381e;
      --green-mid: #1d703b;
      --green-accent: #2e9f55;
      --green-light: #52c477;
      --green-pale: #eef7f1;
      --cream: #f9fbf9;
      --warm-white: #ffffff;
      --text-dark: #121e16;
      --text-mid: #385141;
      --text-light: #6a8c76;
      --gold: #d4af37;
      --error: #e63946;
      --success: #1d703b;
      --shadow-soft: 0 4px 30px rgba(15, 56, 30, 0.08);
      --shadow-card: 0 10px 40px rgba(15, 56, 30, 0.12);
      --sidebar-w: 260px;
    }
    
    /* ----------------------------------------------------
       SCROLLBAR REMOVAL & WHITE PAGE FIX
       ---------------------------------------------------- */
    ::-webkit-scrollbar {
      width: 0px;
      height: 0px;
      background: transparent;
      display: none;
    }
    html, body {
      -ms-overflow-style: none;
      scrollbar-width: none;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow-x: hidden;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body { font-family: 'DM Sans', sans-serif; background: var(--warm-white); color: var(--text-dark); overflow-x: hidden; margin: 0; }
    
    /* SPA View Logic */
    .app-view { display: none; width: 100%; min-height: 100vh; flex-direction: column; }
    .app-view.active { display: flex; animation: fadeInView 0.4s ease; }
    @keyframes fadeInView { from { opacity: 0; } to { opacity: 1; } }

    .auth-view { display: none; min-height: 100vh; background: var(--warm-white); }
    .auth-view.active { display: flex; flex-direction: row; }
    
    .dashboard-view { display: none; min-height: 100vh; background: var(--cream); }
    .dashboard-view.active { display: flex; flex-direction: row; }

    /* =========================================
       1. HOME PAGE STYLES
       ========================================= */
    #navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 100; display: flex; align-items: center; justify-content: space-between; padding: 0 5%; height: 72px; background: rgba(255,255,255,0.92); backdrop-filter: blur(16px); border-bottom: 1px solid rgba(46,159,85,0.12); transition: box-shadow 0.3s; }
    #navbar.scrolled { box-shadow: var(--shadow-soft); }
    .nav-logo { display: flex; align-items: center; gap: 10px; text-decoration: none; }
    .nav-logo-icon { width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, var(--green-mid), var(--green-light)); display: flex; align-items: center; justify-content: center; color: #fff; font-size: 18px; }
    .nav-logo-text { font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: var(--green-dark); }
    .nav-logo-text span { color: var(--green-accent); }
    .nav-links { display: flex; align-items: center; gap: 36px; list-style: none; }
    .nav-links a { text-decoration: none; color: var(--text-mid); font-weight: 500; font-size: 0.95rem; position: relative; transition: color 0.2s; }
    .nav-links a::after { content: ''; position: absolute; bottom: -3px; left: 0; width: 0; height: 2px; background: var(--green-accent); border-radius: 2px; transition: width 0.3s; }
    .nav-links a:hover { color: var(--green-accent); }
    .nav-links a:hover::after { width: 100%; }
    .nav-cta { display: flex; align-items: center; gap: 12px; }
    .btn-outline { padding: 9px 22px; border: 1.5px solid var(--green-accent); color: var(--green-accent); border-radius: 50px; font-size: 0.92rem; font-weight: 600; text-decoration: none; transition: all 0.25s; cursor: pointer; background: transparent; }
    .btn-outline:hover { background: var(--green-accent); color: #fff; }
    .btn-fill { padding: 9px 22px; background: var(--green-mid); color: #fff; border-radius: 50px; font-size: 0.92rem; font-weight: 600; text-decoration: none; transition: all 0.25s; cursor: pointer; border: none; }
    .btn-fill:hover { background: var(--green-dark); transform: translateY(-1px); box-shadow: 0 4px 16px rgba(29,112,59,0.35); }
    .hamburger { display: none; flex-direction: column; gap: 5px; cursor: pointer; background: none; border: none; }
    .hamburger span { display: block; width: 24px; height: 2px; background: var(--text-dark); border-radius: 2px; transition: all 0.3s; }

    .hero { height: 100vh; display: flex; align-items: center; padding: 120px 5% 80px; background: var(--warm-white); position: relative; overflow: hidden; flex-shrink: 0;}
    .hero-bg { position: absolute; inset: 0; z-index: 0; background: radial-gradient(ellipse 70% 60% at 75% 50%, rgba(82,196,119,0.15) 0%, transparent 70%), radial-gradient(ellipse 40% 40% at 20% 80%, rgba(46,159,85,0.10) 0%, transparent 60%); }
    .hero-leaf-1, .hero-leaf-2, .hero-leaf-3 { position: absolute; pointer-events: none; opacity: 0.12; z-index: 0; }
    .hero-leaf-1 { top: 10%; right: 5%; font-size: 160px; color: var(--green-accent); transform: rotate(-20deg); animation: float 7s ease-in-out infinite; }
    .hero-leaf-2 { bottom: 15%; right: 20%; font-size: 80px; color: var(--green-mid); transform: rotate(30deg); animation: float 9s ease-in-out infinite 2s; }
    .hero-leaf-3 { top: 60%; left: 3%; font-size: 60px; color: var(--green-light); transform: rotate(-10deg); animation: float 8s ease-in-out infinite 1s; }
    @keyframes float { 0%,100%{transform:translateY(0) rotate(-20deg)} 50%{transform:translateY(-18px) rotate(-20deg)} }
    .hero-content { position: relative; z-index: 1; max-width: 640px; }
    .hero-badge { display: inline-flex; align-items: center; gap: 8px; background: var(--green-pale); color: var(--green-dark); padding: 6px 16px; border-radius: 50px; font-size: 0.82rem; font-weight: 600; margin-bottom: 24px; border: 1px solid rgba(46,159,85,0.25); animation: fadeUp 0.6s ease both; }
    .hero-title { font-family: 'Playfair Display', serif; font-size: clamp(2.6rem, 5vw, 4rem); font-weight: 900; line-height: 1.12; color: var(--green-dark); margin-bottom: 22px; animation: fadeUp 0.7s ease 0.1s both; }
    .hero-title em { font-style: italic; color: var(--green-accent); }
    .hero-desc { font-size: 1.08rem; color: var(--text-mid); line-height: 1.75; margin-bottom: 36px; font-weight: 300; animation: fadeUp 0.7s ease 0.2s both; }
    .hero-actions { display: flex; gap: 14px; flex-wrap: wrap; animation: fadeUp 0.7s ease 0.3s both; }
    .btn-hero-primary { padding: 14px 32px; background: var(--green-mid); color: #fff; border-radius: 50px; font-size: 1rem; font-weight: 600; text-decoration: none; transition: all 0.25s; display: flex; align-items: center; gap: 9px; box-shadow: 0 6px 24px rgba(29,112,59,0.32); border: none; cursor: pointer; }
    .btn-hero-primary:hover { background: var(--green-dark); transform: translateY(-2px); box-shadow: 0 10px 32px rgba(29,112,59,0.42); }
    .btn-hero-secondary { padding: 14px 32px; border: 1.5px solid rgba(46,159,85,0.4); color: var(--green-mid); border-radius: 50px; font-size: 1rem; font-weight: 600; text-decoration: none; transition: all 0.25s; display: flex; align-items: center; gap: 9px; }
    .btn-hero-secondary:hover { background: var(--green-pale); border-color: var(--green-accent); }
    .hero-stats { display: flex; gap: 36px; margin-top: 52px; animation: fadeUp 0.7s ease 0.4s both; }
    .stat-num { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: var(--green-dark); }
    .stat-label { font-size: 0.82rem; color: var(--text-light); font-weight: 500; margin-top: 2px; }

    section { padding: 96px 5%; flex-shrink: 0; }
    .section-label { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: var(--green-accent); margin-bottom: 12px; }
    .section-title { font-family: 'Playfair Display', serif; font-size: clamp(1.9rem, 3.5vw, 2.8rem); font-weight: 700; color: var(--green-dark); line-height: 1.2; }
    .section-title em { font-style: italic; color: var(--green-accent); }
    .section-desc { font-size: 1.05rem; color: var(--text-mid); line-height: 1.78; font-weight: 300; max-width: 560px; margin-top: 14px; }

    @keyframes fadeUp { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:translateY(0)} }
    .fade-up { opacity: 0; transform: translateY(28px); transition: opacity 0.7s ease, transform 0.7s ease; }
    .fade-up.visible { opacity: 1; transform: translateY(0); }

    /* =========================================
       2. AUTH PAGES (Login & Signup) STYLES
       ========================================= */
    .auth-left { width: 45%; background: linear-gradient(160deg, var(--green-dark) 0%, var(--green-mid) 50%, var(--green-accent) 100%); display: flex; flex-direction: column; justify-content: space-between; padding: 48px; position: relative; overflow: hidden; }
    .auth-left::before { content: ''; position: absolute; inset: 0; background: url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Ccircle cx='40' cy='40' r='38'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat; }
    .auth-left-orb { position: absolute; border-radius: 50%; background: rgba(82,196,119,0.15); filter: blur(40px); }
    .orb1 { width: 300px; height: 300px; bottom: -80px; left: -80px; }
    .orb2 { width: 180px; height: 180px; top: 80px; right: 40px; background: rgba(255,255,255,0.07); }
    .auth-brand { position: relative; z-index: 1; display: flex; align-items: center; gap: 12px; text-decoration: none; cursor: pointer; color: white; font-weight: bold; font-size: 1.25rem; font-family: 'Playfair Display', serif;}
    .auth-hero { position: relative; z-index: 1; }
    .auth-hero-icon { font-size: 64px; color: rgba(255,255,255,0.2); margin-bottom: 20px; }
    .auth-hero h2 { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 900; color: #fff; line-height: 1.2; margin-bottom: 14px; }
    .auth-hero p { color: rgba(255,255,255,0.65); font-size: 0.95rem; line-height: 1.72; }
    .auth-features { position: relative; z-index: 1; display: flex; flex-direction: column; gap: 14px; }
    .auth-feature { display: flex; align-items: center; gap: 12px; }
    .auth-feature i { color: var(--green-light); font-size: 14px; }
    .auth-feature span { color: rgba(255,255,255,0.75); font-size: 0.9rem; }
    
    .left-steps { position: relative; z-index: 1; display: flex; flex-direction: column; gap: 16px; }
    .left-step { display: flex; align-items: flex-start; gap: 14px; }
    .left-step-num { width: 28px; height: 28px; border-radius: 50%; background: rgba(255,255,255,0.15); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 0.78rem; font-weight: 700; flex-shrink: 0; margin-top: 2px; }
    .left-step-text { color: rgba(255,255,255,0.72); font-size: 0.88rem; line-height: 1.55; }
    .left-step-text strong { color: #fff; display: block; font-size: 0.92rem; margin-bottom: 2px; }

    .auth-right { flex: 1; display: flex; align-items: center; justify-content: center; padding: 40px 36px; overflow-y: auto; }
    .auth-form-wrap { width: 100%; max-width: 440px; }
    .auth-form-wrap h1 { font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 700; color: var(--green-dark); margin-bottom: 6px; }
    .auth-form-wrap .subtitle { color: var(--text-light); font-size: 0.93rem; margin-bottom: 28px; }
    .auth-form-wrap .subtitle a { color: var(--green-accent); text-decoration: none; font-weight: 600; cursor: pointer; }

    .tab-switcher { display: flex; background: var(--cream); border-radius: 12px; padding: 4px; margin-bottom: 28px; }
    .tab-btn { flex: 1; padding: 10px; border: none; background: none; border-radius: 9px; font-family: 'DM Sans', sans-serif; font-size: 0.9rem; font-weight: 600; color: var(--text-light); cursor: pointer; transition: all 0.2s; }
    .tab-btn.active { background: #fff; color: var(--green-dark); box-shadow: 0 2px 10px rgba(0,0,0,0.08); }

    .divider { display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }
    .divider::before, .divider::after { content: ''; flex: 1; height: 1px; background: rgba(46,159,85,0.15); }
    .divider span { font-size: 0.8rem; color: var(--text-light); font-weight: 500; white-space: nowrap; }
    
    .form-group { margin-bottom: 16px; position: relative; }
    .form-group label { display: block; font-size: 0.83rem; font-weight: 600; color: var(--text-mid); margin-bottom: 6px; }
    .input-wrap { position: relative; display: flex; align-items: center;}
    .input-icon { position: absolute; left: 14px; top: 50%; transform: translateY(-50%); color: var(--text-light); font-size: 14px; pointer-events: none; }
    .form-group input, .form-group textarea, .form-group select { width: 100%; padding: 12px 16px 12px 40px; border: 1.5px solid rgba(46,159,85,0.2); border-radius: 11px; font-family: 'DM Sans', sans-serif; font-size: 0.92rem; color: var(--text-dark); background: var(--warm-white); transition: all 0.2s; outline: none; }
    .form-group input:focus, .form-group textarea:focus, .form-group select:focus { border-color: var(--green-accent); box-shadow: 0 0 0 3px rgba(46,159,85,0.12); }
    .form-group input.error { border-color: var(--error); }
    .toggle-pass { position: absolute; right: 12px; top: 50%; transform: translateY(-50%); color: var(--text-light); cursor: pointer; font-size: 14px; background: none; border: none; }
    .error-msg { font-size: 0.78rem; color: var(--error); margin-top: 4px; display: none; }
    .error-msg.show { display: block; }
    
    .password-strength { margin-top: 6px; }
    .strength-bar { height: 4px; border-radius: 2px; background: #e0e0e0; margin-bottom: 4px; overflow: hidden; }
    .strength-fill { height: 100%; border-radius: 2px; transition: width 0.3s, background 0.3s; width: 0%; }
    .strength-label { font-size: 0.75rem; color: var(--text-light); }
    .terms { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 20px; }
    .terms input[type="checkbox"] { accent-color: var(--green-accent); width: 15px; height: 15px; margin-top: 2px; flex-shrink: 0; }
    .terms span { font-size: 0.84rem; color: var(--text-mid); line-height: 1.5; }
    .terms span a { color: var(--green-accent); text-decoration: none; font-weight: 600; }
    
    .form-footer { display: flex; justify-content: space-between; align-items: center; margin-bottom: 22px; }
    .remember { display: flex; align-items: center; gap: 8px; cursor: pointer; }
    .remember input[type="checkbox"] { accent-color: var(--green-accent); width: 15px; height: 15px; }
    .remember span { font-size: 0.87rem; color: var(--text-mid); }
    .forgot-link { font-size: 0.87rem; color: var(--green-accent); text-decoration: none; font-weight: 600; }

    .btn-auth { width: 100%; padding: 14px; background: var(--green-mid); color: #fff; border: none; border-radius: 12px; font-family: 'DM Sans', sans-serif; font-size: 1rem; font-weight: 700; cursor: pointer; transition: all 0.25s; display: flex; align-items: center; justify-content: center; gap: 9px; }
    .btn-auth:hover { background: var(--green-dark); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(29,112,59,0.35); }
    .btn-auth:disabled { opacity: 0.6; cursor: not-allowed; }
    
    .alert-box { padding: 12px 16px; border-radius: 10px; margin-bottom: 16px; font-size: 0.88rem; font-weight: 500; display: none; }
    .alert-box.error { background: #fff0f2; color: var(--error); border: 1px solid rgba(230,57,70,0.2); display: flex; align-items: center; gap: 8px; }
    .alert-box.success { background: var(--green-pale); color: var(--green-dark); border: 1px solid rgba(46,159,85,0.25); display: flex; align-items: center; gap: 8px; }

    /* =========================================
       3. DASHBOARD STYLES
       ========================================= */
    .sidebar { width: var(--sidebar-w); background: var(--green-dark); display: flex; flex-direction: column; position: fixed; top: 0; left: 0; bottom: 0; z-index: 50; transition: transform 0.3s; }
    .sidebar-brand { padding: 28px 24px 20px; border-bottom: 1px solid rgba(255,255,255,0.08); cursor: pointer; color: white;}
    .sidebar-brand > div { display: flex; align-items: center; gap: 10px; text-decoration: none; }
    .sidebar-nav { flex: 1; padding: 20px 0; overflow-y: auto; }
    .nav-section-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; color: rgba(255,255,255,0.35); padding: 8px 24px 6px; }
    .sidebar-nav-item { display: flex; align-items: center; gap: 12px; padding: 11px 24px; text-decoration: none; color: rgba(255,255,255,0.62); font-size: 0.9rem; font-weight: 500; transition: all 0.2s; cursor: pointer; border: none; background: none; width: 100%; text-align: left; }
    .sidebar-nav-item:hover { color: #fff; background: rgba(255,255,255,0.07); }
    .sidebar-nav-item.active { color: #fff; background: rgba(82,196,119,0.18); border-right: 3px solid var(--green-light); }
    .sidebar-nav-item i { width: 18px; text-align: center; font-size: 15px; }
    .sidebar-nav-item .badge { margin-left: auto; background: var(--green-accent); color: #fff; font-size: 0.7rem; font-weight: 700; padding: 2px 7px; border-radius: 50px; }
    .sidebar-user { padding: 20px 24px; border-top: 1px solid rgba(255,255,255,0.08); }
    .sidebar-user-info { display: flex; align-items: center; gap: 12px; }
    .user-avatar { width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, var(--green-accent), var(--green-light)); display: flex; align-items: center; justify-content: center; color: #fff; font-size: 15px; font-weight: 700; flex-shrink: 0; }
    .user-name { font-size: 0.88rem; font-weight: 600; color: #fff; }
    .user-role { font-size: 0.75rem; color: rgba(255,255,255,0.45); }
    .btn-logout { margin-left: auto; background: none; border: none; color: rgba(255,255,255,0.4); cursor: pointer; font-size: 15px; transition: color 0.2s; }
    .btn-logout:hover { color: #fff; }

    .main { margin-left: var(--sidebar-w); flex: 1; display: flex; flex-direction: column; min-height: 100vh; }
    .topbar { background: #fff; height: 68px; display: flex; align-items: center; justify-content: space-between; padding: 0 36px; border-bottom: 1px solid rgba(46,159,85,0.1); position: sticky; top: 0; z-index: 40; box-shadow: 0 2px 12px rgba(0,0,0,0.04); }
    .topbar-left { display: flex; align-items: center; gap: 12px; }
    .sidebar-toggle { background: none; border: none; cursor: pointer; font-size: 18px; color: var(--text-mid); display: none; padding: 6px; }
    .topbar-title { font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: var(--green-dark); }
    .topbar-right { display: flex; align-items: center; gap: 16px; }
    
    /* Multilingual Select Wrap */
    .lang-select-wrap { position: relative; display: flex; align-items: center; background: var(--warm-white); border: 1px solid rgba(46,159,85,0.2); border-radius: 8px; padding: 4px 10px; cursor: pointer; }
    .lang-select-wrap i { color: var(--green-accent); font-size: 14px; margin-right: 6px; }
    .lang-select { background: transparent; border: none; font-family: 'DM Sans', sans-serif; font-size: 0.85rem; font-weight: 600; color: var(--text-dark); outline: none; cursor: pointer; appearance: none; padding-right: 14px; }
    .lang-select-wrap::after { content: '▼'; font-family: 'Font Awesome 6 Free'; font-weight: 900; position: absolute; right: 10px; font-size: 10px; color: var(--text-light); pointer-events: none; }

    .topbar-notif { position: relative; background: none; border: none; cursor: pointer; font-size: 17px; color: var(--text-mid); }
    .notif-dot { position: absolute; top: -2px; right: -2px; width: 8px; height: 8px; border-radius: 50%; background: #e63946; border: 1.5px solid #fff; }
    .topbar-user { display: flex; align-items: center; gap: 10px; cursor: pointer; padding: 6px 12px; border-radius: 50px; transition: background 0.2s; position: relative; }
    .topbar-user:hover { background: var(--green-pale); }
    .topbar-avatar { width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, var(--green-accent), var(--green-light)); display: flex; align-items: center; justify-content: center; color: #fff; font-size: 13px; font-weight: 700; }
    .topbar-uname { font-size: 0.88rem; font-weight: 600; color: var(--text-dark); }
    .topbar-uemail { font-size: 0.75rem; color: var(--text-light); }
    .user-dropdown { position: absolute; top: calc(100% + 8px); right: 0; background: #fff; border-radius: 14px; box-shadow: 0 12px 40px rgba(0,0,0,0.14); border: 1px solid rgba(46,159,85,0.1); width: 220px; overflow: hidden; display: none; z-index: 100; }
    .user-dropdown.open { display: block; animation: fadeDown 0.2s ease; }
    @keyframes fadeDown { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }
    .dropdown-header { padding: 14px 16px 10px; border-bottom: 1px solid rgba(46,159,85,0.1); }
    .dropdown-header .name { font-weight: 700; color: var(--text-dark); font-size: 0.92rem; }
    .dropdown-header .email { font-size: 0.78rem; color: var(--text-light); }
    .dropdown-item { display: flex; align-items: center; gap: 10px; padding: 10px 16px; text-decoration: none; color: var(--text-mid); font-size: 0.88rem; transition: background 0.15s; cursor: pointer; background: none; border: none; width: 100%; }
    .dropdown-item:hover { background: var(--green-pale); color: var(--green-dark); }
    .dropdown-item.danger { color: #e63946; }
    .dropdown-item.danger:hover { background: #fff0f2; }

    .content { padding: 32px 36px; flex: 1; }
    .dashboard-page { display: none; }
    .dashboard-page.active { display: block; }

    .greeting-card { background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, var(--green-accent) 100%); border-radius: 20px; padding: 32px 36px; margin-bottom: 28px; display: flex; justify-content: space-between; align-items: center; position: relative; overflow: hidden; }
    .greeting-card::before { content: ''; position: absolute; right: -40px; top: -40px; width: 200px; height: 200px; border-radius: 50%; background: rgba(255,255,255,0.06); }
    .greeting-card::after { content: ''; position: absolute; right: 60px; bottom: -60px; width: 160px; height: 160px; border-radius: 50%; background: rgba(82,196,119,0.12); }
    .greeting-text h2 { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; color: #fff; margin-bottom: 6px; }
    .greeting-text p { color: rgba(255,255,255,0.7); font-size: 0.93rem; }
    .greeting-icon { font-size: 56px; color: rgba(255,255,255,0.2); position: relative; z-index: 1; }

    .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 28px; }
    .stat-card { background: #fff; border-radius: 16px; padding: 22px 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); border: 1px solid rgba(46,159,85,0.08); display: flex; align-items: center; gap: 16px; }
    .stat-icon-wrap { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }
    .stat-icon-wrap.green { background: var(--green-pale); color: var(--green-mid); }
    .stat-icon-wrap.gold { background: #fff8e7; color: var(--gold); }
    .stat-icon-wrap.blue { background: #e8f4fd; color: #2980b9; }
    .stat-icon-wrap.red { background: #fff0f2; color: #e63946; }
    .stat-num { font-family: 'Playfair Display', serif; font-size: 1.7rem; font-weight: 700; color: var(--green-dark); }
    .stat-lbl { font-size: 0.78rem; color: var(--text-light); font-weight: 500; }

    .diagnose-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 28px; }
    .card { background: #fff; border-radius: 20px; padding: 28px; box-shadow: 0 2px 16px rgba(0,0,0,0.05); border: 1px solid rgba(46,159,85,0.08); }
    .card-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
    .card-header-icon { width: 38px; height: 38px; border-radius: 10px; background: var(--green-pale); color: var(--green-mid); display: flex; align-items: center; justify-content: center; font-size: 16px; }
    .card-title { font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; color: var(--green-dark); }
    .card-subtitle { font-size: 0.8rem; color: var(--text-light); }

    /* Upload Zone & Inputs */
    .upload-zone { border: 2px dashed rgba(46,159,85,0.35); border-radius: 16px; padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.2s; background: var(--cream); position: relative; }
    .upload-zone:hover, .upload-zone.dragover { border-color: var(--green-accent); background: var(--green-pale); }
    .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
    .upload-icon { font-size: 44px; color: var(--green-accent); margin-bottom: 12px; }
    .upload-text { font-size: 0.95rem; font-weight: 600; color: var(--text-mid); margin-bottom: 5px; }
    .upload-hint { font-size: 0.8rem; color: var(--text-light); }
    .preview-wrap { display: none; margin-top: 16px; position: relative; }
    .preview-wrap img { width: 100%; max-height: 200px; object-fit: cover; border-radius: 12px; border: 2px solid var(--green-pale); }
    .preview-remove { position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,0.5); color: #fff; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; font-size: 12px; }
    
    .voice-btn { background: var(--green-pale); border: 1px solid var(--green-accent); color: var(--green-dark); border-radius: 11px; padding: 0 16px; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; justify-content: center; height: 46px; }
    .voice-btn:hover { background: var(--green-accent); color: #fff; }
    .voice-btn.recording { background: #e63946; border-color: #e63946; color: #fff; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }

    .btn-detect { width: 100%; margin-top: 16px; padding: 14px; background: var(--green-mid); color: #fff; border: none; border-radius: 12px; font-family: 'DM Sans', sans-serif; font-size: 0.97rem; font-weight: 700; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 9px; transition: all 0.25s; opacity: 0.5; pointer-events: none; }
    .btn-detect.ready { opacity: 1; pointer-events: all; }
    .btn-detect.ready:hover { background: var(--green-dark); transform: translateY(-2px); box-shadow: 0 6px 20px rgba(29,112,59,0.35); }

    .loading-wrap { display: none; text-align: center; padding: 32px 20px; }
    .loading-wrap.show { display: block; animation: fadeIn 0.3s ease; }
    .loader-rings { position: relative; width: 64px; height: 64px; margin: 0 auto 16px; }
    .loader-rings::before, .loader-rings::after { content: ''; position: absolute; inset: 0; border-radius: 50%; border: 3px solid transparent; }
    .loader-rings::before { border-top-color: var(--green-accent); animation: spin 1s linear infinite; }
    .loader-rings::after { border-bottom-color: var(--green-light); animation: spin 1.4s linear infinite reverse; inset: 8px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loader-rings i { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: var(--green-mid); font-size: 20px; }
    .loading-text { font-size: 0.95rem; font-weight: 600; color: var(--green-dark); }
    .loading-steps { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
    .loading-step { font-size: 0.8rem; color: var(--text-light); display: flex; align-items: center; gap: 7px; justify-content: center; opacity: 0; }
    .loading-step.show { opacity: 1; animation: fadeIn 0.4s ease; }
    .loading-step i { color: var(--green-accent); }
    @keyframes fadeIn { from{opacity:0} to{opacity:1} }

    .result-wrap { display: none; animation: fadeUp 0.5s ease; }
    .result-wrap.show { display: block; }
    .result-header { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 16px; }
    .result-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 50px; font-size: 0.78rem; font-weight: 700; }
    .result-badge.diseased { background: #fff0f2; color: #e63946; border: 1px solid rgba(230,57,70,0.2); }
    .result-badge.healthy { background: var(--green-pale); color: var(--green-dark); border: 1px solid rgba(46,159,85,0.25); }
    .disease-name { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700; color: var(--green-dark); margin-bottom: 2px; text-transform: capitalize; }
    .detected-plant-name { font-size: 0.85rem; color: var(--text-light); margin-bottom: 12px; }
    .confidence-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
    .conf-track { flex: 1; height: 6px; background: #e8e8e8; border-radius: 3px; overflow: hidden; }
    .conf-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--green-accent), var(--green-light)); transition: width 1s ease; }
    .conf-label { font-size: 0.8rem; font-weight: 700; color: var(--green-mid); white-space: nowrap; }
    
    .solution-block { background: var(--cream); border-radius: 12px; padding: 16px; border: 1px solid rgba(46,159,85,0.15); margin-bottom: 16px; border-left: 4px solid var(--green-accent); transition: background 0.3s, border-left-color 0.3s; }
    .solution-block h4 { font-size: 0.85rem; font-weight: 700; color: var(--green-dark); margin-bottom: 8px; display: flex; align-items: center; gap: 7px; }
    .solution-block p { font-size: 0.9rem; color: var(--text-mid); line-height: 1.65; white-space: pre-wrap; }
    
    .treatment-tags { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; }
    .tag { padding: 4px 12px; background: rgba(46,159,85,0.12); color: var(--green-mid); border-radius: 50px; font-size: 0.78rem; font-weight: 600; text-transform: capitalize; }
    .btn-new-scan { margin-top: 16px; padding: 10px 20px; background: none; border: 1.5px solid var(--green-accent); color: var(--green-accent); border-radius: 10px; font-family: 'DM Sans', sans-serif; font-size: 0.87rem; font-weight: 600; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 7px; width:100%; justify-content:center; }
    .btn-new-scan:hover { background: var(--green-pale); }

    .history-list { display: flex; flex-direction: column; gap: 12px; }
    .history-item { display: flex; align-items: center; gap: 14px; padding: 14px; background: var(--cream); border-radius: 12px; transition: all 0.2s; cursor: pointer; }
    .history-item:hover { background: var(--green-pale); }
    .history-thumb { width: 48px; height: 48px; border-radius: 10px; background: var(--green-pale); display: flex; align-items: center; justify-content: center; font-size: 20px; color: var(--green-accent); flex-shrink: 0; }
    .history-info { flex: 1; }
    .history-name { font-weight: 600; color: var(--text-dark); font-size: 0.9rem; }
    .history-date { font-size: 0.77rem; color: var(--text-light); margin-top: 2px; }
    .history-status { font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 50px; }
    .status-diseased { background: #fff0f2; color: #e63946; }
    .status-healthy { background: var(--green-pale); color: var(--green-dark); }

    .profile-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 24px; }
    .profile-card { background: #fff; border-radius: 20px; padding: 32px; box-shadow: 0 2px 16px rgba(0,0,0,0.05); border: 1px solid rgba(46,159,85,0.08); text-align: center; }
    .profile-avatar-lg { width: 90px; height: 90px; border-radius: 50%; background: linear-gradient(135deg, var(--green-mid), var(--green-light)); display: flex; align-items: center; justify-content: center; color: #fff; font-size: 32px; font-weight: 700; margin: 0 auto 16px; border: 4px solid var(--green-pale); }
    .profile-name { font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; color: var(--green-dark); }
    .profile-email { font-size: 0.87rem; color: var(--text-light); margin-top: 5px; }
    .profile-member { margin-top: 16px; padding: 8px 16px; background: var(--green-pale); border-radius: 50px; font-size: 0.8rem; font-weight: 600; color: var(--green-mid); display: inline-block; }
    .profile-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 20px; }
    .profile-stat { padding: 12px; background: var(--cream); border-radius: 10px; }
    .profile-stat-n { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700; color: var(--green-dark); }
    .profile-stat-l { font-size: 0.72rem; color: var(--text-light); }
    .profile-info-card { background: #fff; border-radius: 20px; padding: 32px; box-shadow: 0 2px 16px rgba(0,0,0,0.05); border: 1px solid rgba(46,159,85,0.08); }
    .info-group { margin-bottom: 20px; }
    .info-label { font-size: 0.8rem; font-weight: 700; color: var(--text-light); text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 5px; }
    .info-val { font-size: 0.95rem; color: var(--text-dark); font-weight: 500; }
    .info-divider { border: none; border-top: 1px solid rgba(46,159,85,0.1); margin: 20px 0; }

    /* =========================================
       CHAT ASSISTANT STYLES
       ========================================= */
    .chat-container { display: flex; flex-direction: column; height: 60vh; background: #fff; border-radius: 16px; border: 1px solid rgba(46,159,85,0.1); overflow: hidden; margin-top: 16px; }
    .chat-messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; scrollbar-width: none; -ms-overflow-style: none;}
    .chat-messages::-webkit-scrollbar { display: none; }
    .chat-msg { max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word;}
    .chat-msg.user { align-self: flex-end; background: var(--green-mid); color: #fff; border-bottom-right-radius: 4px; }
    .chat-msg.ai { align-self: flex-start; background: var(--green-pale); color: var(--green-dark); border-bottom-left-radius: 4px; border: 1px solid rgba(46,159,85,0.2); }
    .chat-input-wrap { display: flex; padding: 16px; background: #fff; border-top: 1px solid rgba(46,159,85,0.1); gap: 10px; }
    .chat-input { flex: 1; padding: 12px 16px; border: 1.5px solid rgba(46,159,85,0.2); border-radius: 50px; outline: none; font-family: 'DM Sans', sans-serif; font-size: 0.92rem; transition: border-color 0.2s;}
    .chat-input:focus { border-color: var(--green-accent); }
    .chat-btn { width: 44px; height: 44px; border-radius: 50%; background: var(--green-mid); color: #fff; border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; flex-shrink: 0; }
    .chat-btn:hover { background: var(--green-dark); transform: scale(1.05); }
    .chat-btn.recording { background: #e63946 !important; border-color: #e63946 !important; color: #fff !important; animation: pulse 1.5s infinite; }

    /* MEDIA QUERIES */
    @media(max-width:1100px){ .stats-row{grid-template-columns:repeat(2,1fr)} .diagnose-grid{grid-template-columns:1fr} }
    @media(max-width:1024px){
      .steps-grid{grid-template-columns:repeat(2,1fr)}
      .features-grid{grid-template-columns:repeat(2,1fr)}
    }
    @media(max-width:768px){
      .about-grid, .contact-grid, .footer-grid {grid-template-columns:1fr}
      .form-row{grid-template-columns:1fr}
      .steps-grid, .features-grid{grid-template-columns:1fr}
      .nav-links, .nav-cta{display:none}
      .hamburger{display:flex}
      .hero-stats{gap:24px}
      .about-visual{height:280px}
      .footer-grid{grid-template-columns:1fr 1fr}
      .auth-left{display:none}
      .auth-right{padding:32px 24px}
      .sidebar{transform:translateX(-100%)}
      .sidebar.open{transform:translateX(0)}
      .main{margin-left:0}
      .sidebar-toggle{display:flex}
      .content{padding:20px 18px}
      .profile-grid{grid-template-columns:1fr}
      .greeting-card{flex-direction:column;gap:12px;text-align:center}
      .greeting-icon{display:none}
      .lang-select-wrap { display: none; }
    }
    @media(max-width:480px){
      .footer-grid{grid-template-columns:1fr}
      .hero-title{font-size:2.1rem}
      .stats-row{grid-template-columns:1fr}
    }
  </style>
</head>
<body>

  <!-- =========================================
       VIEW: HOME
       ========================================= -->
  <div id="view-home" class="app-view">
    <nav id="navbar">
      <a href="#" onclick="navigate('view-home'); return false;" class="nav-logo">
        <div class="nav-logo-icon"><i class="fa-solid fa-seedling"></i></div>
        <span class="nav-logo-text">Plant<span>Care</span> AI</span>
      </a>
      <ul class="nav-links">
        
      </ul>
      <div class="nav-cta">
        <button onclick="navigate('view-login')" class="btn-outline">Sign Up</button>
        <button onclick="navigate('view-signup')" class="btn-fill">Get Started</button>
      </div>
      <button class="hamburger" id="hamburger" aria-label="Menu">
        <span></span><span></span><span></span>
      </button>
    </nav>

    <section class="hero">
      <div class="hero-bg"></div>
      <i class="fa-solid fa-leaf hero-leaf-1"></i>
      <i class="fa-solid fa-spa hero-leaf-2"></i>
      <i class="fa-solid fa-seedling hero-leaf-3"></i>
      <div class="hero-content">
        <div class="hero-badge"><i class="fa-solid fa-circle-check"></i> AI-Powered Plant Health Intelligence</div>
        <h1 class="hero-title">Detect Plant Diseases with <em>Precision & Speed</em></h1>
        <p class="hero-desc">Upload a photo of any plant and our advanced AI instantly analyzes it, identifies diseases, and provides expert treatment recommendations — so your crops can thrive.</p>
        <div class="hero-actions">
          <button onclick="navigate('view-signup')" class="btn-hero-primary"><i class="fa-solid fa-microscope"></i> Start Diagnosing Free</button>
          
        </div>
        <div class="hero-stats">
          <div class="stat-item"><div class="stat-num">50+</div><div class="stat-label">Diseases Detected</div></div>
          <div class="stat-item"><div class="stat-num">96%</div><div class="stat-label">Accuracy Rate</div></div>
          <div class="stat-item"><div class="stat-num">10K+</div><div class="stat-label">Farmers Helped</div></div>
        </div>
      </div>
    </section>
  </div>

  <!-- =========================================
       VIEW: LOGIN
       ========================================= -->
  <div id="view-login" class="app-view auth-view">
    <div class="auth-left">
      <div class="auth-left-orb orb1"></div>
      <div class="auth-left-orb orb2"></div>
      <div class="auth-brand" onclick="navigate('view-home')">
        <div class="brand-icon"><i class="fa-solid fa-seedling"></i></div>
        <span class="brand-name">PlantCare AI</span>
      </div>
      <div class="auth-hero">
        <div class="auth-hero-icon"><i class="fa-solid fa-leaf"></i></div>
        <h2>Welcome Back to Your Plant Health Dashboard</h2>
        <p>Sign in to continue diagnosing plant diseases and protecting your crops with AI-powered precision.</p>
      </div>
      <div class="auth-features">
        <div class="auth-feature"><i class="fa-solid fa-check-circle"></i><span>Instant AI disease detection</span></div>
        <div class="auth-feature"><i class="fa-solid fa-check-circle"></i><span>Expert treatment recommendations</span></div>
        <div class="auth-feature"><i class="fa-solid fa-check-circle"></i><span>50+ plant diseases covered</span></div>
        <div class="auth-feature"><i class="fa-solid fa-check-circle"></i><span>96% detection accuracy</span></div>
      </div>
    </div>
    <div class="auth-right">
      <div class="auth-form-wrap" id="loginFormWrap">
        <h1>Sign In</h1>
        <p class="subtitle">Don't have an account? <a onclick="navigate('view-signup')">Create one free</a></p>

        <div class="tab-switcher">
          <button class="tab-btn active" id="loginTabEmail" onclick="switchLoginTab('email', this)"><i class="fa-solid fa-envelope"></i> Password</button>
          <button class="tab-btn" id="loginTabPhone" onclick="switchLoginTab('otp', this)"><i class="fa-solid fa-key"></i> Login via OTP</button>
        </div>

        <div class="divider"><span>secure login</span></div>
        <div id="login-alert-box" class="alert-box"></div>

        <form id="loginForm" novalidate>
          <div class="form-group" id="login-email-field">
            <label>Email Address</label>
            <div class="input-wrap">
              <i class="fa-solid fa-envelope input-icon"></i>
              <input type="email" id="login-email" placeholder="you@example.com" autocomplete="email"/>
            </div>
            <span class="error-msg" id="login-email-err">Please enter a valid email address.</span>
          </div>

          <div class="form-group" id="login-password-field">
            <label>Password</label>
            <div class="input-wrap">
              <i class="fa-solid fa-lock input-icon"></i>
              <input type="password" id="login-password" placeholder="Enter your password"/>
              <button type="button" class="toggle-pass" onclick="togglePass('login-password',this)"><i class="fa-solid fa-eye"></i></button>
            </div>
            <span class="error-msg" id="login-pass-err">Password must be at least 6 characters.</span>
          </div>
          
          <div class="form-group" id="login-otp-field" style="display:none">
            <label>Enter 6-Digit OTP</label>
            <div class="input-wrap">
              <i class="fa-solid fa-key input-icon"></i>
              <input type="text" id="login-otp" placeholder="123456"/>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <a href="#" onclick="resetLoginStep(); return false;" class="forgot-link">Use a different email?</a>
            </div>
          </div>
          
          <div class="form-footer" id="forgotPassWrapper">
            <label class="remember">
              <input type="checkbox" id="login-rememberMe"/>
              <span>Remember me</span>
            </label>
            <a href="#" onclick="showForgotPasswordForm(); return false;" class="forgot-link">Forgot password?</a>
          </div>

          <button type="submit" class="btn-auth" id="loginBtn">
            <i class="fa-solid fa-right-to-bracket"></i> <span id="loginBtnText">Sign In</span>
          </button>
        </form>
      </div>

      <!-- FORGOT PASSWORD FORM -->
      <div class="auth-form-wrap" id="forgotFormWrap" style="display:none;">
        <h1>Reset Password</h1>
        <p class="subtitle">Enter your email to receive a secure recovery code.</p>
        
        <div id="forgot-alert-box" class="alert-box"></div>

        <form id="forgotPasswordForm" novalidate>
          <div class="form-group" id="forgot-email-field">
            <label>Email Address</label>
            <div class="input-wrap">
              <i class="fa-solid fa-envelope input-icon"></i>
              <input type="email" id="forgot-email" placeholder="you@example.com"/>
            </div>
            <span class="error-msg" id="forgot-email-err">Please enter a valid email.</span>
          </div>
          <div class="form-group" id="forgot-otp-field" style="display:none;">
            <label>Enter 6-Digit OTP</label>
            <div class="input-wrap">
              <i class="fa-solid fa-key input-icon"></i>
              <input type="text" id="forgot-otp" placeholder="123456"/>
            </div>
            <label style="margin-top:16px;">New Password</label>
            <div class="input-wrap">
              <i class="fa-solid fa-lock input-icon"></i>
              <input type="password" id="forgot-password" placeholder="Min. 6 characters"/>
              <button type="button" class="toggle-pass" onclick="togglePass('forgot-password',this)"><i class="fa-solid fa-eye"></i></button>
            </div>
          </div>
          <button type="submit" class="btn-auth" id="forgotBtn" style="margin-top: 12px;">
            <i class="fa-solid fa-paper-plane"></i> <span id="forgotBtnText">Send Reset OTP</span>
          </button>
          <div style="text-align: center; margin-top: 16px;">
            <a href="#" onclick="showLoginForm(); return false;" class="forgot-link">Back to Login</a>
          </div>
        </form>
      </div>
    </div>
  </div>

  <!-- =========================================
       VIEW: SIGNUP
       ========================================= -->
  <div id="view-signup" class="app-view auth-view">
    <div class="auth-left">
      <div class="auth-left-orb orb1"></div>
      <div class="auth-left-orb orb2"></div>
      <div class="auth-brand" onclick="navigate('view-home')">
        <div class="brand-icon"><i class="fa-solid fa-seedling"></i></div>
        <span class="brand-name">PlantCare AI</span>
      </div>
      <div class="auth-hero">
        <div class="auth-hero-icon"><i class="fa-solid fa-microscope"></i></div>
        <h2>Join Thousands of Farmers Using AI to Save Their Crops</h2>
        <p>Create a free account and start diagnosing plant diseases instantly with cutting-edge AI technology.</p>
      </div>
      <div class="left-steps">
        <div class="left-step"><div class="left-step-num">1</div><div class="left-step-text"><strong>Create your free account</strong>Takes less than 2 minutes to get started.</div></div>
        <div class="left-step"><div class="left-step-num">2</div><div class="left-step-text"><strong>Upload a plant photo</strong>Take a clear picture of any infected plant leaf.</div></div>
        <div class="left-step"><div class="left-step-num">3</div><div class="left-step-text"><strong>Get instant diagnosis</strong>AI detects disease & suggests treatment in seconds.</div></div>
      </div>
    </div>

    <div class="auth-right">
      <div class="auth-form-wrap">
        <h1>Create Account</h1>
        <p class="subtitle">Already have an account? <a onclick="navigate('view-login')">Sign in here</a></p>

        

        <div class="divider"><span>or register with email</span></div>
        <div id="signup-alert-box" class="alert-box"></div>

        <form id="signupForm" novalidate>
          <div class="form-row">
            <div class="form-group">
              <label>First Name *</label>
              <div class="input-wrap">
                <i class="fa-solid fa-user input-icon"></i>
                <input type="text" id="signup-firstName" placeholder="Ravi"/>
              </div>
              <span class="error-msg" id="signup-fn-err">First name is required.</span>
            </div>
            <div class="form-group">
              <label>Last Name *</label>
              <div class="input-wrap">
                <i class="fa-solid fa-user input-icon"></i>
                <input type="text" id="signup-lastName" placeholder="Kumar"/>
              </div>
              <span class="error-msg" id="signup-ln-err">Last name is required.</span>
            </div>
          </div>
          <div class="form-group">
            <label>Email Address *</label>
            <div class="input-wrap">
              <i class="fa-solid fa-envelope input-icon"></i>
              <input type="email" id="signup-email" placeholder="ravi@example.com"/>
            </div>
            <span class="error-msg" id="signup-email-err">Please enter a valid email address.</span>
          </div>
          <div class="form-group">
            <label>Phone Number (optional)</label>
            <div class="input-wrap">
              <i class="fa-solid fa-phone input-icon"></i>
              <input type="tel" id="signup-phone" placeholder="+91 98765 43210"/>
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label>Password *</label>
              <div class="input-wrap">
                <i class="fa-solid fa-lock input-icon"></i>
                <input type="password" id="signup-password" placeholder="Min. 6 characters" oninput="checkStrength(this.value)"/>
                <button type="button" class="toggle-pass" onclick="togglePass('signup-password',this)"><i class="fa-solid fa-eye"></i></button>
              </div>
              <div class="password-strength">
                <div class="strength-bar"><div class="strength-fill" id="strengthBar"></div></div>
                <span class="strength-label" id="strengthLabel">Password strength</span>
              </div>
              <span class="error-msg" id="signup-pass-err">Password must be at least 6 characters.</span>
            </div>
            <div class="form-group">
              <label>Confirm Password *</label>
              <div class="input-wrap">
                <i class="fa-solid fa-lock input-icon"></i>
                <input type="password" id="signup-confirmPass" placeholder="Repeat password"/>
                <button type="button" class="toggle-pass" onclick="togglePass('signup-confirmPass',this)"><i class="fa-solid fa-eye"></i></button>
              </div>
              <span class="error-msg" id="signup-cpass-err">Passwords do not match.</span>
            </div>
          </div>
          <div class="terms">
            <input type="checkbox" id="signup-terms"/>
            <span>I agree to the <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>.</span>
          </div>
          <span class="error-msg" id="signup-terms-err" style="display:none;margin-bottom:12px">You must agree to the terms to continue.</span>
          <button type="submit" class="btn-auth" id="signupBtn">
            <i class="fa-solid fa-user-plus"></i> Create Account
          </button>
        </form>
      </div>
    </div>
  </div>

  <!-- =========================================
       VIEW: DASHBOARD
       ========================================= -->
  <div id="view-dashboard" class="app-view dashboard-view">
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-brand" onclick="navigate('view-home')">
        <div>
          <div class="brand-icon"><i class="fa-solid fa-seedling"></i></div>
          <span class="brand-name">PlantCare AI</span>
        </div>
      </div>
      <nav class="sidebar-nav">
        <div class="nav-section-label" data-i18n="nav_main">Main</div>
        <button class="sidebar-nav-item active" onclick="showDashboardPage('overview', this)"><i class="fa-solid fa-house"></i> <span data-i18n="nav_overview">Overview</span></button>
        <button class="sidebar-nav-item" onclick="showDashboardPage('diagnose', this)"><i class="fa-solid fa-microscope"></i> <span data-i18n="nav_diagnose">Diagnose Plant</span> <span class="badge">New</span></button>
        <button class="sidebar-nav-item" onclick="showDashboardPage('history', this)"><i class="fa-solid fa-clock-rotate-left"></i> <span data-i18n="nav_history">History</span></button>
        <button class="sidebar-nav-item" onclick="showDashboardPage('assistant', this)"><i class="fa-solid fa-robot"></i> <div class="card-title" data-i18n="assistant_title" style="color: inherit; font-size: inherit; font-family: inherit; font-weight: 500; margin: 0;">Agri AI Assistant</div></button>
        
        <div class="nav-section-label" style="margin-top:12px" data-i18n="nav_account">Account</div>
        <button class="sidebar-nav-item" onclick="showDashboardPage('profile', this)"><i class="fa-solid fa-user-circle"></i> <span data-i18n="nav_profile">My Profile</span></button>
        <button class="sidebar-nav-item" onclick="showDashboardPage('disease-library', this)"><i class="fa-solid fa-book-open"></i> <span data-i18n="nav_library">Disease Library</span></button>
        <button class="sidebar-nav-item" onclick="navigate('view-home')"><i class="fa-solid fa-globe"></i> <span data-i18n="nav_home">Home Page</span></button>
      </nav>
      <div class="sidebar-user">
        <div class="sidebar-user-info">
          <div class="user-avatar" id="sidebarAvatar">U</div>
          <div>
            <div class="user-name" id="sidebarName">User</div>
            <div class="user-role">Free Account</div>
          </div>
          <button class="btn-logout" onclick="logout()" title="Logout"><i class="fa-solid fa-right-from-bracket"></i></button>
        </div>
      </div>
    </aside>

    <div class="main">
      <header class="topbar">
        <div class="topbar-left">
          <button class="sidebar-toggle" id="sidebarToggle"><i class="fa-solid fa-bars"></i></button>
          <span class="topbar-title" id="pageTitle" data-i18n="page_dashboard">Dashboard</span>
        </div>
        <div class="topbar-right">
          
          <!-- MULTILINGUAL DROPDOWN -->
          <div class="lang-select-wrap">
            <i class="fa-solid fa-earth-asia"></i>
            <select id="appLang" class="lang-select" onchange="changeLanguage()">
              <option value="en">English</option>
              <option value="hi">हिंदी (Hindi)</option>
              <option value="kn">ಕನ್ನಡ (Kannada)</option>
              <option value="mr">मराठी (Marathi)</option>
              <option value="te">తెలుగు (Telugu)</option>
              <option value="ta">தமிழ் (Tamil)</option>
            </select>
          </div>

          <button class="topbar-notif"><i class="fa-regular fa-bell"></i><span class="notif-dot"></span></button>

          <div class="topbar-user" onclick="toggleDropdown()">
            <div class="topbar-avatar" id="topbarAvatar">U</div>
            <div>
              <div class="topbar-uname" id="topbarName">User</div>
              <div class="topbar-uemail" id="topbarEmail">user@email.com</div>
            </div>
            <i class="fa-solid fa-chevron-down" style="font-size:11px;color:var(--text-light); margin-left:8px;"></i>
            <div class="user-dropdown" id="userDropdown">
              <div class="dropdown-header">
                <div class="name" id="dropName">User</div>
                <div class="email" id="dropEmail">user@email.com</div>
              </div>
              <button class="dropdown-item" onclick="showDashboardPage('profile', null)"><i class="fa-solid fa-user"></i> My Profile</button>
              <button class="dropdown-item" onclick="showDashboardPage('diagnose', null)"><i class="fa-solid fa-microscope"></i> New Diagnosis</button>
              <button class="dropdown-item" onclick="navigate('view-home')"><i class="fa-solid fa-globe"></i> Home Page</button>
              <button class="dropdown-item danger" onclick="logout()"><i class="fa-solid fa-right-from-bracket"></i> Sign Out</button>
            </div>
          </div>
        </div>
      </header>

      <div class="content">
        <!-- OVERVIEW PAGE -->
        <div class="dashboard-page active" id="page-overview">
          <div class="greeting-card">
            <div class="greeting-text">
              <h2><span data-i18n="greet_hello">Hello</span>, <span id="greetName">Farmer</span>! 🌿</h2>
              <p data-i18n="greet_desc">Ready to check your plants today? Upload an image to get an instant diagnosis.</p>
            </div>
            <div class="greeting-icon"><i class="fa-solid fa-leaf"></i></div>
          </div>
          
          <div class="stats-row">
            <div class="stat-card">
              <div class="stat-icon-wrap green"><i class="fa-solid fa-microscope"></i></div>
              <div><div class="stat-num" id="overviewScans">0</div><div class="stat-lbl" data-i18n="stat_scans">Total Scans</div></div>
            </div>
            <div class="stat-card">
              <div class="stat-icon-wrap gold"><i class="fa-solid fa-triangle-exclamation"></i></div>
              <div><div class="stat-num" id="overviewDiseases">0</div><div class="stat-lbl" data-i18n="stat_diseases">Diseases Found</div></div>
            </div>
            <div class="stat-card">
              <div class="stat-icon-wrap blue"><i class="fa-solid fa-seedling"></i></div>
              <div><div class="stat-num" id="overviewHealthy">0</div><div class="stat-lbl" data-i18n="stat_healthy">Healthy Plants</div></div>
            </div>
            <div class="stat-card">
              <div class="stat-icon-wrap red"><i class="fa-solid fa-clock"></i></div>
              <div><div class="stat-num" id="overviewPending">0</div><div class="stat-lbl">Pending Review</div></div>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:1.4fr 1fr;gap:24px">
            <div class="card">
              <div class="card-header">
                <div class="card-header-icon"><i class="fa-solid fa-bolt"></i></div>
                <div><div class="card-title" data-i18n="quick_diagnose">Quick Diagnose</div></div>
              </div>
              <p style="font-size:.87rem;color:var(--text-light);margin-bottom:16px" data-i18n="quick_diagnose_desc">Upload a photo of your plant leaf and get an AI-powered diagnosis in seconds.</p>
              <button class="btn-detect ready" style="margin-top:0" onclick="showDashboardPage('diagnose', null)"><i class="fa-solid fa-microscope"></i> <span data-i18n="start_diagnosing">Start Diagnosis</span></button>
            </div>
            <div class="card">
              <div class="card-header">
                <div class="card-header-icon" style="background:#fff8e7;color:var(--gold)"><i class="fa-solid fa-star"></i></div>
                <div><div class="card-title">Tip of the Day</div></div>
              </div>
              <p style="font-size:.87rem;color:var(--text-mid);line-height:1.65">Early morning is the best time to detect leaf diseases — the light is optimal and leaves show clearest symptoms before midday heat.</p>
            </div>
          </div>
        </div>

        <!-- AI ASSISTANT PAGE -->
        <div class="dashboard-page" id="page-assistant">
          <div class="card" style="height: calc(100vh - 140px); display: flex; flex-direction: column;">
            <div class="card-header" style="flex-shrink: 0;">
              <div class="card-header-icon" style="background:#e8f4fd;color:#2980b9"><i class="fa-solid fa-robot"></i></div>
              <div style="flex: 1;">
                <div class="card-title" data-i18n="assistant_title">Agri AI Assistant</div>
                <div class="card-subtitle" data-i18n="assistant_subtitle">Ask questions about farming, crops, and soil</div>
              </div>
              <button id="ttsToggleBtn" onclick="toggleTTS()" style="background:none; border:none; color:var(--green-mid); cursor:pointer; font-size:20px; padding: 5px;" title="Toggle AI Voice">
                <i class="fa-solid fa-volume-high"></i>
              </button>
            </div>
            
            <div class="chat-container" style="flex: 1; border: none; margin-top: 0; height: 100%;">
              <div class="chat-messages" id="chatMessages">
                <div class="chat-msg ai" data-i18n="chat_welcome">Hello! I am your agricultural assistant. How can I help you with your crops and farming today?</div>
              </div>
              <div class="chat-input-wrap">
                <input type="text" id="chatInput" class="chat-input" placeholder="Ask about fertilizers, weather, crops..." onkeypress="if(event.key === 'Enter') sendChatMessage()">
                <button class="chat-btn" id="chatVoiceBtn" onclick="toggleChatVoice()" style="background: var(--green-pale); color: var(--green-dark); border: 1px solid var(--green-accent);" title="Tap to speak">
                  <i class="fa-solid fa-microphone"></i>
                </button>
                <button class="chat-btn" onclick="sendChatMessage()"><i class="fa-solid fa-paper-plane"></i></button>
              </div>
            </div>
          </div>
        </div>

        <!-- DIAGNOSE PAGE -->
        <div class="dashboard-page" id="page-diagnose">
          <div class="diagnose-grid">
            
            <!-- UPLOAD PANEL -->
            <div class="card">
              <div class="card-header">
                <div class="card-header-icon"><i class="fa-solid fa-upload"></i></div>
                <div><div class="card-title" data-i18n="upload_title">Upload Plant Image</div><div class="card-subtitle">JPG, PNG — Max 10MB</div></div>
              </div>
              
              <div id="diagnose-alert-box" class="alert-box" style="display:none; margin-bottom: 16px;"></div>

              <div class="upload-zone" id="uploadZone" ondragover="event.preventDefault();this.classList.add('dragover')" ondragleave="this.classList.remove('dragover')" ondrop="handleDrop(event)">
                <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])" capture="environment"/>
                <div class="upload-icon"><i class="fa-solid fa-cloud-arrow-up"></i></div>
                <div class="upload-text" data-i18n="upload_drag">Click or drag & drop your image here</div>
                <div class="upload-hint" data-i18n="upload_hint">Take a clear, well-lit photo of the affected leaf</div>
              </div>

              <!-- LIVE CAMERA INTEGRATION -->
              <button type="button" id="openCameraBtn" class="btn-outline" onclick="openCamera()" style="width: 100%; margin-top: 12px; display: flex; justify-content: center; align-items: center; gap: 8px;">
                <i class="fa-solid fa-camera"></i> <span data-i18n="btn_live_camera">Open Live Camera</span>
              </button>

              <div id="cameraContainer" style="display:none; flex-direction:column; align-items:center; margin-top: 16px; border: 2px solid var(--green-pale); border-radius: 16px; overflow: hidden; background: #000;">
                <video id="cameraVideo" autoplay playsinline style="width: 100%; max-height: 300px; object-fit: cover;"></video>
                <div style="padding: 12px; width: 100%; display: flex; gap: 10px; justify-content: center; background: var(--cream);">
                  <button type="button" class="btn-fill" onclick="snapPicture()"><i class="fa-solid fa-camera-retro"></i> Capture</button>
                  <button type="button" class="btn-outline" onclick="closeCamera()"><i class="fa-solid fa-xmark"></i> Cancel</button>
                </div>
                <canvas id="cameraCanvas" style="display:none;"></canvas>
              </div>
              <!-- END LIVE CAMERA -->
              
              <div class="preview-wrap" id="previewWrap">
                <img id="previewImg" src="" alt="Preview"/>
                <button class="preview-remove" onclick="clearImage()"><i class="fa-solid fa-xmark"></i></button>
              </div>

              <!-- CROP NAME INPUT + VOICE -->
              <div style="margin-top:20px;">
                <label style="display:block;font-size:0.85rem;font-weight:700;color:var(--text-mid);margin-bottom:6px" data-i18n="crop_name_label">Crop Name *</label>
                <div style="display:flex;gap:10px;">
                  <div class="input-wrap" style="flex:1; position: relative;">
                    <i class="fa-solid fa-seedling input-icon"></i>
                    <input type="text" id="cropName" placeholder="e.g. Tomato, Rice, Wheat..." style="width:100%;padding:12px 16px 12px 40px;border:1.5px solid rgba(46,159,85,0.2);border-radius:11px;font-family:'DM Sans',sans-serif;font-size:0.95rem;outline:none;" />
                  </div>
                  <button type="button" class="voice-btn" id="voiceBtn" onclick="toggleVoice()" title="Tap to speak">
                    <i class="fa-solid fa-microphone"></i>
                  </button>
                </div>
              </div>

              <button class="btn-detect" id="detectBtn" onclick="runDiagnosis()">
                  <i class="fa-solid fa-microscope"></i> <span data-i18n="btn_detect">Detect Disease</span>
              </button>
            </div>

            <!-- RESULT PANEL -->
            <div class="card">
              <div class="card-header">
                <div class="card-header-icon" style="background:#fff8e7;color:var(--gold)"><i class="fa-solid fa-file-medical"></i></div>
                <div><div class="card-title" data-i18n="report_title">Diagnosis Report</div></div>
              </div>
              
              <div id="emptyState" style="text-align:center;padding:40px 20px;color:var(--text-light)">
                <i class="fa-solid fa-magnifying-glass" style="font-size:42px;opacity:.3;margin-bottom:12px"></i>
                <p style="font-size:.9rem" data-i18n="report_empty">Upload an image and click "Detect" to see results here.</p>
              </div>
              
              <div class="loading-wrap" id="loadingState">
                <div class="loader-rings"><i class="fa-solid fa-leaf"></i></div>
                <div class="loading-text" data-i18n="analyzing">Analyzing your plant...</div>
                <div class="loading-steps">
                  <div class="loading-step" id="ls1"><i class="fa-solid fa-check"></i> <span data-i18n="step_1">Processing image details</span></div>
                  <div class="loading-step" id="ls2"><i class="fa-solid fa-check"></i> <span data-i18n="step_2">Running visual analysis</span></div>
                  <div class="loading-step" id="ls3"><i class="fa-solid fa-check"></i> <span data-i18n="step_3">Identifying pathogen signatures</span></div>
                  <div class="loading-step" id="ls4"><i class="fa-solid fa-check"></i> <span data-i18n="step_4">Generating final report</span></div>
                </div>
              </div>
              
              <div class="result-wrap" id="resultState">
                <div class="result-header" style="justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(46,159,85,0.1); padding-bottom: 10px;">
                  <span class="result-badge" id="resultBadge"></span>
                  <!-- ADDED: Speak Result Button -->
                  <button type="button" id="speakResultBtn" class="btn-outline" style="padding: 6px 14px; font-size: 0.8rem; display:flex; align-items:center; gap:6px;" onclick="speakDiagnosis()">
                    <i class="fa-solid fa-volume-high"></i> Read Aloud
                  </button>
                </div>
                
                <div class="disease-name" id="res-disease-name" style="font-size:1.4rem; color:var(--green-dark); text-transform: capitalize; margin-top: 12px;">—</div>
                
                <!-- ADDED: Detected Plant Name UI -->
                <div class="detected-plant-name" id="res-detected-plant"></div>
                
                <!-- Confidence Bar -->
                <div class="confidence-bar" style="margin-top:10px; margin-bottom:20px;">
                  <div class="conf-track"><div class="conf-fill" id="confFill" style="width:0%"></div></div>
                  <div class="conf-label" id="confLabel">0% Confidence</div>
                </div>
                
                <div class="solution-block" id="solutionBlock" style="margin-top:20px;">
                  <h4><i class="fa-solid fa-circle-question"></i> <span data-i18n="lbl_causes">Why it happens</span></h4>
                  <p id="res-causes" style="margin-bottom:16px; white-space: pre-wrap;"></p>
                  
                  <h4><i class="fa-solid fa-notes-medical"></i> <span data-i18n="lbl_treatment">How to control it</span></h4>
                  <p id="res-treatment" style="white-space: pre-wrap;"></p>
                </div>
                
                <div class="treatment-tags" id="treatmentTags"></div>

                <button class="btn-new-scan" onclick="clearImage()" style="margin-top:20px; width:100%; justify-content:center;">
                  <i class="fa-solid fa-rotate-left"></i> <span data-i18n="btn_new_scan">New Scan</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- HISTORY PAGE -->
        <div class="dashboard-page" id="page-history">
          <div class="card" style="margin-bottom:24px">
            <div class="card-header">
              <div class="card-header-icon"><i class="fa-solid fa-clock-rotate-left"></i></div>
              <div><div class="card-title" data-i18n="nav_history">History</div></div>
            </div>
            <div class="history-list" id="historyListContainer"></div>
          </div>
        </div>

        <!-- PROFILE PAGE -->
        <div class="dashboard-page" id="page-profile">
          <div class="profile-grid">
            <div class="profile-card">
              <div class="profile-avatar-lg" id="profileAvatar">U</div>
              <div class="profile-name" id="profileName">User Name</div>
              <div class="profile-email" id="profileEmail">user@email.com</div>
              <div class="profile-member"><i class="fa-solid fa-star"></i> Free Member</div>
              <div class="profile-stats">
                <div class="profile-stat"><div class="profile-stat-n" id="profileScans">0</div><div class="profile-stat-l">Scans</div></div>
                <div class="profile-stat"><div class="profile-stat-n" id="profileDiseases">0</div><div class="profile-stat-l">Diseases</div></div>
                <div class="profile-stat"><div class="profile-stat-n" id="profileHealthy">0</div><div class="profile-stat-l">Healthy</div></div>
              </div>
            </div>
            <div class="profile-info-card">
              <div class="card-header" style="margin-bottom:24px">
                <div class="card-header-icon"><i class="fa-solid fa-id-card"></i></div>
                <div><div class="card-title">Account Information</div></div>
              </div>
              <div class="info-group">
                <div class="info-label">Full Name</div>
                <div class="info-val" id="infoName">—</div>
              </div>
              <hr class="info-divider"/>
              <div class="info-group">
                <div class="info-label">Email Address</div>
                <div class="info-val" id="infoEmail">—</div>
              </div>
              <hr class="info-divider"/>
              <div class="info-group">
                <div class="info-label">Phone Number</div>
                <div class="info-val" id="infoPhone">Not provided</div>
              </div>
              <hr class="info-divider"/>
              <div class="info-group">
                <div class="info-label">Account Type</div>
                <div class="info-val">Free Plan</div>
              </div>
            </div>
          </div>
        </div>

        <!-- DISEASE LIBRARY PAGE -->
        <div class="dashboard-page" id="page-disease-library">
          <div class="card" style="margin-bottom:24px">
            <div class="card-header">
              <div class="card-header-icon"><i class="fa-solid fa-book-open"></i></div>
              <div><div class="card-title" data-i18n="nav_library">Disease Library</div><div class="card-subtitle">Learn about plant diseases</div></div>
            </div>
            <input type="text" id="diseaseSearch" placeholder="Search diseases..." style="width:100%;padding:10px;border:1px solid rgba(46,159,85,0.2);border-radius:8px;margin-bottom:20px" oninput="filterDiseases()">
            <div id="diseaseLibraryContainer" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px"></div>
          </div>
        </div>

      </div><!-- .content -->
    </div><!-- .main -->
  </div><!-- .dashboard-view -->

<!-- =========================================
     JAVASCRIPT LOGIC
     ========================================= -->
<script>
  // Simple Dictionary for UI Translation
  const i18n = {
    'en': {
      'nav_main': 'Main', 'nav_overview': 'Overview', 'nav_diagnose': 'Diagnose Plant', 'nav_history': 'History', 
      'nav_account': 'Account', 'nav_profile': 'My Profile', 'nav_home': 'Home Page', 'page_dashboard': 'Dashboard',
      'nav_library': 'Disease Library',
      'greet_hello': 'Hello', 'greet_desc': 'Ready to check your plants today? Upload an image to get an instant diagnosis.',
      'stat_scans': 'Total Scans', 'stat_diseases': 'Diseases Found', 'stat_healthy': 'Healthy Plants',
      'quick_diagnose': 'Quick Diagnose', 'quick_diagnose_desc': 'Upload a photo of your plant leaf and get a diagnosis in seconds.',
      'start_diagnosing': 'Start Diagnosis', 'upload_title': 'Upload Plant Image', 'upload_drag': 'Click or drag & drop your image here',
      'upload_hint': 'Take a clear, well-lit photo of the affected leaf', 'crop_name_label': 'Crop Name *',
      'btn_detect': 'Detect Disease', 'report_title': 'Diagnosis Report', 'report_empty': 'Upload an image and click "Detect" to see results here.',
      'analyzing': 'Analyzing your plant...', 'lbl_causes': 'Why it happens', 'lbl_treatment': 'How to control it', 'btn_new_scan': 'New Scan',
      'step_1': 'Processing image details', 'step_2': 'Running visual analysis', 'step_3': 'Identifying pathogen signatures', 'step_4': 'Generating final report',
      'assistant_title': 'Agri AI Assistant', 'assistant_subtitle': 'Ask questions about farming, crops, and soil', 'chat_welcome': 'Hello! I am your agricultural assistant. How can I help you with your crops and farming today?',
      'btn_live_camera': 'Open Live Camera'
    },
    'hi': {
      'nav_main': 'मुख्य', 'nav_overview': 'अवलोकन', 'nav_diagnose': 'पौधे का निदान', 'nav_history': 'इतिहास', 
      'nav_account': 'खाता', 'nav_profile': 'मेरी प्रोफ़ाइल', 'nav_home': 'होम पेज', 'page_dashboard': 'डैशबोर्ड',
      'nav_library': 'रोग पुस्तकालय',
      'greet_hello': 'नमस्ते', 'greet_desc': 'क्या आप आज अपने पौधों की जांच करने के लिए तैयार हैं? छवि अपलोड करें।',
      'stat_scans': 'कुल स्कैन', 'stat_diseases': 'बीमारियां मिलीं', 'stat_healthy': 'स्वस्थ पौधे',
      'quick_diagnose': 'त्वरित निदान', 'quick_diagnose_desc': 'पत्ती की तस्वीर अपलोड करें और सेकंडों में निदान प्राप्त करें।',
      'start_diagnosing': 'निदान शुरू करें', 'upload_title': 'पौधे की छवि अपलोड करें', 'upload_drag': 'छवि को यहाँ क्लिक करें या खींचें',
      'upload_hint': 'प्रभावित पत्ती की स्पष्ट फोटो लें', 'crop_name_label': 'फसल का नाम *',
      'btn_detect': 'बीमारी का पता लगाएं', 'report_title': 'निदान रिपोर्ट', 'report_empty': 'परिणाम देखने के लिए छवि अपलोड करें और "पता लगाएं" पर क्लिक करें।',
      'analyzing': 'आपके पौधे का विश्लेषण किया जा रहा है...', 'lbl_causes': 'यह क्यों होता है (कारण)', 'lbl_treatment': 'इसे कैसे नियंत्रित करें (इलाज)', 'btn_new_scan': 'नया स्कैन',
      'step_1': 'छवि विवरण संसाधित किया जा रहा है', 'step_2': 'दृश्य विश्लेषण चल रहा है', 'step_3': 'रोगजनक लक्षणों की पहचान', 'step_4': 'अंतिम रिपोर्ट तैयार की जा रही है',
      'assistant_title': 'कृषि एआई सहायक', 'assistant_subtitle': 'खेती और फसलों के बारे में पूछें', 'chat_welcome': 'नमस्ते! मैं आपका कृषि सहायक हूँ। मैं आपकी खेती में कैसे मदद कर सकता हूँ?',
      'btn_live_camera': 'लाइव कैमरा खोलें'
    },
    'kn': {
      'nav_main': 'ಮುಖ್ಯ', 'nav_overview': 'ಅವಲೋಕನ', 'nav_diagnose': 'ರೋಗ ಪತ್ತೆಹಚ್ಚಿ', 'nav_history': 'ಇತಿಹಾಸ', 
      'nav_account': 'ಖಾತೆ', 'nav_profile': 'ನನ್ನ ಪ್ರೊಫೈಲ್', 'nav_home': 'ಮುಖಪುಟ', 'page_dashboard': 'ಡ್ಯಾಶ್ಬೋರ್ಡ್',
      'nav_library': 'ರೋಗ ಗ್ರಂಥಾಲಯ',
      'greet_hello': 'ನಮಸ್ಕಾರ', 'greet_desc': 'ಇಂದು ನಿಮ್ಮ ಸಸ್ಯಗಳನ್ನು ಪರಿಶೀಲಿಸಲು ಸಿದ್ಧರಿದ್ದೀರಾ? ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.',
      'stat_scans': 'ಒಟ್ಟು ಸ್ಕ್ಯಾನ್‌ಗಳು', 'stat_diseases': 'ಪತ್ತೆಯಾದ ರೋಗಗಳು', 'stat_healthy': 'ಆರೋಗ್ಯಕರ ಸಸ್ಯಗಳು',
      'quick_diagnose': 'ತ್ವರಿತ ರೋಗನಿರ್ಣಯ', 'quick_diagnose_desc': 'ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಸೆಕೆಂಡುಗಳಲ್ಲಿ ರೋಗನಿರ್ಣಯ ಪಡೆಯಿರಿ.',
      'start_diagnosing': 'ರೋಗನಿರ್ಣಯ ಪ್ರಾರಂಭಿಸಿ', 'upload_title': 'ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ', 'upload_drag': 'ಚಿತ್ರವನ್ನು ಇಲ್ಲಿ ಕ್ಲಿಕ್ ಮಾಡಿ ಅಥವಾ ಎಳೆಯಿರಿ',
      'upload_hint': 'ಎಲೆಯ ಸ್ಪಷ್ಟವಾದ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ', 'crop_name_label': 'ಬೆಳೆಯ ಹೆಸರು *',
      'btn_detect': 'ರೋಗ ಪತ್ತೆಹಚ್ಚಿ', 'report_title': 'ವರದಿ', 'report_empty': 'ಫಲಿತಾಂಶಗಳನ್ನು ನೋಡಲು ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.',
      'analyzing': 'ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...', 'lbl_causes': 'ಕಾರಣಗಳು', 'lbl_treatment': 'ನಿಯಂತ್ರಣ / ಚಿಕಿತ್ಸೆ', 'btn_new_scan': 'ಹೊಸ ಸ್ಕ್ಯಾನ್',
      'step_1': 'ಚಿತ್ರ ಸಂಸ್ಕರಣೆ', 'step_2': 'ದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆ ನಡೆಯುತ್ತಿದೆ', 'step_3': 'ರೋಗಲಕ್ಷಣಗಳ ಗುರುತಿಸುವಿಕೆ', 'step_4': 'ಅಂತಿಮ ವರದಿ ರಚನೆ',
      'assistant_title': 'ಕೃಷಿ AI ಸಹಾಯಕ', 'assistant_subtitle': 'ಕೃಷಿ ಮತ್ತು ಬೆಳೆಗಳ ಬಗ್ಗೆ ಕೇಳಿ', 'chat_welcome': 'ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಕೃಷಿ ಸಹಾಯಕ. ಇಂದು ನಿಮ್ಮ ಕೃಷಿಗೆ ನಾನು ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ?',
      'btn_live_camera': 'ಲೈವ್ ಕ್ಯಾಮೆರಾ ತೆರೆಯಿರಿ'
    },
    'mr': {
      'nav_main': 'मुख्य', 'nav_overview': 'आढावा', 'nav_diagnose': 'वनस्पतीचे निदान', 'nav_history': 'इतिहास', 
      'nav_account': 'खाते', 'nav_profile': 'माझी प्रोफाईल', 'nav_home': 'होम पेज', 'page_dashboard': 'डॅशबोर्ड',
      'nav_library': 'रोग लायब्ररी',
      'greet_hello': 'नमस्कार', 'greet_desc': 'आज तुमची झाडे तपासण्यासाठी तयार आहात? प्रतिमा अपलोड करा.',
      'stat_scans': 'एकूण स्कॅन', 'stat_diseases': 'आढळलेले रोग', 'stat_healthy': 'निरोगी झाडे',
      'quick_diagnose': ' त्वरित निदान', 'quick_diagnose_desc': 'पानाचा फोटो अपलोड करा आणि सेकंदात निदान मिळवा.',
      'start_diagnosing': 'निदान सुरू करा', 'upload_title': 'प्रतिमा अपलोड करा', 'upload_drag': 'तुमची प्रतिमा येथे क्लिक करा किंवा ड्रॅग करा',
      'upload_hint': 'पानाचा स्पष्ट फोटो घ्या', 'crop_name_label': 'पिकाचे नाव *',
      'btn_detect': 'रोग शोधा', 'report_title': 'निदान अहवाल', 'report_empty': 'निकाल पाहण्यासाठी प्रतिमा अपलोड करा.',
      'analyzing': 'तुमच्या वनस्पतीचे विश्लेषण करत आहे...', 'lbl_causes': 'कारणे', 'lbl_treatment': 'उपचार / नियंत्रण', 'btn_new_scan': 'नवीन स्कॅन',
      'step_1': 'प्रतिमा प्रक्रिया', 'step_2': 'दृश्य विश्लेषण', 'step_3': 'लक्षणे ओळखत आहे', 'step_4': 'अंतिम अहवाल तयार करत आहे',
      'assistant_title': 'कृषी AI सहाय्यक', 'assistant_subtitle': 'शेती आणि पिकांबद्दल विचारा', 'chat_welcome': 'नमस्कार! मी तुमचा कृषी सहाय्यक आहे. आज मी तुम्हाला कशी मदत करू शकतो?',
      'btn_live_camera': 'थेट कॅमेरा उघडा'
    },
    'te': {
      'nav_main': 'ప్రధాన', 'nav_overview': 'అవలోకనం', 'nav_diagnose': 'వ్యాధి నిర్ధారణ', 'nav_history': 'చరిత్ర', 
      'nav_account': 'ఖాతా', 'nav_profile': 'నా ప్రొఫైల్', 'nav_home': 'హోమ్ పేజీ', 'page_dashboard': 'డాష్‌బోర్డ్',
      'nav_library': 'వ్యాధి లైబ్రరీ',
      'greet_hello': 'నమస్కారం', 'greet_desc': 'ఈరోజు మీ మొక్కలను తనిఖీ చేయడానికి సిద్ధంగా ఉన్నారా? చిత్రాన్ని అప్‌లోడ్ చేయండి.',
      'stat_scans': 'మొత్తం స్కాన్‌లు', 'stat_diseases': 'వ్యాధులు కనుగొనబడ్డాయి', 'stat_healthy': 'ఆరోగ్యకరమైన మొక్కలు',
      'quick_diagnose': 'త్వరిత నిర్ధారణ', 'quick_diagnose_desc': 'ఆకు ఫోటోను అప్‌లోడ్ చేయండి మరియు సెకన్లలో రోగ నిర్ధారణ పొందండి.',
      'start_diagnosing': 'నిర్ధారణ ప్రారంభించండి', 'upload_title': 'చిత్రాన్ని అప్‌లోడ్ చేయండి', 'upload_drag': 'మీ చిత్రాన్ని ఇక్కడ క్లిక్ చేయండి లేదా లాగండి',
      'upload_hint': 'ఆకు యొక్క స్పష్టమైన ఫోటో తీయండి', 'crop_name_label': 'పంట పేరు *',
      'btn_detect': 'వ్యాధిని గుర్తించండి', 'report_title': 'నివేదిక', 'report_empty': 'ఫలితాలను చూడటానికి చిత్రాన్ని అప్‌లోడ్ చేయండి.',
      'analyzing': 'విశ్లేషిస్తోంది...', 'lbl_causes': 'కారణాలు', 'lbl_treatment': 'చికిత్స / నియంత్రణ', 'btn_new_scan': 'కొత్త స్కాన్',
      'step_1': 'చిత్రం ప్రాసెసింగ్', 'step_2': 'దృశ్య విశ్లేషణ', 'step_3': 'వ్యాధి లక్షణాలను గుర్తించడం', 'step_4': 'తుది నివేదిక సృష్టిస్తోంది',
      'assistant_title': 'వ్యవసాయ AI సహాయకుడు', 'assistant_subtitle': 'వ్యవసాయం మరియు పంటల గురించి అడగండి', 'chat_welcome': 'నమస్కారం! నేను మీ వ్యవసాయ సహాయకుడిని. ఈరోజు నేను మీకు ఎలా సహాయపడగలను?',
      'btn_live_camera': 'లైవ్ కెమెరాను తెరవండి'
    },
    'ta': {
      'nav_main': 'முக்கிய', 'nav_overview': 'கண்ணோட்டம்', 'nav_diagnose': 'நோய் கண்டறிதல்', 'nav_history': 'வரலாறு', 
      'nav_account': 'கணக்கு', 'nav_profile': 'என் சுயவிவரம்', 'nav_home': 'முகப்பு', 'page_dashboard': 'டாஷ்போர்டு',
      'nav_library': 'நோய் நூலகம்',
      'greet_hello': 'வணக்கம்', 'greet_desc': 'இன்று உங்கள் செடிகளை சரிபார்க்க தயாரா? படத்தை பதிவேற்றவும்.',
      'stat_scans': 'மொத்த ஸ்கேன்கள்', 'stat_diseases': 'கண்டறியப்பட்ட நோய்கள்', 'stat_healthy': 'ஆரோக்கியமான செடிகள்',
      'quick_diagnose': 'விரைவான கண்டறிதல்', 'quick_diagnose_desc': 'இலையின் புகைப்படத்தை பதிவேற்றி நொடிகளில் முடிவை பெறவும்.',
      'start_diagnosing': 'பகுப்பாய்வை தொடங்கு', 'upload_title': 'படத்தை பதிவேற்றவும்', 'upload_drag': 'படத்தை இங்கே கிளிக் செய்யவும் அல்லது இழுக்கவும்',
      'upload_hint': 'இலையின் தெளிவான புகைப்படத்தை எடுக்கவும்', 'crop_name_label': 'பயிர் பெயர் *',
      'btn_detect': 'நோயைக் கண்டறி', 'report_title': 'அறிக்கை', 'report_empty': 'முடிவுகளைக் காண படத்தைப் பதிவேற்றவும்.',
      'analyzing': 'பகுப்பாய்வு செய்யப்படுகிறது...', 'lbl_causes': 'காரணங்கள்', 'lbl_treatment': 'கட்டுப்படுத்துவது எப்படி / சிகிச்சை', 'btn_new_scan': 'புதிய ஸ்கேன்',
      'step_1': 'படம் செயலாக்கம்', 'step_2': 'காட்சி பகுப்பாய்வு', 'step_3': 'நோய் அறிகுறிகளை கண்டறிதல்', 'step_4': 'இறுதி அறிக்கை உருவாக்குதல்',
      'assistant_title': 'விவசாய AI உதவியாளர்', 'assistant_subtitle': 'விவசாயம் மற்றும் பயிர்கள் பற்றி கேளுங்கள்', 'chat_welcome': 'வணக்கம்! நான் உங்கள் விவசாய உதவியாளர். இன்று உங்கள் விவசாயத்திற்கு நான் எவ்வாறு உதவ முடியும்?',
      'btn_live_camera': 'நேரடி கேமராவை திற'
    }
  };

  // Change UI Language dynamically
  function changeLanguage() {
    const lang = document.getElementById('appLang').value;
    const dict = i18n[lang] || i18n['en'];
    
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      if (dict[key]) {
        if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
            el.placeholder = dict[key];
        } else {
            el.innerText = dict[key];
        }
      }
    });
  }

  // Voice Recognition Logic
  let recognition;
  function toggleVoice() {
    const btn = document.getElementById('voiceBtn');
    const input = document.getElementById('cropName');
    
    const langMap = { 'en': 'en-IN', 'hi': 'hi-IN', 'kn': 'kn-IN', 'mr': 'mr-IN', 'te': 'te-IN', 'ta': 'ta-IN' };
    const curLang = langMap[document.getElementById('appLang').value] || 'en-IN';

    if (btn.classList.contains('recording')) {
      if(recognition) recognition.stop();
      return;
    }

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert("Voice recognition is not supported in this browser. Please type the crop name.");
      return;
    }

    recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = curLang;
    recognition.interimResults = false;
    
    recognition.onstart = function() {
      btn.classList.add('recording');
      input.placeholder = "Listening...";
    };
    
    recognition.onresult = function(event) {
      input.value = event.results[0][0].transcript;
    };
    
    recognition.onerror = function(event) {
      console.error(event.error);
      input.placeholder = "Error. Try typing.";
    };
    
    recognition.onend = function() {
      btn.classList.remove('recording');
      if(input.value === "") {
        const lang = document.getElementById('appLang').value;
        input.placeholder = (lang === 'en') ? "e.g. Tomato, Rice, Wheat..." : "Type crop name...";
      }
    };
    
    recognition.start();
  }

  // CHAT ASSISTANT VOICE LOGIC
  let chatRecognition;
  function toggleChatVoice() {
    const btn = document.getElementById('chatVoiceBtn');
    const input = document.getElementById('chatInput');

    const langMap = { 'en': 'en-IN', 'hi': 'hi-IN', 'kn': 'kn-IN', 'mr': 'mr-IN', 'te': 'te-IN', 'ta': 'ta-IN' };
    const curLang = langMap[document.getElementById('appLang').value] || 'en-IN';

    if (btn.classList.contains('recording')) {
      if(chatRecognition) chatRecognition.stop();
      return;
    }

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert("Voice recognition is not supported in this browser. Please type your message.");
      return;
    }

    chatRecognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    chatRecognition.lang = curLang;
    chatRecognition.interimResults = false;

    chatRecognition.onstart = function() {
      btn.classList.add('recording');
      input.placeholder = "Listening...";
    };

    chatRecognition.onresult = function(event) {
      input.value = event.results[0][0].transcript;
    };

    chatRecognition.onerror = function(event) {
      console.error(event.error);
      input.placeholder = "Error. Try typing.";
    };

    chatRecognition.onend = function() {
      btn.classList.remove('recording');
      if(input.value === "") {
        input.placeholder = "Ask about fertilizers, weather, crops...";
      }
    };

    chatRecognition.start();
  }

  // Text To Speech (TTS) Logic
  let ttsEnabled = true;
  function toggleTTS() {
      ttsEnabled = !ttsEnabled;
      const btn = document.getElementById('ttsToggleBtn');
      if (ttsEnabled) {
          btn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
      } else {
          if ('speechSynthesis' in window) {
              window.speechSynthesis.cancel();
          }
          btn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i>';
      }
  }

  function speakText(text) {
      if (!ttsEnabled || !('speechSynthesis' in window)) return;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      const langMap = { 'en': 'en-IN', 'hi': 'hi-IN', 'kn': 'kn-IN', 'mr': 'mr-IN', 'te': 'te-IN', 'ta': 'ta-IN' };
      utterance.lang = langMap[document.getElementById('appLang').value] || 'en-IN';
      window.speechSynthesis.speak(utterance);
  }

  // Added Speak Diagnosis logic
  window.latestDiagnosisText = "";
  function speakDiagnosis() {
      if (window.latestDiagnosisText) {
          speakText(window.latestDiagnosisText);
      } else {
          alert("No diagnosis to read.");
      }
  }

  // Assistant Chat Functionality
  async function sendChatMessage() {
      const input = document.getElementById('chatInput');
      const msg = input.value.trim();
      if(!msg) return;

      const msgBox = document.getElementById('chatMessages');
      
      const userDiv = document.createElement('div');
      userDiv.className = 'chat-msg user';
      userDiv.textContent = msg;
      msgBox.appendChild(userDiv);
      
      input.value = '';
      msgBox.scrollTop = msgBox.scrollHeight;

      const aiDiv = document.createElement('div');
      aiDiv.className = 'chat-msg ai';
      aiDiv.innerHTML = '<i class="fa-solid fa-ellipsis fa-fade"></i>';
      msgBox.appendChild(aiDiv);
      msgBox.scrollTop = msgBox.scrollHeight;

      const langSelect = document.getElementById('appLang');
      const langName = langSelect ? langSelect.options[langSelect.selectedIndex].text : 'English';

      try {
          const response = await fetch('/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: msg, language: langName })
          });
          const data = await response.json();
          
          if(data.status === 'success' || data.status === 'demo') {
              let replyText = data.reply || data.message;
              
              // Remove markdown bolding and asterisks
              replyText = replyText.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\*/g, '');
              
              aiDiv.innerHTML = replyText.replace(/\n/g, '<br>');
              speakText(replyText); // trigger speech
          } else {
              aiDiv.textContent = "Error: " + data.message;
          }
      } catch (err) {
          aiDiv.textContent = "Error connecting to AI. Please check your network.";
      }
      msgBox.scrollTop = msgBox.scrollHeight;
  }

  // Navigation Logic
  function navigate(viewId) {
    if (viewId === 'view-dashboard') {
      const session = JSON.parse(localStorage.getItem('plantcare_session') || 'null');
      if (!session) { 
          viewId = 'view-login'; 
      } else { 
          initDashboard(); 
      }
    }
    document.querySelectorAll('.app-view').forEach(v => v.classList.remove('active'));
    document.getElementById(viewId).classList.add('active');
    window.scrollTo(0, 0);

    // Close mobile navs if open
    const links = document.querySelector('.nav-links');
    const cta = document.querySelector('.nav-cta');
    if (links) links.style.display = '';
    if (cta) cta.style.display = '';
    const sidebar = document.getElementById('sidebar');
    if (sidebar) sidebar.classList.remove('open');
  }

  document.addEventListener('DOMContentLoaded', () => {
    navigate('view-home');
    changeLanguage(); 
    
    const hamburger = document.getElementById('hamburger');
    if(hamburger) {
      hamburger.addEventListener('click', function() {
        const links = document.querySelector('.nav-links');
        const cta = document.querySelector('.nav-cta');
        if (links.style.display === 'flex') {
          links.style.display = ''; cta.style.display = '';
        } else {
          links.style.cssText = 'display:flex;flex-direction:column;position:fixed;top:72px;left:0;right:0;background:#fff;padding:20px 5%;gap:18px;box-shadow:0 8px 30px rgba(0,0,0,.1);z-index:99;';
          cta.style.cssText = 'display:flex;position:fixed;top:230px;left:0;right:0;background:#fff;padding:0 5% 20px;z-index:99;gap:12px;';
        }
      });
    }

    const sidebarToggle = document.getElementById('sidebarToggle');
    if(sidebarToggle) {
      sidebarToggle.addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
      });
    }

    // AUTH FORMS - SIGNUP (DB Connected)
    const signupForm = document.getElementById('signupForm');
    if(signupForm) {
      signupForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        hideErrors('view-signup');
        let valid = true;

        const fn = document.getElementById('signup-firstName').value.trim();
        const ln = document.getElementById('signup-lastName').value.trim();
        const em = document.getElementById('signup-email').value.trim();
        const pw = document.getElementById('signup-password').value;
        const cp = document.getElementById('signup-confirmPass').value;
        const tc = document.getElementById('signup-terms').checked;
        const phone = document.getElementById('signup-phone').value.replace(/\s/g,'');

        if (!fn) { document.getElementById('signup-fn-err').classList.add('show'); document.getElementById('signup-firstName').classList.add('error'); valid = false; }
        if (!ln) { document.getElementById('signup-ln-err').classList.add('show'); document.getElementById('signup-lastName').classList.add('error'); valid = false; }
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(em)) { document.getElementById('signup-email-err').classList.add('show'); document.getElementById('signup-email').classList.add('error'); valid = false; }
        if (pw.length < 6) { document.getElementById('signup-pass-err').classList.add('show'); document.getElementById('signup-password').classList.add('error'); valid = false; }
        if (pw !== cp) { document.getElementById('signup-cpass-err').classList.add('show'); document.getElementById('signup-confirmPass').classList.add('error'); valid = false; }
        if (!tc) { document.getElementById('signup-terms-err').style.display = 'block'; valid = false; } else { document.getElementById('signup-terms-err').style.display = 'none'; }

        if (!valid) return;

        const btn = document.getElementById('signupBtn');
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Creating Account...';
        btn.disabled = true;

        try {
            const res = await fetch("/api/signup", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ firstName: fn, lastName: ln, email: em, phone: phone, password: pw })
            });
            const data = await res.json();
            
            if (data.success) {
                localStorage.setItem('plantcare_session', JSON.stringify({ name: fn + ' ' + ln, email: em, phone: phone }));
                showAlert('Account created! Redirecting to dashboard...', 'success', 'signup-alert-box');
                setTimeout(() => {
                    btn.innerHTML = '<i class="fa-solid fa-user-plus"></i> Create Account';
                    btn.disabled = false;
                    signupForm.reset();
                    navigate('view-dashboard');
                }, 1400);
            } else {
                showAlert(data.message, 'error', 'signup-alert-box');
                btn.innerHTML = '<i class="fa-solid fa-user-plus"></i> Create Account';
                btn.disabled = false;
            }
        } catch(err) {
            showAlert("Server error during signup.", 'error', 'signup-alert-box');
            btn.innerHTML = '<i class="fa-solid fa-user-plus"></i> Create Account';
            btn.disabled = false;
        }
      });
    }

    // AUTH FORMS - LOGIN (DB Connected & OTP)
    let loginStep = 1;
    const loginForm = document.getElementById('loginForm');
    if(loginForm) {
      loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        hideErrors('view-login');
        let valid = true;

        const emailVal = document.getElementById('login-email').value.trim();
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(emailVal)) {
          document.getElementById('login-email-err').classList.add('show');
          document.getElementById('login-email').classList.add('error');
          valid = false;
        }

        const btn = document.getElementById('loginBtn');
        const btnText = document.getElementById('loginBtnText');

        if (loginActiveTab === 'email') {
            // PASSWORD LOGIN
            const passVal = document.getElementById('login-password').value;
            if (passVal.length < 6) {
                document.getElementById('login-pass-err').classList.add('show');
                document.getElementById('login-password').classList.add('error');
                valid = false;
            }
            if (!valid) return;

            btn.disabled = true;
            btnText.innerHTML = 'Signing In...';
            
            try {
                const res = await fetch("/api/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email: emailVal, password: passVal })
                });
                const data = await res.json();
                
                if (data.success) {
                    localStorage.setItem('plantcare_session', JSON.stringify(data.user));
                    showAlert('Login successful! Redirecting...', 'success', 'login-alert-box');
                    setTimeout(() => {
                        btnText.innerText = 'Sign In';
                        btn.disabled = false;
                        loginForm.reset();
                        navigate('view-dashboard');
                    }, 1200);
                } else {
                    showAlert(data.message, 'error', 'login-alert-box');
                    btnText.innerText = 'Sign In';
                    btn.disabled = false;
                }
            } catch(err) {
                showAlert("Network error.", 'error', 'login-alert-box');
                btnText.innerText = 'Sign In';
                btn.disabled = false;
            }

        } else {
            // OTP LOGIN FLOW
            if (!valid) return;
            
            if (loginStep === 1) {
                btn.disabled = true;
                btnText.innerHTML = 'Sending...';
                try {
                    const res = await fetch("/api/send-otp", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ email: emailVal })
                    });
                    const data = await res.json();
                    
                    if (data.success || data.message.includes("Dev code:")) {
                        if(data.message.includes("Dev code:")) {
                            showAlert("Check your terminal for the OTP (Dev Mode).", 'success', 'login-alert-box');
                        } else {
                            showAlert(data.message, 'success', 'login-alert-box');
                        }
                        document.getElementById('login-email-field').style.display = 'none';
                        document.getElementById('login-otp-field').style.display = 'block';
                        document.querySelector('.tab-switcher').style.display = 'none';
                        document.querySelector('.divider').style.display = 'none';
                        document.getElementById('forgotPassWrapper').style.display = 'none';
                        loginStep = 2;
                        btnText.innerText = "Verify OTP";
                    } else {
                        showAlert(data.message, 'error', 'login-alert-box');
                    }
                } catch(err) {
                    showAlert("Network error.", 'error', 'login-alert-box');
                } finally {
                    btn.disabled = false;
                }
            } else {
                const otpVal = document.getElementById('login-otp').value.trim();
                if (!otpVal) {
                    showAlert("Please enter the OTP.", 'error', 'login-alert-box');
                    return;
                }
                
                btn.disabled = true;
                btnText.innerHTML = 'Verifying...';
                
                try {
                    const res = await fetch("/api/verify-otp", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ email: emailVal, otp: otpVal })
                    });
                    const data = await res.json();
                    
                    if (data.success) {
                        showAlert('Login successful! Redirecting...', 'success', 'login-alert-box');
                        localStorage.setItem('plantcare_session', JSON.stringify(data.user));
                        setTimeout(() => {
                            btnText.innerText = 'Send OTP';
                            btn.disabled = false;
                            loginForm.reset();
                            resetLoginStep();
                            navigate('view-dashboard');
                        }, 1200);
                    } else {
                        showAlert(data.message, 'error', 'login-alert-box');
                        btnText.innerText = 'Verify OTP';
                        btn.disabled = false;
                    }
                } catch(err) {
                    showAlert("Network error.", 'error', 'login-alert-box');
                    btnText.innerText = 'Verify OTP';
                    btn.disabled = false;
                }
            }
        }
      });
    }

    // AUTH FORMS - FORGOT PASSWORD
    let forgotStep = 1;
    const forgotForm = document.getElementById('forgotPasswordForm');
    if (forgotForm) {
        forgotForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            hideErrors('view-login');
            let valid = true;
            
            const emailVal = document.getElementById('forgot-email').value.trim();
            if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(emailVal)) {
                document.getElementById('forgot-email-err').classList.add('show');
                document.getElementById('forgot-email').classList.add('error');
                valid = false;
            }
            if (!valid) return;

            const btn = document.getElementById('forgotBtn');
            const btnText = document.getElementById('forgotBtnText');
            
            if (forgotStep === 1) {
                btn.disabled = true;
                btnText.innerHTML = 'Sending...';
                try {
                    const res = await fetch("/api/send-otp", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ email: emailVal })
                    });
                    const data = await res.json();
                    
                    if (data.success || data.message.includes("Dev code:")) {
                        if(data.message.includes("Dev code:")) {
                            showAlert("Check your terminal for the OTP (Dev Mode).", 'success', 'forgot-alert-box');
                        } else {
                            showAlert(data.message, 'success', 'forgot-alert-box');
                        }
                        document.getElementById('forgot-email-field').style.display = 'none';
                        document.getElementById('forgot-otp-field').style.display = 'block';
                        forgotStep = 2;
                        btnText.innerText = "Reset Password";
                    } else {
                        showAlert(data.message, 'error', 'forgot-alert-box');
                    }
                } catch(err) {
                    showAlert("Network error.", 'error', 'forgot-alert-box');
                } finally {
                    btn.disabled = false;
                }
            } else {
                const otpVal = document.getElementById('forgot-otp').value.trim();
                const newPassVal = document.getElementById('forgot-password').value;
                if (!otpVal || newPassVal.length < 6) {
                    showAlert("Please enter a valid OTP and a password with at least 6 characters.", 'error', 'forgot-alert-box');
                    return;
                }

                btn.disabled = true;
                btnText.innerHTML = 'Resetting...';

                try {
                    const res = await fetch("/api/reset-password", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ email: emailVal, otp: otpVal, new_password: newPassVal })
                    });
                    const data = await res.json();
                    
                    if (data.success) {
                        showAlert('Password Reset Successfully! Redirecting to login...', 'success', 'forgot-alert-box');
                        setTimeout(() => {
                            forgotForm.reset();
                            showLoginForm();
                            btnText.innerText = 'Send Reset OTP';
                            btn.disabled = false;
                        }, 2000);
                    } else {
                        showAlert(data.message, 'error', 'forgot-alert-box');
                        btnText.innerText = 'Reset Password';
                        btn.disabled = false;
                    }
                } catch(err) {
                    showAlert("Network error.", 'error', 'forgot-alert-box');
                    btnText.innerText = 'Reset Password';
                    btn.disabled = false;
                }
            }
        });
    }

    loadDiseases();
  });

  // Auth/Form Helpers
  function showForgotPasswordForm() {
      document.getElementById('loginFormWrap').style.display = 'none';
      document.getElementById('forgotFormWrap').style.display = 'block';
      hideErrors('view-login');
  }

  function showLoginForm() {
      document.getElementById('forgotFormWrap').style.display = 'none';
      document.getElementById('loginFormWrap').style.display = 'block';
      document.getElementById('forgot-email-field').style.display = 'block';
      document.getElementById('forgot-otp-field').style.display = 'none';
      document.getElementById('forgotBtnText').innerText = 'Send Reset OTP';
      forgotStep = 1;
      hideErrors('view-login');
  }

  function resetLoginStep() {
      loginStep = 1;
      document.getElementById('login-email-field').style.display = 'block';
      document.getElementById('login-otp-field').style.display = 'none';
      document.getElementById('loginBtnText').innerText = loginActiveTab === 'email' ? 'Sign In' : 'Send OTP';
      document.getElementById('login-alert-box').style.display = 'none';
      document.querySelector('.tab-switcher').style.display = 'flex';
      document.querySelector('.divider').style.display = 'flex';
      document.getElementById('forgotPassWrapper').style.display = 'flex';
  }

  let loginActiveTab = 'email';
  function switchLoginTab(tab, btn) {
    loginActiveTab = tab;
    document.getElementById('loginTabEmail').classList.remove('active');
    document.getElementById('loginTabPhone').classList.remove('active');
    btn.classList.add('active');
    
    document.getElementById('login-email-field').style.display = 'block';
    document.getElementById('login-password-field').style.display = tab === 'email' ? 'block' : 'none';
    document.getElementById('login-otp-field').style.display = 'none';
    document.getElementById('loginBtnText').innerText = tab === 'email' ? 'Sign In' : 'Send OTP';
    document.getElementById('forgotPassWrapper').style.display = tab === 'email' ? 'flex' : 'none';
    loginStep = 1;
  }

  function togglePass(id, btn) {
    const inp = document.getElementById(id);
    const icon = btn.querySelector('i');
    if (inp.type === 'password') { inp.type = 'text'; icon.className = 'fa-solid fa-eye-slash'; }
    else { inp.type = 'password'; icon.className = 'fa-solid fa-eye'; }
  }

  function checkStrength(val) {
    const bar = document.getElementById('strengthBar');
    const lbl = document.getElementById('strengthLabel');
    let score = 0;
    if (val.length >= 6) score++;
    if (val.length >= 10) score++;
    if (/[A-Z]/.test(val)) score++;
    if (/[0-9]/.test(val)) score++;
    if (/[^a-zA-Z0-9]/.test(val)) score++;
    const levels = [{w:'20%',c:'#d62839',t:'Weak'},{w:'40%',c:'#ff8c00',t:'Fair'},{w:'60%',c:'#f4c430',t:'Good'},{w:'80%',c:'#74c69d',t:'Strong'},{w:'100%',c:'#2d6a4f',t:'Very Strong'}];
    const l = levels[Math.min(score-1,4)] || {w:'0%',c:'#e0e0e0',t:'Password strength'};
    bar.style.width = l.w; bar.style.background = l.c; lbl.textContent = l.t;
  }

  function showAlert(msg, type, boxId) {
    const a = document.getElementById(boxId);
    a.className = 'alert-box ' + type;
    a.innerHTML = `<i class="fa-solid fa-${type==='error'?'circle-exclamation':'circle-check'}"></i> <div>${msg}</div>`;
    a.style.display = 'flex';
  }

  function hideErrors(containerId) {
    const container = document.getElementById(containerId);
    container.querySelectorAll('.error-msg').forEach(e => e.classList.remove('show'));
    container.querySelectorAll('input').forEach(i => i.classList.remove('error'));
    const alertBox = container.querySelector('.alert-box');
    if(alertBox) alertBox.style.display = 'none';
  }

  function logout() {
    localStorage.removeItem('plantcare_session');
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    if(loginForm) loginForm.reset();
    if(signupForm) signupForm.reset();
    clearImage();
    navigate('view-login');
  }

  // Dashboard Initialization
  function initDashboard() {
    const session = JSON.parse(localStorage.getItem('plantcare_session'));
    if (!session) return;
    
    const name = session.name || 'User';
    const initials = name.split(' ').map(n=>n[0]).join('').substring(0,2).toUpperCase();
    const email = session.email || 'user@email.com';
    const phone = session.phone || '';

    document.getElementById('greetName').textContent = name.split(' ')[0];
    document.getElementById('sidebarName').textContent = name;
    document.getElementById('sidebarAvatar').textContent = initials;
    document.getElementById('topbarAvatar').textContent = initials;
    document.getElementById('topbarName').textContent = name;
    document.getElementById('topbarEmail').textContent = email;
    document.getElementById('dropName').textContent = name;
    document.getElementById('dropEmail').textContent = email;
    
    // Profile tab binds
    document.getElementById('profileAvatar').textContent = initials;
    document.getElementById('profileName').textContent = name;
    document.getElementById('profileEmail').textContent = email;
    document.getElementById('infoName').textContent = name;
    document.getElementById('infoEmail').textContent = email;
    if (phone) document.getElementById('infoPhone').textContent = phone;
    
    updateDashboardStats();
    showDashboardPage('overview', document.querySelector('.sidebar-nav-item.active'));
  }

  async function updateDashboardStats() {
    const session = JSON.parse(localStorage.getItem('plantcare_session'));
    if (!session) return;
    try {
        const res = await fetch('/api/stats/' + encodeURIComponent(session.email));
        const data = await res.json();
        if(data.success) {
            const stats = data.stats;
            const osElem = document.getElementById('overviewScans'); if (osElem) osElem.textContent = stats.total;
            const odElem = document.getElementById('overviewDiseases'); if (odElem) odElem.textContent = stats.diseased;
            const ohElem = document.getElementById('overviewHealthy'); if (ohElem) ohElem.textContent = stats.healthy;
            
            const psElem = document.getElementById('profileScans'); if (psElem) psElem.textContent = stats.total;
            const pdElem = document.getElementById('profileDiseases'); if (pdElem) pdElem.textContent = stats.diseased;
            const phElem = document.getElementById('profileHealthy'); if (phElem) phElem.textContent = stats.healthy;

            const opElem = document.getElementById('overviewPending'); if (opElem) opElem.textContent = Math.max(0, 2 - stats.total);
        }
    } catch(e) { console.error("Error fetching stats:", e); }
  }

  function showDashboardPage(id, btn) {
    document.querySelectorAll('.dashboard-page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sidebar-nav-item').forEach(n => n.classList.remove('active'));
    
    const page = document.getElementById('page-' + id);
    if (page) page.classList.add('active');
    
    if (btn) btn.classList.add('active');
    if (id === 'history') loadHistory();
    
    // Close dropdown on mobile safely
    const drop = document.getElementById('userDropdown');
    if (drop) drop.classList.remove('open');
    const sidebar = document.getElementById('sidebar');
    if (sidebar && window.innerWidth <= 768) sidebar.classList.remove('open');
    
    window.scrollTo(0, 0);
  }
  
  function toggleDropdown() {
    const d = document.getElementById('userDropdown');
    if (d) d.classList.toggle('open');
  }

  document.addEventListener('click', e => {
    if (!e.target.closest('.topbar-user')) {
        const d = document.getElementById('userDropdown');
        if(d) d.classList.remove('open');
    }
  });

  async function loadHistory() {
    const list = document.getElementById('historyListContainer');
    if (!list) return;
    
    const session = JSON.parse(localStorage.getItem('plantcare_session'));
    if (!session) return;

    list.innerHTML = '<p style="text-align:center;padding:20px;color:var(--text-light)">Loading...</p>';
    
    try {
        const res = await fetch('/api/history/' + encodeURIComponent(session.email));
        const data = await res.json();
        
        list.innerHTML = '';
        if (data.success && data.history.length > 0) {
            data.history.forEach(item => {
              const div = document.createElement('div');
              div.className = 'history-item';
              const isH = item.status === 'healthy';
              div.innerHTML = `
                <div class="history-thumb"><i class="fa-solid ${isH ? 'fa-check' : 'fa-leaf'}"></i></div>
                <div class="history-info">
                  <div class="history-name" style="text-transform: capitalize;">${item.disease_name}</div>
                  <div class="history-date">${item.scan_date} | Crop: ${item.crop_name || 'N/A'}</div>
                </div>
                <span class="history-status ${isH ? 'status-healthy' : 'status-diseased'}">${isH ? 'Healthy' : 'Diseased'}</span>
              `;
              list.appendChild(div);
            });
        } else {
            list.innerHTML = '<p style="text-align:center;padding:20px;color:var(--text-light)">No history yet.</p>';
        }
    } catch(e) {
        list.innerHTML = '<p style="text-align:center;padding:20px;color:red">Failed to load history.</p>';
    }
  }

  // Disease Library
  const diseases = [
    { name:'Tomato Leaf Blight', crop:'tomato', healthy:false, solution:'Apply copper-based fungicide every 7 days. Remove and destroy infected leaves. Improve air circulation around plants. Avoid overhead watering.', tags:['Fungicide','Copper Spray','Pruning','Drip Irrigation'] },
    { name:'Bacterial Spot', crop:'tomato', healthy:false, solution:'Apply copper hydroxide spray. Remove infected plant material. Avoid working with plants when wet. Treat seeds with hot water before planting.', tags:['Copper Hydroxide','Seed Treatment','Sanitation','Crop Rotation'] },
    { name:'Powdery Mildew', crop:'tomato', healthy:false, solution:'Spray neem oil or potassium bicarbonate solution weekly. Prune affected areas. Increase airflow by spacing plants properly. Avoid high humidity environments.', tags:['Neem Oil','Potassium Bicarbonate','Sulfur Spray','Pruning'] },
    { name:'Powdery Mildew', crop:'grapes', healthy:false, solution:'Apply sulfur-based fungicide. Ensure good air circulation in vineyards. Prune densely packed areas. Avoid overhead irrigation.', tags:['Sulfur Spray','Pruning','Air Circulation','Fungicide'] },
    { name:'Black Rot', crop:'grapes', healthy:false, solution:'Apply Mancozeb or Myclobutanil. Remove infected mummies and leaves. Prune to open canopy.', tags:['Mancozeb', 'Sanitation', 'Pruning'] },
    { name:'Peanut Rust', crop:'peanut', healthy:false, solution:'Apply fungicides like Azoxystrobin. Rotate crops to break disease cycle. Plant resistant varieties. Ensure proper spacing for air flow.', tags:['Azoxystrobin','Crop Rotation','Resistant Varieties','Spacing'] },
    { name:'Early Leaf Spot', crop:'peanut', healthy:false, solution:'Apply Chlorothalonil fungicide. Practice crop rotation and bury crop residue.', tags:['Chlorothalonil','Crop Rotation','Sanitation'] },
    { name:'Corn Grey Leaf Spot', crop:'corn', healthy:false, solution:'Apply Azoxystrobin or Propiconazole fungicide. Rotate crops yearly to prevent build-up. Use resistant corn varieties and ensure proper field drainage.', tags:['Azoxystrobin','Propiconazole','Crop Rotation','Drainage'] },
    { name:'Corn Smut', crop:'corn', healthy:false, solution:'Remove galls before they burst. Avoid mechanical injury to plants. Rotate crops.', tags:['Sanitation','Careful Handling','Crop Rotation'] },
    { name:'Potato Late Blight', crop:'potato', healthy:false, solution:'Apply fungicides like Mancozeb. Remove infected plants immediately. Plant resistant varieties. Avoid planting in wet areas.', tags:['Mancozeb','Sanitation','Resistant Varieties','Drainage'] },
    { name:'Potato Early Blight', crop:'potato', healthy:false, solution:'Apply Chlorothalonil. Maintain adequate nitrogen levels. Practice crop rotation.', tags:['Chlorothalonil','Fertilization','Crop Rotation'] }
  ];

  function loadDiseases() {
    const container = document.getElementById('diseaseLibraryContainer');
    if(!container || container.innerHTML !== '') return;

    diseases.forEach(d => {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="card-header">
          <div class="card-header-icon"><i class="fa-solid fa-leaf"></i></div>
          <div><div class="card-title">${d.name}</div><div class="card-subtitle">${d.crop.charAt(0).toUpperCase() + d.crop.slice(1)}</div></div>
        </div>
        <p style="font-size:.87rem;color:var(--text-mid);line-height:1.6">${d.solution}</p>
        <div class="treatment-tags" style="margin-top:16px;">${d.tags.map(t => `<span class="tag">${t}</span>`).join('')}</div>
      `;
      container.appendChild(card);
    });
  }

  function filterDiseases() {
    const query = document.getElementById('diseaseSearch').value.toLowerCase();
    const cards = document.querySelectorAll('#diseaseLibraryContainer .card');
    cards.forEach(card => {
      const title = card.querySelector('.card-title').textContent.toLowerCase();
      const subtitle = card.querySelector('.card-subtitle').textContent.toLowerCase();
      const visible = title.includes(query) || subtitle.includes(query);
      card.style.display = visible ? 'block' : 'none';
    });
  }

  // ===============================================
  // LIVE CAMERA LOGIC (WebRTC API)
  // ===============================================
  let cameraStream = null;

  async function openCamera() {
      const uploadZone = document.getElementById('uploadZone');
      const openCamBtn = document.getElementById('openCameraBtn');
      const camContainer = document.getElementById('cameraContainer');
      const video = document.getElementById('cameraVideo');

      try {
          cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
          video.srcObject = cameraStream;
          uploadZone.style.display = 'none';
          openCamBtn.style.display = 'none';
          camContainer.style.display = 'flex';
      } catch (err) {
          console.error("Camera access denied", err);
          alert("Unable to access the camera. Please ensure permissions are granted.");
      }
  }

  function closeCamera() {
      if (cameraStream) {
          cameraStream.getTracks().forEach(track => track.stop());
          cameraStream = null;
      }
      document.getElementById('cameraContainer').style.display = 'none';
      if (!currentUploadedFile) {
          document.getElementById('uploadZone').style.display = 'block';
          document.getElementById('openCameraBtn').style.display = 'flex';
      }
  }

  function snapPicture() {
      const video = document.getElementById('cameraVideo');
      const canvas = document.getElementById('cameraCanvas');
      if (!cameraStream) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(blob => {
          if(blob) {
              const file = new File([blob], "live_capture.jpg", { type: "image/jpeg" });
              closeCamera();
              handleFile(file);
          }
      }, 'image/jpeg', 0.95);
  }

  // ===============================================
  // Image Upload Logic
  // ===============================================
  let currentUploadedFile = null;

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    currentUploadedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      const img = document.getElementById('previewImg');
      if (img) img.src = e.target.result;
      
      const wrap = document.getElementById('previewWrap');
      if (wrap) wrap.style.display = 'block';
      
      const zone = document.getElementById('uploadZone');
      if (zone) zone.style.display = 'none';

      const openCamBtn = document.getElementById('openCameraBtn');
      if(openCamBtn) openCamBtn.style.display = 'none';
      
      checkReady();
    };
    reader.readAsDataURL(file);
  }

  function handleDrop(e) {
    e.preventDefault();
    const zone = document.getElementById('uploadZone');
    if (zone) zone.classList.remove('dragover');
    if (e.dataTransfer && e.dataTransfer.files) {
        handleFile(e.dataTransfer.files[0]);
    }
  }

  function clearImage() {
    currentUploadedFile = null;
    
    const previewWrap = document.getElementById('previewWrap');
    if (previewWrap) previewWrap.style.display = 'none';
    
    const uploadZone = document.getElementById('uploadZone');
    if (uploadZone) uploadZone.style.display = 'block';

    const openCamBtn = document.getElementById('openCameraBtn');
    if(openCamBtn) openCamBtn.style.display = 'flex';
    
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';
    
    const alertBox = document.getElementById('diagnose-alert-box');
    if (alertBox) alertBox.style.display = 'none';
    
    const emptyState = document.getElementById('emptyState');
    if (emptyState) emptyState.style.display = 'block';
    
    const resultState = document.getElementById('resultState');
    if (resultState) resultState.classList.remove('show');
    
    const loadingState = document.getElementById('loadingState');
    if (loadingState) loadingState.classList.remove('show');
    
    const confFill = document.getElementById('confFill');
    if (confFill) confFill.style.width = '0%';
    
    const treatmentTags = document.getElementById('treatmentTags');
    if (treatmentTags) treatmentTags.innerHTML = '';
    
    checkReady();
  }

  function checkReady() {
    const hasImage = currentUploadedFile !== null;
    const btn = document.getElementById('detectBtn');
    if (btn) {
        if (hasImage) btn.classList.add('ready');
        else btn.classList.remove('ready');
    }
  }

  // Main Backend API Call Logic
  async function runDiagnosis() {
    if (!currentUploadedFile) {
        alert("Please upload or capture a plant image first.");
        return;
    }
    
    const cropNameEl = document.getElementById('cropName');
    const cropNameVal = cropNameEl ? cropNameEl.value.trim() : '';
    if (!cropNameVal) {
        alert("Please enter or speak the Crop Name before detecting disease.");
        return;
    }

    const session = JSON.parse(localStorage.getItem('plantcare_session'));
    if (!session) return;

    // Reset and initialize UI safely
    const emptyState = document.getElementById('emptyState');
    if (emptyState) emptyState.style.display = 'none';
    
    const resultState = document.getElementById('resultState');
    if (resultState) resultState.classList.remove('show');
    
    const loadingState = document.getElementById('loadingState');
    if (loadingState) loadingState.classList.add('show');
    
    const alertBox = document.getElementById('diagnose-alert-box');
    if (alertBox) alertBox.style.display = 'none';
    
    const detectBtn = document.getElementById('detectBtn');
    if (detectBtn) {
        detectBtn.disabled = true;
        const lang = document.getElementById('appLang') ? document.getElementById('appLang').value : 'en';
        const txt = i18n[lang]?.analyzing || 'Analyzing...';
        detectBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${txt}`;
    }

    // Reset and show loading steps
    [1,2,3,4].forEach(i => {
        const ls = document.getElementById('ls'+i);
        if (ls) ls.classList.remove('show');
    });
    
    const steps = ['ls1','ls2','ls3','ls4'];
    steps.forEach((s, i) => setTimeout(() => {
        const el = document.getElementById(s);
        if (el) el.classList.add('show');
    }, (i+1)*600));

    const formData = new FormData(); 
    formData.append("file", currentUploadedFile);
    formData.append("crop_name", cropNameVal);
    
    const sel = document.getElementById('appLang');
    const langName = sel && sel.options ? sel.options[sel.selectedIndex].text : "English";
    const langKey = sel ? sel.value : "en";
    formData.append("language", langName);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        
        if(data.status === 'error') throw new Error(data.message);

        await new Promise(r => setTimeout(r, 800));

        const diseaseNameEl = document.getElementById('res-disease-name');
        if (diseaseNameEl) diseaseNameEl.innerText = data.final_class;

        // Populate corrected plant name
        const detectedPlantEl = document.getElementById('res-detected-plant');
        if (detectedPlantEl) {
            if (data.detected_plant && data.detected_plant.toLowerCase() !== cropNameVal.toLowerCase() && data.detected_plant !== "Unknown") {
                detectedPlantEl.innerHTML = `<i class="fa-solid fa-circle-info" style="color:var(--gold);"></i> <strong>AI Correction:</strong> Detected as <strong>${data.detected_plant}</strong> (You entered: ${cropNameVal})`;
            } else {
                detectedPlantEl.innerHTML = `<strong>Plant Species:</strong> ${data.detected_plant || cropNameVal}`;
            }
        }
        
        const causesEl = document.getElementById('res-causes');
        if (causesEl) causesEl.innerText = data.why_it_happens.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\*/g, '');
        
        const treatmentEl = document.getElementById('res-treatment');
        if (treatmentEl) treatmentEl.innerText = data.how_to_control.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\*/g, '');

        const confPercent = Math.round(data.confidence * 100);
        const confLabelEl = document.getElementById('confLabel');
        if (confLabelEl) confLabelEl.textContent = confPercent + '%';
        
        setTimeout(() => { 
            const confFillEl = document.getElementById('confFill');
            if (confFillEl) confFillEl.style.width = confPercent + '%'; 
        }, 200);

        const badge = document.getElementById('resultBadge');
        const solutionBlock = document.getElementById('solutionBlock');
        const isHealthy = data.is_healthy;

        if (isHealthy) {
            if (badge) {
                badge.className = 'result-badge healthy';
                badge.innerHTML = '<i class="fa-solid fa-circle-check"></i> ' + (i18n[langKey]?.stat_healthy || 'Plant is Healthy');
            }
            if (solutionBlock) {
                solutionBlock.style.borderLeftColor = 'var(--green-accent)';
                solutionBlock.style.background = 'var(--green-pale)';
            }
        } else {
            if (badge) {
                badge.className = 'result-badge diseased';
                badge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> ' + (i18n[langKey]?.btn_detect || 'Disease Detected');
            }
            if (solutionBlock) {
                solutionBlock.style.borderLeftColor = '#e63946';
                solutionBlock.style.background = '#fff8f8';
            }
        }
        
        const tagsWrap = document.getElementById('treatmentTags');
        if (tagsWrap) {
            tagsWrap.innerHTML = cropNameVal ? `<span class="tag">${cropNameVal}</span>` : '';
        }

        // Setup TTS string
        window.latestDiagnosisText = `Plant detected: ${data.detected_plant || cropNameVal}. Condition: ${data.final_class}. Causes: ${causesEl.innerText}. Treatment: ${treatmentEl.innerText}`;

        // DB: Save History directly to MySQL
        try {
            await fetch('/api/history', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_email: session.email,
                    crop_name: data.detected_plant || cropNameVal,
                    disease_name: data.final_class,
                    status: isHealthy ? 'healthy' : 'diseased',
                    confidence_score: confPercent
                })
            });
            updateDashboardStats();
        } catch (dbErr) {
            console.error("Failed to save history to DB", dbErr);
        }

        if (loadingState) loadingState.classList.remove('show');
        if (resultState) resultState.classList.add('show');
        
    } catch (error) { 
        console.error(error);
        if (loadingState) loadingState.classList.remove('show');
        if (emptyState) emptyState.style.display = 'block';
        if (alertBox) {
            alertBox.style.display = 'flex';
            alertBox.className = 'alert-box error';
            alertBox.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> <div>Detection Failed: ${error.message || 'Server error'}</div>`;
        }
    } finally { 
        if (detectBtn) {
            detectBtn.disabled = false; 
            const lang = document.getElementById('appLang') ? document.getElementById('appLang').value : 'en';
            detectBtn.innerHTML = `<i class="fa-solid fa-microscope"></i> ${i18n[lang]?.btn_detect || 'Detect Disease'}`;
        }
    }
  }

  window.navigate = navigate;
  window.googleSignIn = googleSignIn;
  window.logout = logout;
  window.showDashboardPage = showDashboardPage;
  window.toggleDropdown = toggleDropdown;
  window.changeLanguage = changeLanguage;
  window.toggleVoice = toggleVoice;
  window.toggleChatVoice = toggleChatVoice;
  window.toggleTTS = toggleTTS;
  window.handleFile = handleFile;
  window.handleDrop = handleDrop;
  window.clearImage = clearImage;
  window.checkReady = checkReady;
  window.runDiagnosis = runDiagnosis;
  window.contactSubmit = contactSubmit;
  window.switchLoginTab = switchLoginTab;
  window.resetLoginStep = resetLoginStep;
  window.togglePass = togglePass;
  window.checkStrength = checkStrength;
  window.filterDiseases = filterDiseases;
  window.sendChatMessage = sendChatMessage;
  window.openCamera = openCamera;
  window.closeCamera = closeCamera;
  window.snapPicture = snapPicture;
  window.showForgotPasswordForm = showForgotPasswordForm;
  window.showLoginForm = showLoginForm;
  window.speakDiagnosis = speakDiagnosis;
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index(): 
    return HTML_CONTENT

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run direct CNN training on raw folders.")
    parser.add_argument("--serve", action="store_true", help="Run the FastAPI web server.")
    args = parser.parse_args()

    if args.train:
        train_direct(RAW_DATASET_DIR)
    elif args.serve:
        # Pre-flight check for DB
        print("Checking Database Connection...")
        conn = get_db_connection()
        if conn:
            print("[SUCCESS] Connected to XAMPP MySQL Database 'plantcare_db'!")
            conn.close()
        else:
            print("[WARNING] Could not connect to MySQL Database. Ensure XAMPP is running.")

        local_ip = get_local_ip()
        print("\n" + "="*50)
        print(">>> WEB SERVER SUCCESSFULLY STARTED! <<<")
        print("="*50)
        print(f"To use on THIS computer, click here: http://localhost:8000")
        print(f"To use on your MOBILE PHONE, open:   http://{local_ip}:8000")
        print("="*50 + "\n")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    else:
        print("\n[ERROR] You must specify a command.")
        print("Run 'python plant_disease_system.py --train' to train the model.")
        print("Run 'python plant_disease_system.py --serve' to start the web server.")