# 🎙️ Exam Verification System (Audio)

A Flask-based web application for verifying exam integrity via audio recordings.  
Upload or record audio, process it, and ensure the exam session is authentic and secure.

🔗 **GitHub Repo:** [nnamdiindu/Exam_Verification_System_Audio](https://github.com/nnamdiindu/Exam_Verification_System_Audio)  

---

## 📸 Screenshots

> Replace the image URLs below with your actual screenshots.

### 🎧 Upload / Record Audio  
<img width="1265" height="745" alt="image" src="https://github.com/user-attachments/assets/b02f36ed-3b80-4529-87ae-c31ad7b4279f" />

### 🔍 Verification Interface  
<img width="1262" height="785" alt="image" src="https://github.com/user-attachments/assets/498f8353-b4e0-424f-8035-4106349b74d2" />

### ✅ Login Page
<img width="680" height="659" alt="image" src="https://github.com/user-attachments/assets/71aa0137-267a-454c-b23e-c299266d1605" />


---

## 🚀 Features

- 🎤 **Audio Upload or Recording** — Users can upload audio files (e.g. WAV, MP3) or record directly in-browser.  
- 🧠 **Audio Processing / Analysis** — Process the audio for verification (e.g. speaker check, noise detection).  
- ✅ **Verification Results** — Show a summary of whether the audio passes integrity checks.  
- 💾 **Database Logging** — Store verification sessions and metadata, including timestamp, user details, and results.  
- 🔐 **Secure File Handling** — Temporary storage and sanitization of audio files.

---

## 🛠️ Built With

- **Python 3** + **Flask**  
- **Flask-WTF / WTForms** (if using forms)  
- **SQLAlchemy / Flask-Migrate** (if using a database)  
- **JavaScript / HTML5** (for client-side recording)  
- **Audio libraries** (e.g. `pydub`, `wave`, or others, depending on your implementation)  
- **Bootstrap / CSS** for UI (optional)

---

## 📦 Installation (Local Setup)

```bash
# 1. Clone the repository
git clone https://github.com/nnamdiindu/Exam_Verification_System_Audio.git
cd Exam_Verification_System_Audio

# 2. Create and activate a virtual environment
python -m venv venv
# On macOS / Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a .env file in the project root
# Example .env:
# SECRET_KEY=your-secret-key
# DATABASE_URL=sqlite:///verification.db

# 5. Run database migrations (if applicable)
flask db upgrade

# 6. Run the development server
flask run
