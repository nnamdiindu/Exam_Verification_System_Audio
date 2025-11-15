import json
import os
from datetime import datetime, timezone, date, time
from decimal import Decimal
from typing import Optional, List
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, url_for, flash
from flask_login import UserMixin, LoginManager, current_user, login_user, logout_user
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, ForeignKey, DateTime, Boolean, Numeric, LargeBinary, Date, Time, func, select, \
    distinct, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from werkzeug.security import generate_password_hash
from werkzeug.utils import redirect
import hashlib
import base64
from sqlalchemy import and_, or_
from voice_recognition import VoiceBiometricSystem as DigitalPersonaUSBScanner
from forms import UserProfile
from flask_bootstrap import Bootstrap5
import hashlib
import base64
from sqlalchemy import and_, or_
from voice_recognition import VoiceBiometricSystem as DigitalPersonaUSBScanner


def ensure_timezone_aware(dt):
    """Ensure datetime is timezone-aware (UTC)"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

# Initialize voice biometric system (add after app setup)
biometric_scanner = DigitalPersonaUSBScanner()

# Scanner Persona Configuration
SCANNER_PERSONA = {
    "name": "Voice Biometric",
    "version": "2.1.4",
    "responses": {
        "idle": "SYSTEM READY - AWAITING STUDENT DATA...",
        "student_found": "STUDENT VERIFIED - VOICE CAPTURE AUTHORIZED",
        "student_not_found": "ERROR: STUDENT RECORD NOT FOUND IN DATABASE",
        "already_enrolled": "WARNING: VOICE PRINT ALREADY EXISTS FOR THIS STUDENT",
        "scanning_start": "INITIATING VOICE CAPTURE SEQUENCE...",
        "scanning_progress": "ANALYZING VOICE PATTERNS AND FEATURES...",
        "scanning_complete": "VOICE TEMPLATE GENERATED SUCCESSFULLY",
        "enrollment_start": "ENCRYPTING AND STORING VOICE BIOMETRIC DATA...",
        "enrollment_success": "ENROLLMENT COMPLETE - VOICE BIOMETRIC DATA SECURED",
        "enrollment_error": "ENROLLMENT FAILED - PLEASE RETRY OPERATION",
        "quality_low": "VOICE QUALITY INSUFFICIENT - PLEASE SPEAK AGAIN",
        "system_error": "SYSTEM ERROR - PLEASE CONTACT ADMINISTRATOR",
        "hardware_error": "MICROPHONE NOT RESPONDING - CHECK CONNECTION"
    }
}

def generate_scanner_response(status, additional_data=None):
    """Generate standardized scanner persona responses"""
    response = {
        "scanner_name": SCANNER_PERSONA["name"],
        "scanner_version": SCANNER_PERSONA["version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "message": SCANNER_PERSONA["responses"].get(status, "UNKNOWN STATUS"),
        "data": additional_data or {}
    }
    return response


app = Flask(__name__)

load_dotenv()

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DB_URI")
app.secret_key = os.environ.get("SECRET_KEY")
bootstrap = Bootstrap5(app)


login_manager = LoginManager()
login_manager.init_app(app)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
db.init_app(app)
migrate = Migrate(app, db)


class Student(db.Model):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    registration_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    department: Mapped[Optional[str]] = mapped_column(String(100))


    # Relationships
    voiceprints: Mapped[List["VoiceprintTemplate"]] = relationship(
        back_populates="student",
        lazy="selectin",  # Changed from "select" for better performance
        cascade="all, delete-orphan"
    )
    registrations: Mapped[List["ExamRegistration"]] = relationship(
        back_populates="student",
        lazy="select"
    )

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def has_voiceprint(self):
        """Check if student has an active voiceprint enrolled"""
        return any(vp.is_active for vp in self.voiceprints)

    @property
    def active_voiceprint(self):
        """Get the most recent active voiceprint"""
        active = [vp for vp in self.voiceprints if vp.is_active]
        return active[-1] if active else None

    def __repr__(self):
        return f"<Student {self.registration_number}: {self.full_name}>"


class VoiceprintTemplate(db.Model):
    """Voice biometric template storage"""
    __tablename__ = "voiceprint_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(Integer, ForeignKey("students.id"), nullable=False, index=True)

    # Voice template data (stored as JSON string)
    template_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    template_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    # Audio metadata
    sample_rate: Mapped[int] = mapped_column(Integer, default=16000)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=3)
    audio_format: Mapped[str] = mapped_column(String(20), default="MFCC")

    # Quality metrics
    quality_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    pitch_mean: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    confidence_level: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))

    # Status and timestamps
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    enrollment_date: Mapped[datetime] = mapped_column(DateTime, nullable=False,
                                                      default=lambda: datetime.now(timezone.utc))
    last_verified: Mapped[Optional[datetime]] = mapped_column(DateTime)
    verification_count: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    student: Mapped["Student"] = relationship(back_populates="voiceprints")
    verification_attempts: Mapped[List["VerificationAttempt"]] = relationship(
        back_populates="matched_template",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Voiceprint {self.id} - Student {self.student_id} - Quality: {self.quality_score}>"


class Exam(db.Model):
    __tablename__ = "exams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_code: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    exam_title: Mapped[str] = mapped_column(String(255), nullable=False)
    exam_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[Optional[time]] = mapped_column(Time)
    venue: Mapped[Optional[str]] = mapped_column(String(255))
    duration_minutes: Mapped[int] = mapped_column(Integer, default=180)
    status: Mapped[str] = mapped_column(String(20), default="scheduled",
                                        index=True)  # scheduled, active, completed, cancelled
    max_capacity: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc),
                                                 onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    registrations: Mapped[List["ExamRegistration"]] = relationship(
        back_populates="exam",
        lazy="select",
        cascade="all, delete-orphan"
    )
    verifications: Mapped[List["VerificationAttempt"]] = relationship(
        back_populates="exam",
        lazy="select",
        cascade="all, delete-orphan"
    )
    sessions: Mapped[List["VerificationSession"]] = relationship(
        back_populates="exam",
        cascade="all, delete-orphan"
    )
    attendance_records: Mapped[List["AttendanceRecord"]] = relationship(
        back_populates="exam",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Exam {self.exam_code}: {self.exam_title}>"


class ExamRegistration(db.Model):
    __tablename__ = "exam_registrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False, index=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False, index=True)
    registration_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    seat_number: Mapped[Optional[str]] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="registered")  # registered, verified, absent

    # Relationships
    student: Mapped["Student"] = relationship(back_populates="registrations")
    exam: Mapped["Exam"] = relationship(back_populates="registrations")

    def __repr__(self):
        return f"<ExamRegistration Student:{self.student_id} Exam:{self.exam_id}>"


class VerificationAttempt(db.Model):
    """Record of each voice verification attempt"""
    __tablename__ = "verification_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False, index=True)
    student_id: Mapped[Optional[int]] = mapped_column(ForeignKey("students.id"), nullable=True, index=True)
    template_matched: Mapped[Optional[int]] = mapped_column(ForeignKey("voiceprint_templates.id"), index=True)

    # Verification details
    verification_timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc),
                                                             index=True)
    verification_status: Mapped[str] = mapped_column(String(20),
                                                     nullable=False)  # success, failed, duplicate, not_registered
    confidence_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))

    # Additional info
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    exam: Mapped["Exam"] = relationship(back_populates="verifications")
    student: Mapped[Optional["Student"]] = relationship()
    matched_template: Mapped[Optional["VoiceprintTemplate"]] = relationship(
        back_populates="verification_attempts"
    )

    def __repr__(self):
        return f"<VerificationAttempt {self.id} - {self.verification_status} - {self.confidence_score}%>"


class AttendanceRecord(db.Model):
    """Track confirmed student attendance after successful verification"""
    __tablename__ = "attendance_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False, index=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False, index=True)
    verification_attempt_id: Mapped[int] = mapped_column(ForeignKey("verification_attempts.id"), nullable=False,
                                                         unique=True)

    check_in_time: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    seat_number: Mapped[Optional[str]] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="present")  # present, late, left_early

    # Additional tracking
    check_out_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    exam: Mapped["Exam"] = relationship(back_populates="attendance_records")
    student: Mapped["Student"] = relationship()
    verification_attempt: Mapped["VerificationAttempt"] = relationship()

    def __repr__(self):
        return f"<Attendance Exam:{self.exam_id} Student:{self.student_id} - {self.status}>"


class VerificationSession(db.Model):
    """Track verification sessions for exams"""
    __tablename__ = "verification_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), nullable=False, index=True)
    operator_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)

    # Session timing
    session_start: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    session_end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # Statistics
    total_attempts: Mapped[int] = mapped_column(Integer, default=0)
    successful_verifications: Mapped[int] = mapped_column(Integer, default=0)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_attempts: Mapped[int] = mapped_column(Integer, default=0)

    # Session notes
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    exam: Mapped["Exam"] = relationship(back_populates="sessions")
    operator: Mapped["User"] = relationship()

    @property
    def success_rate(self):
        if self.total_attempts == 0:
            return 0.0
        return round((self.successful_verifications / self.total_attempts) * 100, 2)

    def __repr__(self):
        return f"<VerificationSession {self.id} - Exam:{self.exam_id} - Active:{self.is_active}>"


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="operator")  # admin, operator
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)

    def __repr__(self):
        return f"<User {self.email} - {self.role}>"


@login_manager.user_loader
def load_user(user_id):
    return db.get_or_404(User, user_id)


with app.app_context():
    db.create_all()


biometric_scanner = DigitalPersonaUSBScanner()

SCANNER_PERSONA = {
    "name": "Voice Biometric",
    "version": "2.1.4",
    "responses": {
        "idle": "SYSTEM READY - AWAITING STUDENT DATA...",
        "student_found": "STUDENT VERIFIED - VOICE CAPTURE AUTHORIZED",
        "student_not_found": "ERROR: STUDENT RECORD NOT FOUND IN DATABASE",
        "already_enrolled": "WARNING: VOICE PRINT ALREADY EXISTS FOR THIS STUDENT",
        "scanning_start": "INITIATING VOICE CAPTURE SEQUENCE...",
        "scanning_progress": "ANALYZING VOICE PATTERNS AND FEATURES...",
        "scanning_complete": "VOICE TEMPLATE GENERATED SUCCESSFULLY",
        "enrollment_start": "ENCRYPTING AND STORING VOICE BIOMETRIC DATA...",
        "enrollment_success": "ENROLLMENT COMPLETE - VOICE BIOMETRIC DATA SECURED",
        "enrollment_error": "ENROLLMENT FAILED - PLEASE RETRY OPERATION",
        "quality_low": "VOICE QUALITY INSUFFICIENT - PLEASE SPEAK AGAIN",
        "system_error": "SYSTEM ERROR - PLEASE CONTACT ADMINISTRATOR",
        "hardware_error": "MICROPHONE NOT RESPONDING - CHECK CONNECTION"
    }
}

def generate_scanner_response(status, additional_data=None):
    """Generate standardized scanner persona responses"""
    response = {
        "scanner_name": SCANNER_PERSONA["name"],
        "scanner_version": SCANNER_PERSONA["version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "message": SCANNER_PERSONA["responses"].get(status, "UNKNOWN STATUS"),
        "data": additional_data or {}
    }
    return response

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    form = UserProfile()
    if form.validate_on_submit():
        user = db.session.execute(db.select(User).where(User.email == request.form.get("email"))).scalar()

        if user:
            flash("Email has already been registered, please login", "error")
            return redirect(url_for("register"))

        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if password == confirm_password:
            hashed_and_salted_password = generate_password_hash(
                password=password,
                salt_length=8,
                method="pbkdf2:sha256"
            )

            new_profile = User(
                full_name=request.form.get("full_name"),
                email=request.form.get("email"),
                password=hashed_and_salted_password,
            )

            db.session.add(new_profile)
            db.session.commit()

            login_user(new_profile)

            return redirect(url_for("dashboard"))

        else:
            flash("Password does not match, please try again", "error")
            return redirect(url_for("register"))

    return render_template("register.html", form=form)


@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.json

        username = data["username"]
        # password = data["password"]

        result = db.session.execute(db.select(User).where(User.email == username.lower()))
        user = result.scalars().first()
        print(user)
        if user:
            login_user(user)
            return jsonify({
                "success": True,
                "redirect_url": url_for("dashboard")  # dashboard route
            })
        else:
            return jsonify({"success": False, "error": "Invalid credentials"})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    #Main Dashboard
    stats = {
        'total_students': db.session.execute(
            select(func.count()).select_from(Student)
        ).scalar(),

        'enrolled_fingerprints': db.session.execute(
            select(func.count(distinct(VoiceprintTemplate.student_id)))
        ).scalar(),

        'total_exams': db.session.execute(
            select(func.count()).select_from(Exam)
        ).scalar(),

        'active_exams': db.session.execute(
            select(func.count()).select_from(Exam).where(Exam.status == 'active')
        ).scalar()
    }

    # Recent verifications query
    stmt = select(
        VerificationAttempt,
        Student.registration_number,
        Student.first_name,
        Student.last_name,
        Exam.exam_code
    ).outerjoin(Student).join(Exam).order_by(
        VerificationAttempt.verification_timestamp.desc()
    ).limit(5)

    recent_verifications = db.session.execute(stmt).all()
    return render_template("dashboard.html",
                           stats=stats,
                           recent_verifications=recent_verifications,
                           current_user=current_user)


@app.route("/register-course", methods=["GET", "POST"])
def course_registration():
    available_courses = Exam.query.order_by(Exam.exam_date.desc()).all()
    registered_courses = ExamRegistration.query.filter_by(
        student_id=current_user.id
    ).join(Exam).order_by(ExamRegistration.registration_date.desc()).all()

    if request.method == "POST":
        selected_exam_codes = request.form.getlist("selected_exam_codes")

        # Check if any exams were selected
        if not selected_exam_codes:
            flash("Please select at least one exam to register for.", "warning")
            return render_template("course-registration.html", exams=available_courses)

        try:
            # Get exam IDs from the selected exam codes
            selected_exams = Exam.query.filter(Exam.exam_code.in_(selected_exam_codes)).all()
            exam_ids = [exam.id for exam in selected_exams]

            # Check for existing registrations to prevent duplicates
            existing_registrations = ExamRegistration.query.filter(
                ExamRegistration.student_id == current_user.id,
                ExamRegistration.exam_id.in_(exam_ids)
            ).all()

            existing_exam_ids = [reg.exam_id for reg in existing_registrations]
            new_exam_ids = [exam_id for exam_id in exam_ids if exam_id not in existing_exam_ids]

            if existing_registrations:
                existing_exam_codes = [exam.exam_code for exam in selected_exams if exam.id in existing_exam_ids]
                flash(f"You are already registered for: {', '.join(existing_exam_codes)}", "warning")

            if not new_exam_ids:
                flash("No new registrations to process.", "info")
                return render_template("course-registration.html", exams=available_courses)

            # Create new registrations
            successful_registrations = 0
            for exam in selected_exams:
                if exam.id in new_exam_ids:
                    # Create new ExamRegistration record
                    registration = ExamRegistration(
                        student_id=current_user.id,
                        exam_id=exam.id,
                        registration_date=datetime.now(timezone.utc),
                        seat_number=None  # Will be assigned later by admin
                    )
                    db.session.add(registration)
                    successful_registrations += 1

            # Commit all changes to database
            db.session.commit()

            flash(f"Successfully registered for {successful_registrations} exam(s)!", "success")
            print(f"Student {current_user.id} registered for {successful_registrations} exams")

            # Redirect to prevent form resubmission
            return redirect(url_for('course_registration'))

        except Exception as e:
            # Rollback in case of error
            db.session.rollback()
            flash(f"Registration failed: {str(e)}", "error")
            print(f"Registration error: {str(e)}")

    return render_template("course-registration.html", exams=available_courses, exam_data=registered_courses)

@app.route("/results")
def view_result():
    return render_template("result.html", current_user=current_user)

@app.route("/enrollment")
def enrollment():
    """Enhanced enrollment page with real scanner integration"""
    scanner_status = biometric_scanner.is_connected
    scanner_type = biometric_scanner.scanner_type

    return render_template("enrollment.html",
                           scanner_available=scanner_status,
                           scanner_type=scanner_type,
                           scanner_persona=SCANNER_PERSONA)


@app.route("/api/lookup-student", methods=["POST"])
def lookup_student():
    """Look up student by registration number"""
    try:
        reg_number = request.json.get("registration_number")

        if not reg_number:
            return jsonify({"error": "Registration number required"}), 400

        # Query with eager loading of voiceprints
        student = db.session.execute(
            select(Student)
            .where(Student.registration_number == reg_number)
        ).scalar_one_or_none()

        if not student:
            return jsonify({"error": "Student not found"}), 404

        # Get voiceprint info
        has_voiceprint = student.has_voiceprint
        active_voiceprint = student.active_voiceprint

        return jsonify({
            "id": student.id,
            "registration_number": student.registration_number,
            "first_name": student.first_name,
            "last_name": student.last_name,
            "department": student.department or 'N/A',
            "has_fingerprint": has_voiceprint,  # Keep name for frontend compatibility
            "fingerprint_count": len([vp for vp in student.voiceprints if vp.is_active]),
            "last_enrollment": active_voiceprint.enrollment_date.isoformat() if active_voiceprint else None
        })

    except Exception as e:
        print(f"Student lookup error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/verification")
def verification():
    # Live verification page
    active_exams = db.session.execute(
        select(Exam).where(
            Exam.status.in_(["scheduled", "active"])
        ).order_by(Exam.exam_date, Exam.start_time)
    ).scalars().all()

    return render_template("verification.html", active_exams=active_exams)


@app.route("/api/scanner-status")
def get_scanner_status():
    """Get current scanner system status"""
    try:
        scanner_data = {
            "operational": biometric_scanner.is_connected,
            "scanner_type": biometric_scanner.scanner_type,
            "temperature": "Normal",
            "last_calibration": "2024-01-15T10:30:00Z"
        }

        if biometric_scanner.is_connected:
            response = generate_scanner_response("idle", scanner_data)
        else:
            response = generate_scanner_response("hardware_error", scanner_data)

        return jsonify(response)

    except Exception as e:
        return jsonify(generate_scanner_response("system_error", {
            "error": str(e)
        })), 500


@app.route("/api/initiate-scan", methods=["POST"])
def initiate_fingerprint_scan():
    """Start voice scanning process"""
    try:
        data = request.json
        student_id = data.get('student_id')

        if not student_id:
            return jsonify(generate_scanner_response("system_error", {
                "error": "Student ID required"
            })), 400

        # Verify student exists
        student = db.session.execute(
            select(Student).where(Student.id == student_id)
        ).scalar_one_or_none()

        if not student:
            return jsonify(generate_scanner_response("student_not_found"))

        # Check if already enrolled
        if student.has_voiceprint:
            return jsonify(generate_scanner_response("already_enrolled", {
                "student_name": student.full_name,
                "existing_templates": len(student.voiceprints)
            }))

        # Start scanning process
        return jsonify(generate_scanner_response("scanning_start", {
            "student_name": student.full_name,
            "student_reg": student.registration_number
        }))

    except Exception as e:
        return jsonify(generate_scanner_response("system_error", {
            "error": str(e)
        })), 500


@app.route("/api/capture-fingerprint", methods=["POST"])
def capture_fingerprint():
    """Capture voice biometric from microphone - FIXED"""
    try:
        data = request.json
        student_id = data.get('student_id')

        if not biometric_scanner.is_connected:
            return jsonify(generate_scanner_response("hardware_error")), 500

        print(f"Starting voice capture for student {student_id}...")

        # Capture voice from microphone
        scan_result = biometric_scanner.capture_fingerprint()

        print(f"Voice captured. Quality: {scan_result['quality_score']}")

        if scan_result['quality_score'] < 50:  # Lowered threshold
            return jsonify(generate_scanner_response("quality_low", {
                "quality_score": scan_result['quality_score'],
                "retry_recommended": True
            }))

        # Extract voice-specific metrics safely
        try:
            template_data = scan_result.get('template_data', {})
            pitch_mean = template_data.get('pitch_mean', 0)
        except Exception as e:
            print(f"Warning: Could not extract pitch: {e}")
            pitch_mean = 0

        # CRITICAL: Return the JSON template string, not base64 audio
        response_data = {
            "quality_score": scan_result['quality_score'],
            "minutiae_count": len(str(scan_result.get('template', ''))),
            "pitch_mean": float(pitch_mean) if pitch_mean else 0,
            "scan_data": scan_result['template'],  # ← This is the JSON template string!
            "audio_data": scan_result.get('image_data', '')  # Legacy field
        }

        return jsonify(generate_scanner_response("scanning_complete", response_data))

    except Exception as e:
        print(f"Capture error: {e}")
        import traceback
        traceback.print_exc()

        return jsonify(generate_scanner_response("system_error", {
            "error": str(e)
        })), 500


@app.route("/api/enroll-fingerprint", methods=["POST"])
def enroll_fingerprint():
    """Enroll captured voiceprint into database - SIMPLIFIED"""
    try:
        data = request.json
        student_id = data.get('student_id')
        scan_data = data.get('scan_data')  # This is now the JSON template string!
        quality_score = data.get('quality_score', 85.0)

        print(f"\n=== ENROLLMENT REQUEST ===")
        print(f"Student ID: {student_id}")
        print(f"Quality Score: {quality_score}")

        if not student_id or not scan_data:
            return jsonify(generate_scanner_response("enrollment_error", {
                "error": "Missing required data"
            })), 400

        # Get student
        student = db.session.execute(
            select(Student).where(Student.id == student_id)
        ).scalar_one_or_none()

        if not student:
            return jsonify(generate_scanner_response("student_not_found")), 404

        print(f"Student found: {student.full_name}")

        # Deactivate old voiceprints
        if student.voiceprints:
            for old_vp in student.voiceprints:
                old_vp.is_active = False
            print(f"Deactivated {len(student.voiceprints)} old voiceprints")

        # CRITICAL FIX: scan_data is already a JSON string!
        # Just encode it as UTF-8 bytes for storage
        try:
            # Verify it's valid JSON
            template_dict = json.loads(scan_data)
            print(f"✓ Template is valid JSON")

            # Extract pitch for database
            pitch_mean = template_dict.get('pitch_mean', 0)

            # Encode as UTF-8 bytes for database storage
            template_data = scan_data.encode('utf-8')

            print(f"✓ Template ready: {len(scan_data)} chars, {len(template_data)} bytes")

        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON template: {e}")
            return jsonify(generate_scanner_response("enrollment_error", {
                "error": "Invalid template format"
            })), 400

        # Create template hash
        template_hash = hashlib.sha256(template_data).hexdigest()
        print(f"Template hash: {template_hash[:16]}...")

        # Check for duplicate hash
        existing = db.session.execute(
            select(VoiceprintTemplate)
            .where(VoiceprintTemplate.template_hash == template_hash)
        ).scalar_one_or_none()

        if existing:
            print(f"⚠️  Duplicate template hash found!")
            return jsonify(generate_scanner_response("enrollment_error", {
                "error": "This voiceprint has already been enrolled"
            })), 400

        # Create voiceprint template record
        voiceprint = VoiceprintTemplate(
            student_id=student_id,
            template_data=template_data,  # UTF-8 encoded JSON string
            template_hash=template_hash,
            sample_rate=16000,
            duration_seconds=5,
            audio_format="MFCC",
            quality_score=Decimal(str(quality_score)),
            pitch_mean=Decimal(str(pitch_mean)) if pitch_mean else None,
            enrollment_date=datetime.now(timezone.utc),
            is_active=True,
            verification_count=0
        )

        db.session.add(voiceprint)
        db.session.commit()

        print(f"✓ Voiceprint enrolled successfully! ID: {voiceprint.id}")
        print(f"=== ENROLLMENT COMPLETE ===\n")

        return jsonify(generate_scanner_response("enrollment_success", {
            "student_name": student.full_name,
            "template_id": voiceprint.id,
            "quality_score": float(voiceprint.quality_score),
            "enrollment_time": voiceprint.enrollment_date.isoformat()
        }))

    except Exception as e:
        db.session.rollback()
        print(f"✗ ENROLLMENT ERROR: {e}")
        import traceback
        traceback.print_exc()

        return jsonify(generate_scanner_response("enrollment_error", {
            "error": str(e)
        })), 500


@app.route("/api/start-verification-session", methods=["POST"])
def start_verification_session():
    """Start a new verification session for an exam"""
    try:
        data = request.json
        exam_id = data.get('exam_id')
        operator_id = current_user.id if current_user.is_authenticated else 1

        if not exam_id:
            return jsonify({"error": "Exam ID required"}), 400

        # Check if exam exists and is active
        exam = db.session.execute(
            select(Exam).where(Exam.id == exam_id)
        ).scalar_one_or_none()

        if not exam:
            return jsonify({"error": "Exam not found"}), 404

        if exam.status not in ['active', 'scheduled']:
            return jsonify({"error": "Exam is not available for verification"}), 400

        # End any existing active session for this exam
        existing_session = db.session.execute(
            select(VerificationSession).where(
                and_(VerificationSession.exam_id == exam_id,
                     VerificationSession.is_active == True)
            )
        ).scalar_one_or_none()

        if existing_session:
            existing_session.session_end = datetime.now(timezone.utc)
            existing_session.is_active = False

        # Create new session
        new_session = VerificationSession(
            exam_id=exam_id,
            operator_id=operator_id,
            session_start=datetime.now(timezone.utc),
            is_active=True
        )

        db.session.add(new_session)
        db.session.commit()

        return jsonify({
            "success": True,
            "session_id": new_session.id,
            "exam_title": exam.exam_title,
            "exam_code": exam.exam_code,
            "venue": exam.venue,
            "message": f"Verification session started for {exam.exam_code}"
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/verify-fingerprint", methods=["POST"])
def verify_fingerprint():
    """Enhanced voice biometric verification with duplicate detection"""
    try:
        data = request.json
        exam_id = data.get('exam_id')

        print(f"\n=== VERIFICATION REQUEST STARTED ===")
        print(f"Exam ID: {exam_id}")

        if not exam_id:
            return jsonify({"error": "Exam ID required"}), 400

        # Check if verification session is active
        active_session = db.session.execute(
            select(VerificationSession).where(
                and_(VerificationSession.exam_id == exam_id,
                     VerificationSession.is_active == True)
            )
        ).scalar_one_or_none()

        if not active_session:
            print("ERROR: No active session found")
            return jsonify({
                "success": False,
                "status": "error",
                "message": "No active verification session found"
            }), 400

        print(f"Active session found: {active_session.id}")

        # Check microphone connection
        if not biometric_scanner.is_connected:
            print("ERROR: Microphone not connected")
            return jsonify({
                "success": False,
                "status": "error",
                "message": "Microphone not available"
            }), 500

        print("Microphone connected, starting voice capture...")

        # Capture current voice sample
        try:
            current_scan = biometric_scanner.capture_fingerprint()
            print(f"Voice captured! Quality: {current_scan.get('quality_score', 'N/A')}")
        except Exception as e:
            print(f"ERROR during voice capture: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False,
                "status": "error",
                "message": f"Voice capture failed: {str(e)}"
            }), 500

        # Get all active voiceprint templates
        print("Fetching stored voice templates...")
        voiceprints = db.session.execute(
            select(VoiceprintTemplate)
            .where(VoiceprintTemplate.is_active == True)
        ).scalars().all()

        print(f"Found {len(voiceprints)} active voiceprint templates")

        if len(voiceprints) == 0:
            print("WARNING: No voiceprint templates in database!")
            return jsonify({
                "success": False,
                "status": "no_match",
                "message": "No enrolled voiceprints found in database"
            })

        best_match = None
        best_score = 0
        match_details = None

        # Match against all templates
        print("Starting voice matching process...")
        for idx, voiceprint in enumerate(voiceprints):
            try:
                # Get stored template - it's already a JSON string stored as bytes
                if isinstance(voiceprint.template_data, bytes):
                    # Don't decode - pass the bytes directly or decode if it's valid JSON
                    try:
                        stored_template = voiceprint.template_data.decode('utf-8')
                    except UnicodeDecodeError:
                        # If bytes are not UTF-8 (old binary format), skip this template
                        print(f"  - Skipping voiceprint {voiceprint.id}: Invalid format (needs re-enrollment)")
                        continue
                else:
                    stored_template = voiceprint.template_data

                print(f"Comparing with voiceprint {idx + 1}/{len(voiceprints)} (Student ID: {voiceprint.student_id})")

                # Compare
                confidence_score, is_match = biometric_scanner.compare_fingerprints(
                    current_scan['template'],
                    stored_template
                )

                print(f"  - Confidence: {confidence_score:.2f}%, Match: {is_match}")

                if confidence_score > best_score and is_match:
                    best_match = voiceprint
                    best_score = confidence_score
                    match_details = {"confidence": confidence_score}
                    print(f"  - NEW BEST MATCH! Score: {best_score:.2f}%")

            except UnicodeDecodeError as e:
                print(f"  - Skipping voiceprint {voiceprint.id}: Encoding error (old format)")
                continue
            except Exception as e:
                print(f"  - Error matching voiceprint {voiceprint.id}: {e}")
                continue

                # COMPLETE REPLACEMENT FOR THE MATCHING LOOP:

                best_match = None
                best_score = 0
                match_details = None

                # Match against all templates
                print("Starting voice matching process...")
                for idx, voiceprint in enumerate(voiceprints):
                    try:
                        # Get stored template safely
                        stored_template = None

                        if isinstance(voiceprint.template_data, bytes):
                            try:
                                # Try to decode as UTF-8 JSON string
                                stored_template = voiceprint.template_data.decode('utf-8')

                                # Validate it's actually JSON
                                import json
                                json.loads(stored_template)  # Will raise if not valid JSON

                            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                                print(f"  - Voiceprint {voiceprint.id} has invalid format (needs re-enrollment)")
                                print(f"    Error: {e}")
                                continue
                        else:
                            stored_template = voiceprint.template_data

                        if not stored_template:
                            print(f"  - Skipping empty voiceprint {voiceprint.id}")
                            continue

                        print(
                            f"Comparing with voiceprint {idx + 1}/{len(voiceprints)} (Student ID: {voiceprint.student_id})")

                        # Compare using voice biometric system
                        confidence_score, is_match = biometric_scanner.compare_fingerprints(
                            current_scan['template'],  # Current voice sample
                            stored_template  # Stored voiceprint
                        )

                        print(f"  - Confidence: {confidence_score:.2f}%, Match: {is_match}")

                        if confidence_score > best_score and is_match:
                            best_match = voiceprint
                            best_score = confidence_score
                            match_details = {"confidence": confidence_score}
                            print(f"  - NEW BEST MATCH! Score: {best_score:.2f}%")

                    except Exception as e:
                        print(f"  - Error matching voiceprint {voiceprint.id}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        print(f"\nMatching complete. Best score: {best_score:.2f}%")

        # Update session statistics
        active_session.total_attempts += 1

        # Create verification attempt record
        verification = VerificationAttempt(
            exam_id=exam_id,
            student_id=best_match.student_id if best_match else None,
            verification_timestamp=datetime.now(timezone.utc),
            verification_status="success" if best_match else "failed",
            confidence_score=Decimal(str(best_score)) if best_match else None,
            template_matched=best_match.id if best_match else None
        )

        db.session.add(verification)
        db.session.flush()

        if best_match:
            # Check for duplicate verification
            existing_attendance = db.session.execute(
                select(AttendanceRecord).where(
                    and_(AttendanceRecord.exam_id == exam_id,
                         AttendanceRecord.student_id == best_match.student_id)
                )
            ).scalar_one_or_none()

            if existing_attendance:
                print(f"DUPLICATE: Student already verified")
                active_session.duplicate_attempts += 1
                db.session.commit()

                return jsonify({
                    "success": False,
                    "status": "duplicate",
                    "message": f"Student {best_match.student.full_name} already verified for this exam",
                    "student": {
                        "id": best_match.student.id,
                        "name": best_match.student.full_name,
                        "registration_number": best_match.student.registration_number,
                        "department": best_match.student.department
                    },
                    "first_verification_time": existing_attendance.check_in_time.isoformat(),
                    "confidence_score": float(best_score)
                })

            # Check if student is registered for exam
            registration = db.session.execute(
                select(ExamRegistration).where(
                    and_(ExamRegistration.exam_id == exam_id,
                         ExamRegistration.student_id == best_match.student_id)
                )
            ).scalar_one_or_none()

            if not registration:
                print(f"ERROR: Student not registered for this exam")
                active_session.failed_attempts += 1
                verification.verification_status = "not_registered"
                db.session.commit()

                return jsonify({
                    "success": False,
                    "status": "not_registered",
                    "message": f"Student {best_match.student.full_name} is not registered for this exam",
                    "student": {
                        "id": best_match.student.id,
                        "name": best_match.student.full_name,
                        "registration_number": best_match.student.registration_number,
                        "department": best_match.student.department
                    },
                    "confidence_score": float(best_score)
                })

            # Successful verification - create attendance
            attendance = AttendanceRecord(
                exam_id=exam_id,
                student_id=best_match.student_id,
                verification_attempt_id=verification.id,
                check_in_time=datetime.now(timezone.utc),
                seat_number=registration.seat_number,
                status="present"
            )

            # Update voiceprint verification count
            best_match.verification_count += 1
            best_match.last_verified = datetime.now(timezone.utc)

            db.session.add(attendance)
            active_session.successful_verifications += 1
            db.session.commit()

            print(f"SUCCESS: Attendance recorded for {best_match.student.full_name}")
            print(f"=== VERIFICATION COMPLETE ===\n")

            return jsonify({
                "success": True,
                "status": "verified",
                "message": f"Welcome {best_match.student.full_name}!",
                "student": {
                    "id": best_match.student.id,
                    "name": best_match.student.full_name,
                    "full_name": best_match.student.full_name,
                    "registration_number": best_match.student.registration_number,
                    "department": best_match.student.department or "N/A",
                    "seat_number": registration.seat_number
                },
                "confidence_score": float(best_score),
                "confidence": float(best_score),  # For compatibility
                "verification_id": verification.id,
                "attendance_id": attendance.id,
                "check_in_time": attendance.check_in_time.isoformat(),
                "match_details": match_details
            })

        else:
            # No match found
            active_session.failed_attempts += 1
            db.session.commit()

            print(f"FAILED: No matching voice (best score: {best_score:.2f}%)")
            print(f"=== VERIFICATION COMPLETE ===\n")

            return jsonify({
                "success": False,
                "status": "no_match",
                "message": "Voice not recognized. Please try again.",
                "verification_id": verification.id,
                "best_score": float(best_score)
            })

    except Exception as e:
        db.session.rollback()
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "status": "error",
            "message": f"System error: {str(e)}"
        }), 500


@app.route("/api/end-verification-session", methods=["POST"])
def end_verification_session():
    """End the current verification session"""
    try:
        data = request.json
        exam_id = data.get('exam_id')

        # Find active session
        active_session = db.session.execute(
            select(VerificationSession).where(
                and_(VerificationSession.exam_id == exam_id,
                     VerificationSession.is_active == True)
            )
        ).scalar_one_or_none()

        if not active_session:
            return jsonify({"error": "No active session found"}), 404

        # End session
        active_session.session_end = datetime.now(timezone.utc)
        active_session.is_active = False
        db.session.commit()

        # Calculate session statistics
        session_duration = active_session.session_end - active_session.session_start

        return jsonify({
            "success": True,
            "message": "Verification session ended",
            "session_summary": {
                "duration_minutes": int(session_duration.total_seconds() / 60),
                "total_attempts": active_session.total_attempts,
                "successful_verifications": active_session.successful_verifications,
                "failed_attempts": active_session.failed_attempts,
                "duplicate_attempts": active_session.duplicate_attempts,
                "success_rate": active_session.success_rate
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/verification-session-stats/<int:exam_id>")
def get_verification_session_stats(exam_id):
    """Get current session statistics"""
    try:
        active_session = db.session.execute(
            select(VerificationSession).where(
                and_(VerificationSession.exam_id == exam_id,
                     VerificationSession.is_active == True)
            )
        ).scalar_one_or_none()

        if not active_session:
            return jsonify({
                "session_active": False,
                "error": "No active session",
                "total_attempts": 0,
                "successful_verifications": 0,
                "failed_attempts": 0,
                "duplicate_attempts": 0,
                "success_rate": 0,
                "recent_verifications": []
            }), 200

        # Calculate session duration safely
        try:
            current_time = datetime.now(timezone.utc)
            session_start = ensure_timezone_aware(active_session.session_start)

            session_duration = current_time - session_start
            duration_minutes = int(session_duration.total_seconds() / 60)
        except Exception as e:
            print(f"Error calculating duration: {e}")
            duration_minutes = 0

        # Get recent verifications
        try:
            session_start = ensure_timezone_aware(active_session.session_start)

            recent_verifications = db.session.execute(
                select(VerificationAttempt, Student.first_name, Student.last_name, Student.registration_number)
                .outerjoin(Student)
                .where(
                    and_(VerificationAttempt.exam_id == exam_id,
                         VerificationAttempt.verification_timestamp >= session_start)
                )
                .order_by(VerificationAttempt.verification_timestamp.desc())
                .limit(10)
            ).all()

            recent_list = []
            for verification, first_name, last_name, reg_number in recent_verifications:
                recent_list.append({
                    "timestamp": verification.verification_timestamp.isoformat(),
                    "status": verification.verification_status,
                    "student_name": f"{first_name} {last_name}" if first_name else "Unknown",
                    "registration_number": reg_number or "N/A",
                    "confidence": float(verification.confidence_score) if verification.confidence_score else 0
                })
        except Exception as e:
            print(f"Error fetching recent verifications: {e}")
            recent_list = []

        # Calculate success rate
        if active_session.total_attempts > 0:
            success_rate = round(
                (active_session.successful_verifications / active_session.total_attempts) * 100, 2
            )
        else:
            success_rate = 0

        return jsonify({
            "session_active": True,
            "session_duration_minutes": duration_minutes,
            "total_attempts": active_session.total_attempts or 0,
            "successful_verifications": active_session.successful_verifications or 0,
            "failed_attempts": active_session.failed_attempts or 0,
            "duplicate_attempts": active_session.duplicate_attempts or 0,
            "success_rate": success_rate,
            "recent_verifications": recent_list
        })

    except Exception as e:
        print(f"Session stats error: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "session_active": False,
            "error": str(e),
            "total_attempts": 0,
            "successful_verifications": 0,
            "failed_attempts": 0,
            "duplicate_attempts": 0,
            "success_rate": 0,
            "recent_verifications": []
        }), 200


@app.route("/exams")
def exam_management():
    exams = Exam.query.order_by(Exam.exam_date.desc()).all()

    # Add registration counts
    exam_data = []
    for exam in exams:
        registered_count = ExamRegistration.query.filter_by(exam_id=exam.id).count()
        verified_count = VerificationAttempt.query.filter_by(
            exam_id=exam.id, verification_status='success'
        ).count()

        exam_data.append({
            'exam': exam,
            'registered_count': registered_count,
            'verified_count': verified_count
        })

    return render_template('exam_management.html', exam_data=exam_data)



@app.route('/api/create-exam', methods=['POST'])
def create_exam():
    """Create new exam"""
    try:
        data = request.json

        exam = Exam(
            exam_code=data['exam_code'],
            exam_title=data['exam_title'],
            exam_date=datetime.strptime(data['exam_date'], '%Y-%m-%d').date(),
            start_time=datetime.strptime(data['start_time'], '%H:%M').time(),
            venue=data.get('venue', ''),
            duration_minutes=int(data.get('duration_minutes', 180))
        )

        db.session.add(exam)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Exam created successfully',
            'exam_id': exam.id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/update-exam-status/<int:exam_id>', methods=['POST'])
def update_exam_status(exam_id):
    """Update exam status"""
    try:
        new_status = request.json.get('status')

        exam = Exam.query.get(exam_id)
        if not exam:
            return jsonify({'error': 'Exam not found'}), 404

        exam.status = new_status
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Exam status updated to {new_status}'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/register-student-for-exam', methods=['POST'])
def register_student_for_exam():
    """Register student for an exam"""
    try:
        data = request.json
        student_id = data.get('student_id')
        exam_id = data.get('exam_id')

        # Check if already registered
        existing = ExamRegistration.query.filter_by(
            student_id=student_id, exam_id=exam_id
        ).first()

        if existing:
            return jsonify({'error': 'Student already registered for this exam'}), 400

        registration = ExamRegistration(
            student_id=student_id,
            exam_id=exam_id,
            seat_number=data.get('seat_number')
        )

        db.session.add(registration)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Student registered successfully'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)