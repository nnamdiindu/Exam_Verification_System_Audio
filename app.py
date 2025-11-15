import os
from datetime import datetime, timezone, date, time
from decimal import Decimal
from ensurepip import bootstrap
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
    # Look up student by registration number
    reg_number = request.json.get("registration_number")

    if not reg_number:
        return jsonify({
            "error": "Registration number required"
        }), 400

    # student = Student.query.filter_by(registration_number=reg_number).first()
    student = db.session.execute(select(Student).where(Student.registration_number == reg_number)).scalar()

    if not student:
        return jsonify({
            "error": "Student not found"
        }), 404

    return jsonify({
        "id": student.id,
        "registration_number": student.registration_number,
        "first_name": student.first_name,
        "last_name": student.last_name,
        "department": student.department or 'N/A',
        "has_fingerprint": student.has_fingerprint,
        "fingerprint_count": len(student.fingerprints)
    })


@app.route("/verification")
def verification():
    # Live verification page
    active_exams = db.session.execute(
        select(Exam).where(
            Exam.status.in_(["scheduled", "active"])
        ).order_by(Exam.exam_date, Exam.start_time)
    ).scalars().all()

    return render_template("verification.html", active_exams=active_exams)


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