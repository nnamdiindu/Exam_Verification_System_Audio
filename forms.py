from flask_wtf import FlaskForm
from wtforms.fields.datetime import DateField
from wtforms.fields.simple import StringField, SubmitField, TextAreaField, PasswordField
from wtforms.validators import DataRequired

class UserProfile(FlaskForm):
    first_name = StringField("First Name", validators=[DataRequired()])
    last_name = StringField("Last Name", validators=[DataRequired()])
    registration_number = StringField("Reg No: ", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired()])
    department = StringField("Department", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired()])
    submit = SubmitField("Submit")