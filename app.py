from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')  # Change this to a strong secret key

# === Database setup ===
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# === Login manager setup ===
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# === Models ===
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    read = db.Column(db.Boolean, default=False)

class AboutUs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    match_format = db.Column(db.String(10), nullable=False)
    team1 = db.Column(db.String(50), nullable=False)
    team2 = db.Column(db.String(50), nullable=False)
    venue = db.Column(db.String(100), nullable=False)
    toss_winner = db.Column(db.String(50), nullable=False)
    predicted_winner = db.Column(db.String(50), nullable=False)
    probabilities = db.Column(db.Text)  # JSON string
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    team1 = db.Column(db.String(50), nullable=False)
    team2 = db.Column(db.String(50), nullable=False)
    match_format = db.Column(db.String(10), nullable=False)
    venue = db.Column(db.String(100), nullable=False)
    match_date = db.Column(db.Date, nullable=True) 


# Create all tables
with app.app_context():
    db.create_all()

# === Load ML model and encoders ===
model = joblib.load(os.path.join(basedir, 'model.pkl'))
team_encoder = joblib.load(os.path.join(basedir, 'le_team.pkl'))
venue_encoder = joblib.load(os.path.join(basedir, 'le_venue.pkl'))

# === Flask-Login user loader ===
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================== ROUTES ==================

@app.route('/')
def home():
    teams = list(team_encoder.classes_)
    venues = list(venue_encoder.classes_)
    recommendations = Recommendation.query.all()
    return render_template('index.html', teams=teams, venues=venues, recommendations=recommendations)

@app.route('/message_us', methods=['GET', 'POST'])
@login_required
def message_us():
    if request.method == 'POST':
        topic = request.form.get('topic')
        content = request.form.get('content')
        if not topic or not content:
            flash("Please fill all fields!", "danger")
            return redirect(url_for('message_us'))

        new_msg = Message(user_id=current_user.id, topic=topic, content=content)
        db.session.add(new_msg)
        db.session.commit()
        flash("Message submitted successfully!", "success")
        return redirect(url_for('message_us'))
    return render_template('message_us.html')

@app.route('/history')
@login_required
def history():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    predictions = []
    for p in user_predictions:
        predictions.append({
            'id': p.id,
            'format': p.match_format,
            'team1': p.team1,
            'team2': p.team2,
            'venue': p.venue,
            'toss_winner': p.toss_winner,
            'predicted_winner': p.predicted_winner,
            'probabilities': json.loads(p.probabilities),
            'timestamp': p.timestamp
        })
    return render_template('history.html', predictions=predictions)

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.user_id != current_user.id:
        flash("You cannot delete this prediction.", "danger")
        return redirect(url_for('history'))

    db.session.delete(prediction)
    db.session.commit()
    flash("Prediction deleted successfully.", "success")
    return redirect(url_for('history'))

@app.route('/about')
def about():
    info = AboutUs.query.first()
    return render_template('about.html', info=info)

# --------------------
# Unified login & register
# --------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            flash("Invalid username or password.", "login_error")
            return redirect(url_for('login'))

        login_user(user)
        if user.is_admin:
            return redirect(url_for('admin_panel'))
        else:
            return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully!", "success")
    return redirect(url_for('home'))

# --------------------
# Register route
# --------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        gender = request.form.get('gender')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Check for existing username/email
        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "danger")
            return redirect(url_for('register'))

        if len(password) < 6:
            flash("Password must be at least 6 characters!", "danger")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')

        # === SINGLE ADMIN SETUP ===
        # Replace these credentials with your chosen admin
        ADMIN_USERNAME = "admin"
        ADMIN_PASSWORD = "admin123"  # plaintext only for first creation, hashed in DB

        if username == ADMIN_USERNAME:
            new_user = User(username=username, email=email, gender=gender, password=hashed_pw, is_admin=True)
        else:
            new_user = User(username=username, email=email, gender=gender, password=hashed_pw, is_admin=False)

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please log in.", "register_success")

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    new_email = request.form.get('email')
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')

    # If user wants to update email or password, current password is required
    if new_email or new_password:
        if not current_password:
            flash("Current password is required to update email or password.", "danger")
            return redirect(url_for('profile'))

        # Verify current password
        if not check_password_hash(current_user.password, current_password):
            flash("Current password is incorrect.", "danger")
            return redirect(url_for('profile'))

    # Update email if provided
    if new_email:
        current_user.email = new_email

    # Update password if provided
    if new_password:
        current_user.password = generate_password_hash(new_password, method='pbkdf2:sha256')

    db.session.commit()
    flash("Profile updated successfully!", "success")
    return redirect(url_for('profile'))




# --------------------
# Admin panel routes
# --------------------
@app.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash("Access denied: Admins only.", "danger")
        return redirect(url_for('home'))

    return render_template('admin.html')

@app.route('/admin/manage_users')
@login_required
def manage_users():
    if not current_user.is_admin:
        flash("Access denied: Admins only.", "danger")
        return redirect(url_for('home'))

    admins = User.query.filter_by(is_admin=True).all()
    users = User.query.filter_by(is_admin=False).all()
    return render_template('manage_users.html', admins=admins, users=users)

@app.route('/make_admin/<int:user_id>', methods=['POST'])
@login_required
def make_admin(user_id):
    # Only allow current admins to promote users
    if not current_user.is_admin:
        flash("You are not authorized to perform this action.", "danger")
        return redirect(url_for('manage_users'))

    user = User.query.get_or_404(user_id)

    if user.is_admin:
        flash(f"{user.username} is already an admin.", "info")
    else:
        user.is_admin = True
        db.session.commit()
        flash(f"{user.username} is now an admin!", "success")

    return redirect(url_for('manage_users'))


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        flash("Access denied.", "danger")
        return redirect(url_for('home'))

    user = User.query.get_or_404(user_id)
    if user.is_admin:
        flash("Cannot delete another admin.", "danger")
        return redirect(url_for('manage_users'))

    db.session.delete(user)
    db.session.commit()
    flash(f"User '{user.username}' deleted.", "success")
    return redirect(url_for('manage_users'))

@app.route('/admin/messages', methods=['GET', 'POST'])
@login_required
def admin_messages():
    if not current_user.is_admin:
        flash("Access denied: Admins only.", "danger")
        return redirect(url_for('home'))

    messages = db.session.query(Message, User.username).join(User, Message.user_id == User.id).all()
    return render_template('admin_messages.html', messages=messages)

@app.route('/admin/messages/mark_read/<int:msg_id>', methods=['POST'])
@login_required
def mark_message_read(msg_id):
    if not current_user.is_admin:
        flash("Access denied: Admins only.", "danger")
        return redirect(url_for('home'))
    msg = Message.query.get_or_404(msg_id)
    msg.read = True
    db.session.commit()
    flash("Message marked as read.", "success")
    return redirect(url_for('admin_messages'))

@app.route('/admin/messages/mark_unread/<int:msg_id>', methods=['POST'])
@login_required
def mark_message_unread(msg_id):
    if not current_user.is_admin:
        flash("Access denied: Admins only.", "danger")
        return redirect(url_for('home'))
    msg = Message.query.get_or_404(msg_id)
    msg.read = False
    db.session.commit()
    flash("Message marked as unread.", "success")
    return redirect(url_for('admin_messages'))

@app.route('/admin/messages/delete/<int:msg_id>', methods=['POST'])
@login_required
def delete_message(msg_id):
    if not current_user.is_admin:
        flash("Access denied: Admins only.", "danger")
        return redirect(url_for('home'))
    msg = Message.query.get_or_404(msg_id)
    db.session.delete(msg)
    db.session.commit()
    flash("Message deleted successfully.", "success")
    return redirect(url_for('admin_messages'))

@app.route('/admin/recommendations', methods=['GET', 'POST'])
@login_required
def admin_recommendations():
    if not current_user.is_admin:
        flash("Access denied: Admins only.", "danger")
        return redirect(url_for('home'))

    if request.method == 'POST':
        team1 = request.form.get('team1')
        team2 = request.form.get('team2')
        match_format = request.form.get('format')
        venue = request.form.get('venue')
        match_date_str = request.form.get('match_date')

        # Convert string to Python date
        match_date = datetime.strptime(match_date_str, "%Y-%m-%d").date() if match_date_str else None

        new_rec = Recommendation(
            team1=team1,
            team2=team2,
            match_format=match_format,
            venue=venue,
            match_date=match_date
        )
        db.session.add(new_rec)
        db.session.commit()

        return redirect(url_for('admin_recommendations'))

    recs = Recommendation.query.all()
    teams = list(team_encoder.classes_)
    venues = list(venue_encoder.classes_)

    return render_template('admin_recommendations.html', recs=recs, teams=teams, venues=venues)





@app.route('/admin/recommendations/delete/<int:rec_id>', methods=['POST'])
@login_required
def delete_recommendation(rec_id):
    if not current_user.is_admin:
        flash("Access denied: Admins only.", "danger")
        return redirect(url_for('admin_recommendations'))
    
    rec = Recommendation.query.get_or_404(rec_id)
    db.session.delete(rec)
    db.session.commit()
    flash("Recommendation deleted successfully!", "success")
    return redirect(url_for('admin_recommendations'))

@app.route('/admin/about', methods=['GET', 'POST'])
@login_required
def admin_about():
    if not current_user.is_admin:
        flash("Access denied: Admins only.", "danger")
        return redirect(url_for('home'))
    
    about = AboutUs.query.first()
    if request.method == 'POST':
        new_content = request.form.get('content')
        if about:
            about.content = new_content
        else:
            about = AboutUs(content=new_content)
            db.session.add(about)
        db.session.commit()
        flash("About Us updated successfully!", "success")
        return redirect(url_for('admin_about'))
    return render_template('admin_about.html', about=about)

# --------------------
# Prediction API
# --------------------
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        format_ = data['format']
        team1 = data['team1']
        team2 = data['team2']
        venue = data['venue']
        toss_winner = data['toss_winner']

        if format_ not in ["ODI", "T20"]:
            return jsonify({'error': 'Invalid format selected'})

        team1_enc = team_encoder.transform([team1])[0]
        team2_enc = team_encoder.transform([team2])[0]
        venue_enc = venue_encoder.transform([venue])[0]
        toss_winner_enc = team_encoder.transform([toss_winner])[0]
        format_enc = 0 if format_ == "ODI" else 1

        X_input = np.array([[team1_enc, team2_enc, venue_enc, toss_winner_enc, format_enc]])
        probs = model.predict_proba(X_input)[0]
        class_labels = model.classes_
        decoded_labels = team_encoder.inverse_transform(class_labels)
        prob_dict = {decoded_labels[i]: float(probs[i]) for i in range(len(decoded_labels))}

        filtered_probs = {team1: prob_dict.get(team1, 0), team2: prob_dict.get(team2, 0)}
        total = filtered_probs[team1] + filtered_probs[team2]
        if total > 0:
            normalized_probs = {team: round(prob / total * 100, 2) for team, prob in filtered_probs.items()}
        else:
            normalized_probs = {team1: 50.0, team2: 50.0}

        winner = max(normalized_probs, key=normalized_probs.get)

        new_prediction = Prediction(
            user_id=current_user.id,
            match_format=format_,
            team1=team1,
            team2=team2,
            venue=venue,
            toss_winner=toss_winner,
            predicted_winner=winner,
            probabilities=json.dumps(normalized_probs)
        )
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({'winner': winner, 'probabilities': normalized_probs})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
