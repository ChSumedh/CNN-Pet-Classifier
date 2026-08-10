from flask import Flask, render_template, session, flash, request, redirect, url_for
import os
import base64
import io
import shutil

import numpy as np
from PIL import Image
from models import db, User, Feedback
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__, template_folder='templates', static_folder='static')

db_url = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.secret_key = os.environ.get('SECRET_KEY', 'temporary-key')

db.init_app(app)

with app.app_context():
    db.create_all()

with app.app_context():
    username = 'admin'
    password = 'changeme123' 
    existing = User.query.filter_by(name=username).first()

    if existing:
        print(f'User "{username}" already exists.')
    else:
        admin_user = User(name=username, isAdmin=True)
        admin_user.set_password(password)

        db.session.add(admin_user)
        db.session.commit()

        print(f'Admin user "{username}" created successfully.')

KERAS_MODEL_PATH = 'Oxford-IIIT.keras'
TFLITE_MODEL_PATH = 'cat_dog_model.tflite' 


def load_interpreter():
    global interpreter, input_details, output_details
    interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def unload_interpreter():
    global interpreter, input_details, output_details
    interpreter = None
    input_details = None
    output_details = None

load_interpreter()

classes = [
    'Abyssinian',
    'Bengal',
    'Birman',
    'Bombay',
    'British_Shorthair',
    'Egyptian_Mau',
    'Maine_Coon',
    'Persian',
    'Ragdoll',
    'Russian_Blue',
    'Siamese',
    'Sphynx',
    'american_bulldog',
    'american_pit_bull_terrier',
    'basset_hound',
    'beagle',
    'boxer',
    'chihuahua',
    'english_cocker_spaniel',
    'english_setter',
    'german_shorthaired',
    'great_pyrenees',
    'havanese',
    'japanese_chin',
    'keeshond',
    'leonberger',
    'miniature_pinscher',
    'newfoundland',
    'pomeranian',
    'pug',
    'saint_bernard',
    'samoyed',
    'scottish_terrier',
    'shiba_inu',
    'staffordshire_bull_terrier',
    'wheaten_terrier',
    'yorkshire_terrier'
]


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/register/verification', methods=['POST'])
def verifyRegistration():

    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    password2 = request.form.get('password2', '')

    if password != password2:
        flash("Both Passwords must be Equal")
        return redirect(url_for('register'))

    if User.query.filter_by(name=username).first() is not None:
        flash("Username already taken")
        return redirect(url_for('register'))

    user = User(name=username)
    user.set_password(password=password)

    db.session.add(user)
    db.session.commit()

    return redirect(url_for('login'))


@app.route('/split', methods=['POST'])
def split():

    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')

    user = User.query.filter_by(name=username).first()

    if user is None or not user.is_password(password):
        flash('Invalid Username or password')
        return redirect(url_for('login'))

    session['user_id'] = user.id

    if user.isAdmin:
        return redirect(url_for('admin'))

    return redirect(url_for('model'))


@app.route('/logout')
def logout():

    session.pop('user_id', None)

    return redirect(url_for('home'))


@app.route('/model')
def model():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('model.html')


def preprocess_image(file, target_size=(224, 224)):
    img = Image.open(io.BytesIO(file)).convert('RGB')
    img = img.resize(target_size)

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route('/prediction', methods=['POST'])
def prediction():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    image = request.files.get('image')

    if image is None or image.filename == '':
        flash('Please select an image')
        return redirect(url_for('model'))

    image_bytes = image.read()

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        flash('Invalid image')
        return redirect(url_for('model'))

    if img.width < 224 or img.height < 224:
        flash('Image size too small')
        return redirect(url_for('model'))

    image_array = preprocess_image(image_bytes)
    
    interpreter.set_tensor(
        input_details[0]['index'],
        image_array
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]['index']
    )

    prediction = int(np.argmax(output[0]))
    feedback = Feedback(
        image=image_bytes,
        predicted_class=classes[prediction],
        user_id=session.get('user_id')
    )

    db.session.add(feedback)
    db.session.commit()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return render_template(
        'prediction.html',
        output=prediction,
        image=image_base64,
        classes=classes,
        feedback_id=feedback.id
    )


@app.route('/correct/<int:feedback_id>')
def correct_prediction(feedback_id):

    feedback = Feedback.query.get_or_404(feedback_id)

    image = base64.b64encode(feedback.image).decode('utf-8')

    return render_template(
        'correct.html',
        image=image,
        prediction=classes.index(feedback.predicted_class),
        classes=classes,
        feedback_id=feedback.id
    )


@app.route('/correct/submit', methods=['POST'])
def submit_correction():

    feedback_id = request.form.get('feedback_id')
    correct_class = request.form.get('correct_class')

    feedback = Feedback.query.get_or_404(feedback_id)

    feedback.correct_class = correct_class

    db.session.commit()

    return redirect(url_for('model'))


def get_current_user():
    user_id = session.get('user_id')

    if user_id is None:
        return None

    return db.session.get(User, user_id)


@app.route('/admin')
def admin():
    user = get_current_user()

    if user is None:
        return redirect(url_for('login'))

    if not user.isAdmin:
        return redirect(url_for('model'))

    feedback_count = Feedback.query.filter(
        Feedback.correct_class.isnot(None)
    ).count()

    return render_template('admin.html', feedback_count=feedback_count)


@app.route('/admin/fine_tune', methods=['POST'])
def fine_tune():
    from train import fine_tune_model
    user = get_current_user()

    if user is None or not user.isAdmin:
        return redirect(url_for('login'))

    feedbacks = Feedback.query.filter(
        Feedback.correct_class.isnot(None)
    ).all()

    if not feedbacks:
        flash('No corrected feedback available to train on')
        return redirect(url_for('admin'))

    unload_interpreter()
    success, paths, error = fine_tune_model(KERAS_MODEL_PATH, feedbacks)
    
    if not success:
        load_interpreter()
        flash(f'Fine-tuning failed: {error}')
        return redirect(url_for('admin'))

    new_keras_path, new_tflite_path = paths

    shutil.move(new_keras_path, KERAS_MODEL_PATH)
    shutil.move(new_tflite_path, TFLITE_MODEL_PATH)

    load_interpreter()
    feedback_ids = [fb.id for fb in feedbacks]
    Feedback.query.filter(Feedback.id.in_(feedback_ids)).delete(synchronize_session=False)
    db.session.commit()

    flash(f'Model fine-tuned successfully on {len(feedbacks)} feedback entries')
    return redirect(url_for('admin'))


@app.route('/admin/promote', methods=['POST'])
def promote_user():
    user = get_current_user()

    if user is None or not user.isAdmin:
        return redirect(url_for('login'))

    username = request.form.get('username', '').strip()

    target = User.query.filter_by(name=username).first()

    if target is None:
        flash('User not found')
        return redirect(url_for('admin'))

    if target.isAdmin:
        flash(f'{username} is already an admin')
        return redirect(url_for('admin'))

    target.isAdmin = True
    db.session.commit()

    flash(f'{username} is now an admin')

    return redirect(url_for('admin'))


if __name__ == "__main__":
    app.run()
