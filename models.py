from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    isAdmin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def is_password(self, password):
        return check_password_hash(self.password, password)


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    image = db.Column(db.LargeBinary, nullable=False)

    predicted_class = db.Column(
        db.String(100),
        nullable=False
    )

    correct_class = db.Column(
        db.String(100),
        nullable=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )