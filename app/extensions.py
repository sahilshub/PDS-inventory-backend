from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_apscheduler import APScheduler
from flask_socketio import SocketIO


db = SQLAlchemy()
mail = Mail()
scheduler = APScheduler()
socketio = SocketIO(cors_allowed_origins="*")


