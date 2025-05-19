import os
import random
from flask import Flask
from flask_cors import CORS
from app.routes import main
from app.extensions import *
from dotenv import load_dotenv
from google import generativeai as genai


def create_app():
    app = Flask(__name__)
    load_dotenv()
    
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY'),
        JWT_SECRET_KEY=os.getenv('JWT_SECRET_KEY'),
        MAIL_SERVER='smtp.googlemail.com',
        MAIL_PORT=587,
        MAIL_USERNAME=os.getenv('MAIL_USERNAME'),
        MAIL_PASSWORD=os.getenv('MAIL_PASSWORD'),
        MAIL_USE_TLS=True,
        MAIL_USE_SSL=False,
        SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URI'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SCHEDULER_API_ENABLED=True
    )

    genai.configure(api_key=os.getenv(f"GOOGLE_API_KEY{random.choice(['1','2','3'])}"))

    db.init_app(app)
    mail.init_app(app)
    scheduler.init_app(app) 
    socketio.init_app(app)
    CORS(app)

    from app.routes import main
    app.register_blueprint(main)

    with app.app_context():
        db.create_all()

    return app

