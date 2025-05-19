# run.py
from app import create_app
from app.extensions import socketio, scheduler
from geopy.distance import geodesic

app = create_app()

@socketio.on('connect')
def handle_connect():
    print("Client connected")

if __name__ == '__main__':
    with app.app_context():
        if not scheduler.running:
            scheduler.start() 

    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
