from flask import Flask, request, jsonify
import json
import parser
from pprint import pprint
import cpu_simulator
from flask_socketio import SocketIO
from flask_socketio import send, emit
from flask_cors import CORS, cross_origin

from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*")
thread = None
thread_lock = Lock()


def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count},
                      namespace='/test')


@app.route("/parse", methods=["POST"])
def hello():
    return jsonify(parser.parse(request.json["program"]))


disconnected = set()
@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    disconnected.add(request.sid)

@socketio.on('connect', namespace='/test')
def test_connect():
    emit('json', )

@socketio.on('json', namespace='/test')
def handle_json(*args, **kwargs):
    if not args:
        return
    program = json.loads(args[0])['program']
    print(program)
    for simulation in cpu_simulator.simulation(program, frequency=5):
        send(simulation, json=True)
        socketio.sleep(0.0001)
        if request.sid in disconnected:
            print("DISCONNECTED!")
            return

if __name__ == '__main__':
    socketio.run(app, debug=True)
