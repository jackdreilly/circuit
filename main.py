import json
import parser

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit, send
import os

import cpu_simulator

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/parse", methods=["POST"])
def parse():
    return jsonify(parser.parse(request.json["program"]))


disconnected = set()
paused = set()
@socketio.on('disconnect', namespace='/test')
def handle_disconnect():
    disconnected.add(request.sid)
@socketio.on('pause', namespace='/test')
def handle_pause():
    paused.add(request.sid)
@socketio.on('resume', namespace='/test')
def handle_resume():
    paused.discard(request.sid)

@socketio.on('json', namespace='/test')
def handle_json(*args, **kwargs):
    if not args:
        return
    payload = json.loads(args[0])
    program = payload['program']
    beefy = bool(payload.get('beefy', 0))
    print(program)
    pack= []
    for simulation in cpu_simulator.simulation(program, frequency=1, beefy=beefy):
        pack.append(simulation)
        if not ((len(pack) + 1) % 100):
            send(pack, json=True)
            pack = []
        socketio.sleep(0.0001)
        while request.sid in paused:
            socketio.sleep(1)
        if request.sid in disconnected:
            print("DISCONNECTED!")
            disconnected.discard(request.sid)
            paused.discard(request.sid)
            return


if __name__ == "__main__":
    socketio.run(debug=True, host='0.0.0.0',
                 port=int(os.environ.get('PORT', 8080)))
