from flask import Flask
from threading import Thread

app = Flask(__name__)

@app.route('/')
def home():
    return 'JARVIS online'

def run_server():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run_server)
    t.daemon = True
    t.start()
