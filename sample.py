from flask import Flask, Response, render_template
from imutils.video.pivideostream import PiVideoStream
import cv2
import time


app = Flask(__name__)
camera = PiVideoStream(resolution=(400, 304), framerate=10).start()
time.sleep(2)


def gen(camera):
    while True:
        frame = camera.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=False,
        threaded=True,
        use_reloader=False
    )
