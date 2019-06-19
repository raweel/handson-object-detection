from flask import Flask, Response, render_template
from imutils.video.pivideostream import PiVideoStream
import cv2
import time
import numpy as np

app = Flask(__name__)
camera = PiVideoStream(resolution=(400, 304), framerate=10).start()
time.sleep(2)

net = cv2.dnn.readNetFromCaffe('/home/pi/models/MobileNetSSD_deploy.prototxt',
        '/home/pi/models/MobileNetSSD_deploy.caffemodel')

def detect(frame):
    frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(
        image=frame,
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5
    )

net.setInput(blob)
out = net.forward()

boxes = out[0,0,:,3:7] * np.array([300, 300, 300, 300])
classes = out[0,0,:,1]
confidences = out[0,0,:,2]

    for i, box in enumerate(boxes):
        confidence = confidences[i]
        if confidence < 0.2:
            continue
        idx = int(classes[i])
        if idx != 15:
            continue

        (startX, startY, endX, endY) = box.astype('int')
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        label = '{}: {:.2f}%'.format('Person', confidence * 100)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame



def gen(camera):
    while True:
        frame = camera.read()
        processed_frame = detect(frame.copy())
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
