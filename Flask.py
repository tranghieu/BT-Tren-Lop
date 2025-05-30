from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import time
import os

app = Flask(__name__)

VIDEO_SOURCE = 0  # hoặc đường dẫn file video
car_model = YOLO(r'C:\Users\tranghieu\Downloads\smarts city\yolov8n.pt')
tl_model = YOLO(r'C:\Users\tranghieu\Downloads\smarts city\best_traffic_nano_yolo.pt')

line_pts = [(164, 150), (382, 176)]
track_history = {}
frame_count = 0

def side_of_line(pt, p1, p2):
    x1,y1 = p1; x2,y2 = p2
    return (x2-x1)*(pt[1]-y1) - (y2-y1)*(pt[0]-x1)

def gen_frames():
    global frame_count, track_history
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count += 1
            # Xử lý frame (phát hiện, vẽ vạch, track vi phạm tương tự code bạn có)
            # Mình demo vẽ vạch và text đơn giản ở đây:

            cv2.line(frame, line_pts[0], line_pts[1], (0,255,0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Mã hóa frame thành JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Trả về dạng multipart HTTP response (MJPEG)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')  # bạn tạo file templates/index.html

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
