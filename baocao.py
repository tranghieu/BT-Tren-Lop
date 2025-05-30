from flask import Flask, Response
import cv2
import os
import time
import csv
from datetime import datetime
import smtplib
from email.message import EmailMessage
from ultralytics import YOLO
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

VIDEO_SOURCE = r"C:\Users\tranghieu\Downloads\smarts city\hi2.mp4"
car_model = YOLO(r'C:\Users\tranghieu\Downloads\smarts city\yolov8n.pt')
tl_model  = YOLO(r'C:\Users\tranghieu\Downloads\smarts city\best_traffic_nano_yolo.pt')
os.makedirs('vi_pham', exist_ok=True)

line_pts = [(164, 150), (382, 176)]

track_history = {}
frame_count = 0

csv_file = 'vi_pham_log.csv'

def save_violation_log(tid, violation_time, violation_type, position, image_path):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Thời gian', 'ID xe', 'Loại vi phạm', 'Tọa độ', 'Ảnh vi phạm'])
        writer.writerow([violation_time, tid, violation_type, position, image_path])

def send_email_with_attachment():
    to_email = "tranghieu0626@gmail.com"
    subject = "Báo cáo vi phạm giao thông tự động"
    body = "Đây là báo cáo vi phạm giao thông được gửi tự động."

    if not os.path.exists(csv_file):
        print("File báo cáo không tồn tại, không gửi email.")
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = "tranghieu0626@gmail.com"  # Thay bằng email bạn
    msg['To'] = to_email
    msg.set_content(body)

    with open(csv_file, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(csv_file)

    msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login("tranghieu0626@gmail.com", "your_app_password")  # Thay mật khẩu ứng dụng
            smtp.send_message(msg)
        print("Email đã gửi thành công!")
    except Exception as e:
        print("Lỗi gửi email:", e)

def side_of_line(pt, p1, p2):
    x1,y1 = p1; x2,y2 = p2
    return (x2-x1)*(pt[1]-y1) - (y2-y1)*(pt[0]-x1)

def gen_frames():
    global frame_count, track_history
    for result in car_model.track(
        source=VIDEO_SOURCE,
        conf=0.5,
        iou=0.5,
        classes=[2],
        persist=True,
        stream=True
    ):
        frame_count += 1
        frame = result.orig_img.copy()

        # Vẽ vạch
        cv2.line(frame, line_pts[0], line_pts[1], (0,255,0), 2)

        # Detect đèn giao thông
        tl_res = tl_model(frame, conf=0.3)[0]
        tl_state = None
        for tl_box in tl_res.boxes:
            x1_l,y1_l,x2_l,y2_l = tl_box.xyxy.cpu().numpy().astype(int)[0]
            cls_id = int(tl_box.cls.cpu().item())
            conf_l = float(tl_box.conf.cpu().item())
            name   = tl_model.model.names[cls_id]
            color  = (0,255,0) if name=='green' else (0,0,255) if name=='red' else (255,255,0)
            cv2.rectangle(frame, (x1_l,y1_l),(x2_l,y2_l), color, 2)
            cv2.putText(frame, f"{name}:{conf_l:.2f}", (x1_l,y1_l-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if tl_state is None or conf_l > tl_state[1]:
                tl_state = (name, conf_l)

        light_label = tl_state[0] if tl_state else "no-light"
        cv2.putText(frame, f"Light: {light_label}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0) if light_label=='green' else (0,0,255), 2)

        # Xử lý từng xe
        for box in result.boxes:
            tid = int(box.id.cpu().item())
            x1,y1,x2,y2 = box.xyxy.cpu().numpy().astype(int)[0]
            cx = (x1 + x2)//2
            cy = y2

            if tid not in track_history:
                track_history[tid] = {
                    'pt': (cx,cy),
                    'crossed': False,
                    'violation': False,
                    'violation_time': None
                }
            rec = track_history[tid]

            box_color = (0,0,255) if rec['violation'] else (255,0,0)

            if not rec['crossed']:
                s_prev = side_of_line(rec['pt'], line_pts[0], line_pts[1])
                s_curr = side_of_line((cx,cy), line_pts[0], line_pts[1])
                if s_prev * s_curr < 0:
                    if light_label == 'red':
                        rec['violation'] = True
                        rec['violation_time'] = time.time()
                        # Lưu ảnh vi phạm
                        crop = result.orig_img[y1:y2, x1:x2]
                        fname = os.path.join('vi_pham', f"car_{tid}_{frame_count}.jpg")
                        cv2.imwrite(fname, crop)
                        print(f"[VI PHAM] Saved {fname}")

                        violation_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_violation_log(tid, violation_time_str, 'Vượt đèn đỏ', f"({cx},{cy})", fname)
                    rec['crossed'] = True

            rec['pt'] = (cx,cy)

            cv2.rectangle(frame, (x1,y1),(x2,y2), box_color, 2)
            cv2.putText(frame, f"ID:{tid}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.circle(frame, (cx,cy), 4, box_color, -1)

            if rec['violation'] and rec['violation_time'] is not None:
                if time.time() - rec['violation_time'] <= 1.0:
                    cv2.putText(frame, "VI PHAM", (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        violation_count = sum(1 for v in track_history.values() if v['violation'])
        cv2.putText(frame, f"Violations: {violation_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Mã hóa frame và yield
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <html><head><title>Giám sát vi phạm</title></head><body>
    <h1>Giám sát vi phạm giao thông</h1>
    <img src="/video_feed" width="960" />
    </body></html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=send_email_with_attachment, trigger="interval", seconds=60)  # Gửi mail mỗi 60 giây
    scheduler.start()
    try:
        app.run(debug=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
