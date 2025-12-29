# https://github.com/MaciejBorowiecki/BlueSpark-autonomy/blob/main/bluespark_distance_estimation/triangle_similarity/compute_focal_length.py
import cv2
from ultralytics import YOLO

# choose appropriate model
model = YOLO("yolo11s.pt").to("cpu")

KNOWN_WIDTH = 7.5   # real widht of the object
DISTANCE_CM = 30.0  # distance between object and the camera in cm
THRESH = 0.5        # threshold for yolo model

# Turn on Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# User instructions
print("Set an object with width KNOWN_WIDTH to the distance DISTANCE_CM and press 'c' to measure.")
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Yolo detection on every frame
    results = model(frame, imgsz=224, verbose=False)
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            w = x2 - x1
            
            # Show object width in pixels
            cv2.putText(frame, f"w_px: {w}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # frame preview
    cv2.imshow("calibration", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # compute focal length when 'c' is pressed 
        # find first bbox in last frame
        found = False
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) >= THRESH:
                    # get last measured width w_px from detection
                    w_px = int(box.xyxy[0][2] - box.xyxy[0][0])
                    
                    # calculate focal_length and print user instructions
                    focal_length = (w_px * DISTANCE_CM) / KNOWN_WIDTH
                    print(f"Measured pixel width: {w_px} px")
                    print(f"Computed focal_length: {focal_length:.2f}")
                    print("Paste this value to distance_utils.py -> FOCAL_LENGTH")
                    found = True
                    break
            if found:
                break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()