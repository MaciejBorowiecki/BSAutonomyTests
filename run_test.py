import cv2
import json
import math
import os
import time
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from datetime import datetime

# --- CONFIGURATION ---
MODEL_PATH = 'models/bs11n.pt'      
CONFIG_PATH = 'object_config.json'  
FOCAL_LENGTH = 600.0                
SAVE_FRAMES = True                  
SHOW_PREVIEW = True                 

INFERENCE_SIZE = 320  
CONF_THRESHOLD = 0.5  

def setup_performance():
    """Applies performance tweaks found in the repo."""ow
    LOGGER.setLevel("ERROR")

def load_object_config(path):
    if not os.path.exists(path):
        print(f"ERROR: Configuration file {path} not found!")
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
        return data.get('objects', {})

def get_next_run_dir(base_dir):
    base_path = Path(base_dir)
    i = 1
    while True:
        run_dir = base_path / f"run{i}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        i += 1

def calculate_angle(center_x, image_width, focal_length):
    offset = center_x - (image_width / 2)
    angle_rad = math.atan(offset / focal_length)
    return math.degrees(angle_rad)

def distance_to_camera(known_width, focal_length, per_width):
    if per_width == 0: return 0
    return (known_width * focal_length) / per_width

def main():
    setup_performance() 
    
    object_config = load_object_config(CONFIG_PATH)
    print(f"Loaded config. Performance Mode: imgsz={INFERENCE_SIZE}, conf={CONF_THRESHOLD}")
    
    print(f"Loading model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    model_name = Path(MODEL_PATH).stem
    run_dir = get_next_run_dir(f"runs/{model_name}")
    print(f"Recording started. Results saved in: {run_dir}")

    print("Press 'q' to quit.")

    frame_count_saved = 0
    total_frames_processed = 0
    start_time = time.time()
    warned_classes = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        total_frames_processed += 1
        annotated_frame = frame.copy()
        
        results = model(frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                # Coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w_px = x2 - x1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Display logic
                dist_str = ""
                angle_str = ""
                color = (150, 150, 150)

                if class_name in object_config:
                    obj_data = object_config[class_name]
                    real_width = obj_data['real_width'] 

                    dist_m = distance_to_camera(real_width, FOCAL_LENGTH, w_px)
                    angle_deg = calculate_angle(center_x, width, FOCAL_LENGTH)

                    dist_str = f"| D: {dist_m:.2f}m"
                    angle_str = f"| A: {angle_deg:.1f}"
                    color = (0, 255, 0) 

                    cv2.line(annotated_frame, (int(width/2), int(height)), (int(center_x), int(center_y)), (0, 255, 255), 1)
                else:
                    if class_name not in warned_classes:
                        # Log warning only once per class
                        print(f"[WARN] No config for '{class_name}'")
                        warned_classes.add(class_name)

                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                label = f"{class_name} {dist_str} {angle_str}"
                text_y = max(int(y1) - 10, 20)
                cv2.putText(annotated_frame, label, (int(x1), text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Crosshair
        cv2.line(annotated_frame, (int(width/2), int(height/2)-10), (int(width/2), int(height/2)+10), (0,0,255), 1)
        cv2.line(annotated_frame, (int(width/2)-10, int(height/2)), (int(width/2)+10, int(height/2)), (0,0,255), 1)

        # Save Frame
        if SAVE_FRAMES:
            filename = run_dir / f"frame_{frame_count_saved:06d}.jpg"
            cv2.imwrite(str(filename), annotated_frame)
            frame_count_saved += 1

        # Preview
        if SHOW_PREVIEW:
            cv2.imshow('BS11N Optimized', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Stats
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_fps = total_frames_processed / elapsed_time if elapsed_time > 0 else 0

    cap.release()
    cv2.destroyAllWindows()
    
    print("-" * 30)
    print("Optimization Summary:")
    print(f"Inference Size: {INFERENCE_SIZE}px (Repo uses 224/320)")
    print(f"Conf Threshold: {CONF_THRESHOLD}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"AVERAGE FPS: {avg_fps:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    main()