import cv2
import numpy as np
import json

# Load YOLO model (adjust paths as needed for your model files)
model = cv2.dnn.readNet("yolov5s.weights", "yolov5s.cfg")  # Example for YOLOv5s
layer_names = model.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

# Load restricted zones from a JSON file or define manually
def load_zones(config_file="zones_config.json"):
    with open(config_file, "r") as f:
        zones = json.load(f)["zones"]
    return zones

no_human_zones = load_zones()  # Or you can define manually as in previous examples

# Draw no-human zones on the frame
def draw_no_human_zones(frame, zones):
    for zone in zones:
        cv2.rectangle(frame, tuple(zone["coords"][0]), tuple(zone["coords"][1]), (0, 0, 255), 2)
        cv2.putText(frame, f"Zone {zone['zone_id']}", (zone["coords"][0][0], zone["coords"][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Detect humans in the frame using YOLO model
def detect_humans(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    detections = model.forward(output_layers)

    human_boxes = []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Human class ID for YOLO COCO dataset (0 for person)
            if class_id == 0 and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                human_boxes.append((x, y, w, h))
    return human_boxes

# Check if any detected human is in a restricted zone
def check_no_human_zones(frame, human_boxes, zones):
    for (x, y, w, h) in human_boxes:
        human_center = (x + w // 2, y + h // 2)
        
        for zone in zones:
            x1, y1 = zone["coords"][0]
            x2, y2 = zone["coords"][1]
            if x1 <= human_center[0] <= x2 and y1 <= human_center[1] <= y2:
                cv2.putText(frame, "ALERT: Human in Restricted Zone!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Main function for video capture and detection
def main():
    # Select video source: 0 for webcam or path to video file
    cap = cv2.VideoCapture(0)  # Change to "path/to/video.mp4" for a local video file

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        human_boxes = detect_humans(frame)
        draw_no_human_zones(frame, no_human_zones)
        check_no_human_zones(frame, human_boxes, no_human_zones)

        # Display the output frame
        cv2.imshow("PPE Detection - No Human Zones", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
