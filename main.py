from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

class_mapping = {
    0: 'Vladimir hu-man'
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

ret = True

while ret:
    ret, frame = cap.read()

    results = model.track(frame, persist=True)
    for result in results:
        for cls_id, custom_label in class_mapping.items():
            if cls_id in result.names:  # check if the class id is in the results
                result.names[cls_id] = custom_label  # replace the class name with the custom label

    frame_ = results[0].plot()

    cv2.imshow('frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
