import cv2
import numpy as np
import base64
from dotenv import load_dotenv
import os


def prediction(encoded_img):
    load_dotenv()

    # Load Yolo
    model_dir = os.environ.get("MODEL_DIR")
    model_weight = "{}/{}".format(model_dir, os.environ.get("MODEL_WEIGHT"))
    model_config = "{}/{}".format(model_dir, os.environ.get("MODEL_CONFIG"))
    net = cv2.dnn.readNet(model_weight, model_config)

    # Name custom object
    classes = ["banana","grapes","kiwi","lemon","lettuce","orange","paprika","pineapple","pomegranate","sweet potato"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    # Load Image
    decoded_str = base64.b64decode(encoded_img)
    np_data = np.fromstring(decoded_str, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

    _, buffer = cv2.imencode(".jpg", img)
    img_encoded = base64.b64encode(buffer)
    
    return {
        'img': img_encoded.decode("utf-8"),
        'label': label
    }

