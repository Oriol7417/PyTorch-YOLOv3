import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import time
import os 

from pytorchyolo import models
import detect
from pytorchyolo.utils.utils import load_classes

def save_image(img, path, file_prefix):
    if not os.path.exists(path):
        os.makedirs(path)
        
    file_list = os.listdir(path)
    file_list_log = [file for file in file_list if file.startswith(file_prefix)]
    count = 1 + len(file_list_log)
    file_path = os.path.join(path, file_prefix + '{0}.jpg'.format(count)).replace("\\","/")
    cv2.imwrite(file_path, img)

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def run():
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-c", "--classes", type=str, default="data/coco.names", help="Path to classes label file (.names)")
    args = parser.parse_args()

    # Load the YOLO model
    model = models.load_model(args.model, args.weights)
    classes = load_classes(args.classes)
    
    # Set Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [list(cmap(i)) for i in np.linspace(0, 1, len(classes))]
    colors = (np.array(colors) * 255).tolist()

    # Open video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Draw image
    while(True):
        ret, img = cap.read()

        if ret:
            prevTime = time.time()
            # Runs the YOLO model on the image
            boxes = detect.detect_image(model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            curTime = time.time()

            # Draw box and text
            for x1, y1, x2, y2, conf, cls_pred in boxes:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                text=f"{classes[int(cls_pred)]}: {conf:.2f}"
                color = colors[int(cls_pred)]
                img = cv2.rectangle(img, (x1,y1), (x2,y2), color[:2], 2) 
                draw_text(img, text, 
                          font=cv2.FONT_HERSHEY_PLAIN, 
                          font_scale=2, 
                          pos=(x1, y1), 
                          text_color=(255,255,255), 
                          text_color_bg=color[:2])
                
            # Draw fps            
            sec = curTime - prevTime
            fps = 1/(sec)
            text_fps = 'FPS: ' + f"{fps:.3f}"
            draw_text(img, text_fps, 
                        font=cv2.FONT_HERSHEY_PLAIN, 
                        font_scale=2, 
                        pos=(10, 10), 
                        text_color=(255,255,255))

            # show image
            cv2.imshow('webcam', img)

            # close when pushed esc key
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                save_image(img, 'output', 'capture_')
                        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()