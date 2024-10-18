import numpy as np
from PIL import Image, ImageDraw
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.25
model.iou = 0.45
model.max_det = 100

def load_image_pixels(filename, shape):
    image = Image.open(filename)
    image = image.resize(shape)
    image = np.asarray(image)
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, 0)
    return image

image_path = 'planta.png'
input_w, input_h = 416, 416
image = load_image_pixels(image_path, (input_w, input_h))

results = model(image_path)

def draw_boxes(results):
    results.show()

draw_boxes(results)