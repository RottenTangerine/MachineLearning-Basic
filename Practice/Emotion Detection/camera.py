import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import CNN

idx_to_class = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth').to(device)


def predict_pic(pic, model, device, idx_to_class):
    pic = pic[200:400, 220:420]
    pic = cv2.resize(pic, (48, 48))
    pic = np.reshape(pic, (1, 1, 48, 48))
    tensor = torch.from_numpy(pic).to(device).float()

    output = model(tensor)
    _, predicted = torch.max(output, 1)

    return idx_to_class[predicted]


camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera")
    exit()

frame_counter = 0
while True:
    ret, frame = camera.read()
    frame_counter += 1
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if frame_counter % 60 == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.imshow(frame)
        currentAxis = plt.gca()
        rect = patches.Rectangle((220, 200), 200, 200, linewidth=3, edgecolor='r', facecolor='none')
        currentAxis.add_patch(rect)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.resize(frame, (48, 48))
        output = predict_pic(frame, model, device, idx_to_class)
        print(output)
        plt.title(output)
        plt.show()
