import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.linear = nn.Linear(12 * 12 * 16, class_num)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.reshape(out.size(0), -1)
        return self.linear(out)

idx_to_class = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.ckpt').to(device)

def predict_pic(pic, model, device, idx_to_class):
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

    if frame_counter%60 == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.resize(frame, (48, 48))
        output = predict_pic(frame, model, device, idx_to_class)
        print(output)
        plt.title(output)
        plt.show()