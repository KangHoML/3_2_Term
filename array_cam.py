import cv2
import torch
import torchvision.transforms as transforms
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 150, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)

# Model
class OX_Model_CNN(nn.Module):
    def __init__(self):
        super(OX_Model_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        
        self.fc1 = nn.Linear(32*18*18, 512)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32*18*18)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
    
base_dir = 'C:/Users/Hong/source/repos/TicTacToe/OX_class_model/'
predict_dir = os.path.join(base_dir, 'predict')
class_names = {0:'None', 1:'O', 2:'X'}

model = OX_Model_CNN()
model.load_state_dict(torch.load('OX_class_model.pth'))
model.eval()

while True:
    predict_result = []
    success, img = cap.read()
    
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    
    grid_x_size = frameWidth//3
    grid_y_size = frameHeight//3
    
    for i in range(1,3):
        # 세로선
        cv2.line(img, (grid_x_size * i, 0), (grid_x_size *i, frameHeight), (255,255,255), 2)
        # 가로선
        cv2.line(img, (0, grid_y_size * i), (frameWidth, grid_y_size * i), (255,255,255), 2)
    
    cv2.imshow("Result", img)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        for i in range(3):
            for j in range(3):
                start_x, start_y = grid_x_size*j, grid_y_size*i
                end_x, end_y = start_x + grid_x_size, start_y + grid_y_size
                
                grid_img = img[start_y:end_y, start_x:end_x]
                
                saved_img_path = os.path.join(predict_dir, f'target_img_{i}_{j}.png')
                cv2.imwrite(saved_img_path, grid_img)
                print("Screen shot!")
            
                saved_img = Image.open(saved_img_path).convert('L')
                saved_img = transform(saved_img)
                saved_img = saved_img.unsqueeze(0)
            
                with torch.no_grad():
                    output = model(saved_img)
                    _, predicted = torch.max(output, 1)
                    predicted_label = predicted.item()
                
                predict_result[i][j] = predicted_label
                print(predict_result)
        print()