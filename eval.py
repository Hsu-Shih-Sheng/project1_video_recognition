#%% import
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler
import pandas as pd
import os
from tqdm import tqdm_notebook as tqdm
import cv2
import time
from function import readfile, ImgDataset, new_resnet18

#%% device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% test video to frame (only first time)


try:
   os.mkdir('./test_data')

   test = os.listdir('./data/test/')

   for videoFile in tqdm(test):
      count = 0


      VIDEO_PATH = './data/test/' + videoFile


      def extract_frames(video_path):

         video = cv2.VideoCapture()
         if not video.open(video_path):
               print("can not open the video")
               exit(1)
         cap = int(video.get(cv2.CAP_PROP_FRAME_COUNT)/15)
         if cap == 0:
            cap = 1         
         count = 1
         while True:
               _, frame = video.read()

               if frame is None:
                  break

               if count % cap == 0:

                  save_path ='./test_data/'+ videoFile +"_frame%d.jpg" % count
                  cv2.imwrite(save_path, frame)
               count += 1
         video.release()

      def main():

         extract_frames(VIDEO_PATH)

      if __name__ == '__main__':
         main()
         
except FileExistsError:
   pass

#%% read test data
print('Reading test data')
print('...')

test_x = readfile('./test_data/', False)

print('...')
print('Reading complete')

#%% transform
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),

])

#%% testing & result csv
print("Testing")
print('...')
batch_size = 256
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = new_resnet18()
model = model.to(device)
###
FILE = './save_model/resnet18_best_model_state_dict.pt'
model.load_state_dict(torch.load(FILE))
###
model.eval()
prediction2 = []
with torch.no_grad():
    for i, data in tqdm(enumerate(test_loader)):
        test_pred = model(data.to(device))

        test_label2 = test_pred.cpu().data.numpy()


        for x in test_label2:
            prediction2.append(x)


result_path = os.listdir('./test_data/')
output_name = []
for i, file in enumerate(result_path):
    output_name.append(file.split("_")[0])

result = pd.DataFrame()
result['name'] = output_name
result['label_all'] = prediction2
name = result['name'].unique()

with open('prediction.csv', 'w') as f:
    f.write('name,label\n')
    for i in tqdm(name):
        y = np.argmax(result[result['name'] == i]['label_all'].sum())
        f.write('{},{}\n' .format(i, y))

print('...')
print('Testing Complete')