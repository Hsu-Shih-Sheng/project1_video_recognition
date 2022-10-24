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

#%% train video to frame (only first time)

try:
    os.mkdir('./train_data')
    
    path = os.listdir('./data/train/')

    for i in tqdm(range(len(path))):

        train = os.listdir('./data/train/'+ path[i] + '/')

        for videoFile in train:
            count = 0
 
            VIDEO_PATH = './data/train/'+ path[i] + '/' + videoFile 

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

                        save_path ='./train_data/'+ path[i] + '_' + videoFile +"_frame%d.jpg" % count
                        cv2.imwrite(save_path, frame)
                    count += 1
                video.release()

            def main():

                extract_frames(VIDEO_PATH)

            if __name__ == '__main__':
                main()
                
except FileExistsError:
   pass



#%% read train data

print('Reading train data')
print('...')

train_x, train_y = readfile('./train_data/', True)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)

print('...')
print('Reading complete')

#%% transform & dataloader
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),

])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),

])

batch_size = 256

train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
#%% model & hyperparameter
lr = 0.001

model = new_resnet18()
     
model = model.to(device)

loss = nn.CrossEntropyLoss()
 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5, min_lr=0.0000001)

num_epoch = 50


#%% saving best model
try:
   os.mkdir('./save_model')
         
except FileExistsError:
   pass

#%% training
print("Training")
print('...')

the_last_loss = 10000000000
patience = 4
trigger_times = 0
final_val_acc = 0

for epoch in range(num_epoch):

    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        train_pred = model(data[0].to(device))
        batch_loss = loss(train_pred, data[1].to(device))
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()


    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))
            

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

    scheduler.step(val_loss/ val_set.__len__())
    
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.4f Loss: %3.4f | Val Acc: %3.4f loss: %3.4f' % \
            (epoch +1, num_epoch, time.time() - epoch_start_time, \
            train_acc/ train_set.__len__(), train_loss/ train_set.__len__(), \
            val_acc/ val_set.__len__(), val_loss/ val_set.__len__()))

    if final_val_acc < val_acc/ val_set.__len__():
            final_val_acc = val_acc/ val_set.__len__()
            torch.save(model.state_dict(), './save_model/resnet18_best_model_state_dict.pt')

    torch.save(model.state_dict(), './save_model/resnet18_' + str(epoch+1) + '_model_state_dict.pt')

print('...')
print('Training Complete')


