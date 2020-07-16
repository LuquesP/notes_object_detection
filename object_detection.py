import os
import numpy as np
import torch
import torchvision
from PIL import Image
import csv 
import pandas as pd  

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from engine import train_one_epoch, evaluate
import utils 
import transforms as T

NUM_CLASSES = 40 

class MusicDataset(object): 
    def __init__(self, root, transforms, dataframe):
        self.root = root 
        self.transforms = transforms
        self.df = dataframe
        self.imgs = dataframe['path_to_image'].values 
        self.class_list = [] 
        self.__get_label_dict__() 

    def __get_label_dict__(self): 
        class_names = self.df['class_name'].unique()  
        self.class_list.append({'Id': 0 , 'Name': 'Background'})
        for i, c in enumerate(class_names): 
            j = i + 1 
            self.class_list.append({'Id': j, 'Name': c})
    
    def __get_label_id__(self, label): 
        label_ids = []
        for l in label: 
            for e in self.class_list:  
                if e['Name'] == l:
                    label_ids.append(e['Id'])
        return label_ids 

    def __get_label_name__(self, label_id): 
        label_ids = []
        for lid in label_id: 
            for e in self.class_list: 
                if e['Id'] == label_id: 
                    label_ids.append(e['Name'])
        return label_ids

    def __get_num_classes__(self): 
        return len(self.class_list) 

    def __getitem__(self, index): 
        image_path = os.path.join(self.root, 'images', self.imgs[index].split('/')[-1]) 
        
        records = self.df[self.df['path_to_image'] == self.imgs[index]]
        # print(records) 
        # image_id = image_path.split("/")[-1]
        image = Image.open(image_path) 

        boxes = records[['top', 'left', 'bottom', 'right']].values  
        labels = records[['class_name']].values
        
        label_ids = self.__get_label_id__(labels)
        labels = torch.as_tensor(label_ids, dtype=torch.int64)
       
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
       
        image_id = torch.as_tensor([index]) 
       
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
       
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target  = {}
        target['boxes'] = boxes 
        target['labels'] = labels 
        target['image_id'] = image_id
        target['area'] = area 
        target['iscrowd'] = iscrowd 

        if self.transforms is not None:
            img, target = self.transforms(image, target)
        
        return img, target  

    def __len__(self): 
        return len(self.imgs) 

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model():

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model 

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    data_dir = os.path.join(os.getcwd(), 'data', 'normalized', 'deepscores') 
    df_train = pd.read_csv(os.path.join(data_dir, 'training.csv'))
    df_validation = pd.read_csv(os.path.join(data_dir, 'validation.csv'))

    dataset_train = MusicDataset(data_dir, get_transform(train=False), df_train)
    dataset_validation = MusicDataset(data_dir, get_transform(train=False), df_validation) 
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1,shuffle=True, num_workers=4, collate_fn=utils.collate_fn )
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=1,shuffle=True, num_workers=4, collate_fn=utils.collate_fn )

    model = get_model() 
    model.to(device) 

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_validation, device=device)

    print("That's it!")

if __name__ == "__main__":
    # data_dir = os.path.join(os.getcwd(), 'data', 'normalized', 'deepscores') 
    # df = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))     
    # data = MusicDataset(data_dir, get_transform(train=False), df)
    # image, target = data.__getitem__(2) 
    # print(image) 
    main() 