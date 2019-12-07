import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import random
from torch import optim
from torch.optim import lr_scheduler
import copy
from Multi_NetWork import Net
import datetime

ROOT_DIR = '../Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = './csv/Multi_train_make_anno_good.csv'
VAL_ANNO = './csv/Multi_val_make_anno_good.csv'
CLASSES = ['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']


class MyDataSet(Dataset):
    def __init__(self, root_dir, anno_file, transform=False):
        self.root_dir = root_dir
        self.anno_file = anno_file
        self.transform = transform
        if (not os.path.isfile(anno_file)):
            print(self.anno_file, ' is not exist!')
        self.file_info = pd.read_csv(self.anno_file, index_col=0)
    def __len__(self):
        return len(self.file_info)
    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if (not os.path.isfile(image_path)):
            print(image_path, ' is not exist!')
        image = Image.open(image_path).convert('RGB')
        label_classes = self.file_info['classes'][idx]
        label_species = self.file_info['species'][idx]
        sample = {'image': image, 'classes': label_classes, 'species': label_species}
        if (self.transform):
            sample['image'] = self.transform(image)
        return sample


train_transform = transforms.Compose([transforms.Resize((500, 500)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
test_transform = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = MyDataSet(root_dir=ROOT_DIR + TRAIN_DIR, anno_file=TRAIN_ANNO, transform=train_transform)
test_dataset = MyDataSet(root_dir=ROOT_DIR + VAL_DIR, anno_file=VAL_ANNO, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=True)

data_loader = {'train': train_loader, 'val': test_loader}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

def visual_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, CLASSES[sample['classes']],SPECIES[sample['species']])
    plt.imshow(transforms.ToPILImage()(sample['image']))
    plt.show()

# visual_dataset()

def train_model(net, optimizer, criterion, scheduler, epochs=50,rate=0.5):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list = {'train': [], 'val': []}
    best_model = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    for epoch in range(epochs):
        print('Epoch:{}/{}'.format(epoch, epochs - 1))
        print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
        for phase in ['train', 'val']:
            if (phase == 'train'):
                net.train()
            else:
                net.eval()
            running_loss = .0
            corrects_classes = 0
            start_time=datetime.datetime.now()
            for idx, data in enumerate(data_loader[phase]):
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                labels_species = data['species'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_class, outputs_species = net(inputs)
                    _, pre_classes = torch.max(outputs_class, 1)
                    _, pre_species = torch.max(outputs_species, 1)
                    loss_class = criterion(outputs_class, labels_classes)
                    loss_species = criterion(outputs_species, labels_species)
                    loss = rate*loss_class + (1-rate)*loss_species
                    if (phase == 'train'):
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                corrects_classes += torch.sum((pre_classes == labels_classes) & (pre_species == labels_species))
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = 1.0 * corrects_classes.double() / len(data_loader[phase].dataset)
            print('{} loss: {:.3f} acc_rate: {:.3f} runtime(s): {}'.format(phase, epoch_loss, epoch_acc,
                                                                          datetime.datetime.now()-start_time))
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)
            if (phase == 'val' and epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model = copy.deepcopy(net.state_dict())
                print('Best val classes Acc: {:.4f}'.format(best_acc))
        scheduler.step()
    net.load_state_dict(best_model)
    time_str=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    torch.save(net.state_dict(), './model_pt/best_model_%s.pt'%time_str)
    print('Final val Classes Acc: {:.4f}'.format(best_acc))
    return net, Loss_list, Accuracy_list

net = Net().to(device)
optimizer = optim.SGD(net.parameters(), lr=.01, momentum=0.9)
# optimizer=optim.Adam(net.parameters(),lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
epoch_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30,  gamma=0.7)
net,Loss_list,Accuracy_list=train_model(net, optimizer, criterion, epoch_lr_scheduler, epochs=100,rate=0.8)

#save Loss_list
anno_loss_list=pd.DataFrame(Loss_list)
time_str=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
anno_loss_list.to_csv('./csv/loss_list_%s.csv' % time_str)

#save Accuracy_list
Accuracy_list={'train':torch.tensor(Accuracy_list['train']).numpy(),'val':torch.tensor(Accuracy_list['val']).numpy()}
anno_acc_list=pd.DataFrame(Accuracy_list)
time_str=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
anno_acc_list.to_csv('./csv/acc_list_%s.csv' % time_str)
