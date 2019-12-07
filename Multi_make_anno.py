import pandas as pd
import os
from PIL import Image
import numpy as np

ROOTIMAGES = '../Dataset/'
PHASE = ['train', 'val']
CLASSES = ['Birds', 'Mammals']  # 0,1
SPECIES = ['rabbits', 'rats', 'chickens']

bad_train=pd.read_csv('../Dataset/bad_image/train_bad_images_number.csv',header=None)
bad_train=bad_train+'.jpg'
bad_train=np.array(bad_train).reshape(-1,)
# bad_val=pd.read_csv('../Dataset/bad_image/val_bad_images_number.csv',header=None)
# bad_val=bad_val+'.jpg'
# bad_val=np.array(bad_val).reshape(-1,)
bad_val=[]

# os.makedirs(PHASE,exist_ok=True)

data_info = {'train': {'path': [], 'classes': [], 'species': []},
             'val': {'path': [], 'classes': [], 'species': []}}

for p in PHASE:
    # pha = []
    for s in SPECIES:
        dir = os.listdir(ROOTIMAGES + p + '/' + s)
        DIR = ROOTIMAGES + p + '/' + s + '/'
        for item in dir:
            if((p=='train' and item not in bad_train)or
              (p=='val' and item not in bad_val)):
                try:
                    img = Image.open(DIR + item)
                    data_info[p]['path'].append(DIR + item)
                    if (s == 'rabbits'):
                        data_info[p]['species'].append(0)
                        data_info[p]['classes'].append(0)
                    elif (s == 'rats'):
                        data_info[p]['species'].append(1)
                        data_info[p]['classes'].append(0)
                    elif (s == 'chickens'):
                        data_info[p]['species'].append(2)
                        data_info[p]['classes'].append(1)
                except OSError:
                    pass
    print(p,' data_info.len:',len(data_info[p]['path']))
    anno = pd.DataFrame(data_info[p])
    # print(anno)
#     anno.to_csv('Multi_%s_make_anno.csv' % p)
    anno.to_csv('./csv/Multi_%s_make_anno_good.csv' % p)