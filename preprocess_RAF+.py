import os
import numpy as np
import h5py
from PIL import Image

raf_path = 'RAF-DB'
train_path = raf_path + '/' + 'train'
test_path = raf_path + '/' + 'test'

data_path = [train_path, test_path]

datapath = os.path.join('data_raf', 'RAF_.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))
datafile = h5py.File(datapath, 'w')

for path in data_path:
    anger_path = os.path.join(path, 'Anger')
    disgust_path = os.path.join(path, 'Disgust')
    fear_path = os.path.join(path, 'Fear')
    happiness_path = os.path.join(path, 'Happiness')
    neutral_path = os.path.join(path, 'Neutral')
    sadness_path = os.path.join(path, 'Sadness')
    surprise_path = os.path.join(path, 'Surprise')

    tot_path = [
        anger_path, disgust_path, fear_path, happiness_path, neutral_path,
        sadness_path, surprise_path
    ]

    # # Creat the list to store the data and label information
    data_x = []
    data_y = []

    print("Start to save data!!!")

    for i in range(len(tot_path)):
        print("i = ", i)
        files = os.listdir(tot_path[i])
        files.sort()
        for filename in files:
            I = np.array(Image.open(os.path.join(tot_path[i], filename)))
            data_x.append(I.tolist())
            data_y.append(i)

    print(np.shape(data_x))
    print(np.shape(data_y))

    datafile.create_dataset(
        "data_" + path.split('/')[-1] + '_img', dtype='uint8', data=data_x
    )
    datafile.create_dataset(
        "data" + path.split('/')[-1] + '_label', dtype='int64', data=data_y
    )

datafile.close()
print("Save data finish!!!")
