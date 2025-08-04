

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore")


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""# Load Data"""

os.listdir("Blood cell Cancer [ALL]")

path_main = "Blood cell Cancer [ALL]"
for folder in os.listdir(path_main):
    list_of_elements = os.listdir(os.path.join(path_main, folder))

    print(f'Folder: {folder}\n')
    print(f'Number of elements: {len(list_of_elements)}\n')
    print(f'First item\'s name: {list_of_elements[0]}\n')
    print('***************************')

"""# plot images"""

Benign_path = "Blood cell Cancer [ALL]/Benign"
malignant_pre_B_path = "Blood cell Cancer [ALL]/[Malignant] Pre-B"
malignant_pro_B_path = "Blood cell Cancer [ALL]/[Malignant] Pro-B"
malignant_early_pre_B_path = "Blood cell Cancer [ALL]/[Malignant] early Pre-B"

def plot_imgs(item_dir,num_imgs=15):
    all_item_dirs = os.listdir(item_dir)
    item_files=[os.path.join(item_dir,file) for file in all_item_dirs][:num_imgs]

    plt.figure(figsize=(10,10))
    for idx,img_path in enumerate (item_files):
        plt.subplot(5,5,idx+1)
        img=plt.imread(img_path)
        plt.imshow(img)

    plt.tight_layout()

"""# **Benign**"""

plot_imgs(Benign_path,num_imgs=15)

"""# **malignant-pre-B**"""

plot_imgs(malignant_pre_B_path,num_imgs=15)

"""# **malignant-pro-B**"""

plot_imgs(malignant_pro_B_path,num_imgs=15)

"""# **malignant-early-pre-B**"""

plot_imgs(malignant_early_pre_B_path,num_imgs=15)

"""# **Make a list of labels and file paths**"""

Benign_dir = ['Blood cell Cancer [ALL]/Benign']

Malignant_Pre_B_dirs = [

    'Blood cell Cancer [ALL]/[Malignant] Pre-B'
]


Malignant_Pro_B_dirs =[

    'Blood cell Cancer [ALL]/[Malignant] Pro-B'
]


Malignant_early_Pre_B_dirs = [

    'Blood cell Cancer [ALL]/[Malignant] early Pre-B'
]

filepaths = []
labels = []
dict_lists = [Benign_dir, Malignant_Pre_B_dirs,Malignant_Pro_B_dirs, Malignant_early_Pre_B_dirs]
class_labels = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']

for i, dir_list in enumerate(dict_lists):
    for j in dir_list:
        flist = os.listdir(j)
        for f in flist:
            fpath = os.path.join(j, f)
            filepaths.append(fpath)
            labels.append(class_labels[i])

Fseries = pd.Series(filepaths, name="filepaths")
Lseries = pd.Series(labels, name="labels")
bloodCell_data = pd.concat([Fseries, Lseries], axis=1)
bloodCell_df = pd.DataFrame(bloodCell_data)
print(bloodCell_df.head())
print(bloodCell_df["labels"].value_counts())

"""# **splitting data**"""

train_images, test_images = train_test_split(bloodCell_df, test_size=0.3, random_state=42)
train_set, val_set = train_test_split(bloodCell_df, test_size=0.2, random_state=42)

print(train_set.shape)
print(test_images.shape)
print(val_set.shape)

"""# **Data Augmentation**"""

image_gen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input)


train = image_gen.flow_from_dataframe(dataframe= train_set,x_col="filepaths",y_col="labels",
                                      target_size=(224,224),
                                      color_mode='rgb',
                                      class_mode="categorical",
                                      batch_size=8,
                                      shuffle=False
                                     )


test = image_gen.flow_from_dataframe(dataframe= test_images,x_col="filepaths", y_col="labels",
                                     target_size=(224,224),
                                     color_mode='rgb',
                                     class_mode="categorical",
                                     batch_size=8,
                                     shuffle= False
                                    )


val = image_gen.flow_from_dataframe(dataframe= val_set,x_col="filepaths", y_col="labels",
                                    target_size=(224,224),
                                    color_mode= 'rgb',
                                    class_mode="categorical",
                                    batch_size=8,
                                    shuffle=False
                                   )

classes=list(train.class_indices.keys())
print (classes)

"""# **Show Augmented Images**"""

def show_Blood_images(image_gen):
    test_dict = test.class_indices
    classes = list(test_dict.keys())
    images, labels=next(image_gen) # get a sample batch from the generator
    plt.figure(figsize=(20,20))
    length = len(labels)
    if length<25:
        r=length
    else:
        r=25
    for i in range(r):
        plt.subplot(5,5,i+1)
        image=(images[i]+1)/2 #scale images between 0 and 1
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color="green",fontsize=16)
        plt.axis('off')
    plt.show()
show_Blood_images(train)

"""# *Using VGG19 Model*"""

vgg_model=VGG19(weights="imagenet",include_top=False,input_shape=(224,224,3))

model=Sequential()

model.add(vgg_model)

model.add(Flatten())

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))


model.add(Dense(128,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(4,activation="softmax"))

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

history = model.fit(train, epochs=5, validation_data=val, verbose=1)

from keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.evaluate(test, verbose=1)

"""# **plot history**"""

model_loss = pd.DataFrame(history.history)
model_loss.plot()

"""# **Predict test**"""

pred = model.predict(test)
pred = np.argmax(pred, axis=1)

labels = (train.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred2 = [labels[k] for k in pred]

y_test = test_images.labels # set y_test to the expected output
print(classification_report(y_test, pred2))
print("Accuracy of the Model:","{:.1f}%".format(accuracy_score(y_test, pred2)*100))

class_labels = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']


cm = confusion_matrix(y_test, pred2)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues')

plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=class_labels)
plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()