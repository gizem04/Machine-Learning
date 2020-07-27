# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 02:45:53 2020

@author: gizem arslan
"""

import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense



IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
RESIM_KANALLAR=3

dosyaİsimleri = os.listdir("train/")
kategoriler = []
for dosyaİsmi in dosyaİsimleri:
    kategori = dosyaİsmi.split('.')[0]
    if kategori == 'e,a':
        kategoriler.append(1)
    if kategori == 'b':
        kategoriler.append(2)
    if kategori == 'p':
        kategoriler.append(3)
    if kategori == 't':
        kategoriler.append(4) 
    if kategori == 'c':
        kategoriler.append(5)
    if kategori == 'ç':
        kategoriler.append(6)
    if kategori == 'h':
        kategoriler.append(7)
    if kategori == 'd':
        kategoriler.append(8) 
    if kategori == 'z':
        kategoriler.append(9)
    if kategori == 'r':
        kategoriler.append(10)
    if kategori == 'j':
        kategoriler.append(11)
    if kategori == 'ş':
        kategoriler.append(12) 
    if kategori == 's':
        kategoriler.append(13)
    if kategori == 'd,z':
        kategoriler.append(14)
    if kategori == 'a,h':
        kategoriler.append(15)
    if kategori == 'g':
        kategoriler.append(16) 
    if kategori == 'f':
        kategoriler.append(17)
    if kategori == 'k':
        kategoriler.append(18)
    if kategori == 'l':
        kategoriler.append(19) 
    if kategori == 'm':
        kategoriler.append(20)
    if kategori == 'n':
        kategoriler.append(21)
    if kategori == 'v,o,ö,u,ü':
        kategoriler.append(22)
    if kategori == 'la':
        kategoriler.append(23)
    if kategori == 'y,ı,i':
        kategoriler.append(24) 

df = pd.DataFrame({
    'dosyaİsmi': dosyaİsimleri,
    'kategori': kategoriler
})
    
df['kategori'].value_counts().plot.bar()


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, RESIM_KANALLAR)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(24, activation='softmax')) 
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

df["kategori"] = df["kategori"].replace({1: 'e,a', 2: 'b',3: 'p', 4: 't',5: 'c', 6: 'ç',7: 'h', 8: 'd',9: 'z', 10: 'r',11: 'j', 12: 'ş',13: 's', 14: 'd,z',15: 'a,h', 16: 'g',17: 'f',18: 'k', 19: 'l',20: 'm', 21: 'n',22: 'v,o,ö,u,ü', 23: 'la',24: 'y,ı,i'}) 
eğitim_df, doğrulama_df = train_test_split(df, test_size=0.20, random_state=42)
eğitim_df = eğitim_df.reset_index(drop=True)
doğrulama_df = doğrulama_df.reset_index(drop=True)

eğitim_df['kategori'].value_counts().plot.bar()

toplam_eğitim = eğitim_df.shape[0]
toplam_doğrulama = doğrulama_df.shape[0]
batch_size=15

eğitim_dataOluştur = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

eğitim_oluştur = eğitim_dataOluştur.flow_from_dataframe(
    eğitim_df, 
    "train2/", 
    x_col='dosyaİsmi',
    y_col='kategori',
    target_size=IMAGE_SIZE,
    class_mode='sparse',
    batch_size=batch_size
)

doğrulama_veriOluştur = ImageDataGenerator(rescale=1./255)
doğrulama_oluştur = doğrulama_veriOluştur.flow_from_dataframe(
    doğrulama_df, 
    "train2/", 
    x_col='dosyaİsmi',
    y_col='kategori',
    target_size=IMAGE_SIZE,
    class_mode='sparse',
    batch_size=batch_size
)
örnek_df = eğitim_df.sample(n=1).reset_index(drop=True)
örnek_oluştur = eğitim_dataOluştur.flow_from_dataframe(
    örnek_df, 
    "train2/", 
    x_col='dosyaİsmi',
    y_col='kategori',
    target_size=IMAGE_SIZE,
    class_mode='sparse'
)

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in örnek_oluştur:
        imge = X_batch[0]
        plt.imshow(imge)
        break
plt.tight_layout()
plt.show()


history = model.fit(
    eğitim_oluştur, 
    epochs=20,
    verbose=True,
    validation_data=doğrulama_oluştur,

)
model.save_weights("model1.h5")

test_dosyaİsimleri = os.listdir("test/")
test_df = pd.DataFrame({
    'dosyaİsmi': test_dosyaİsimleri
})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "test/", 
    x_col='dosyaİsmi',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

test_df['kategori'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in eğitim_oluştur.class_indices.items())
test_df['kategori'] = test_df['kategori'].replace(label_map)

sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 12))
for index, row in sample_test.iterrows():
    dosyaİsmi = row['dosyaİsmi']
    kategori = row['kategori']
    img = load_img("test/"+dosyaİsmi, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(dosyaİsmi + '(' + "{}".format(kategori) + ')' )
plt.tight_layout()
plt.show()













    
    
    
