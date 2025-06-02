##
## Kolokvij - od prošle
##

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten
from tensorflow.keras.datasets import cifar100
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Učitavanje CIFAR-100 skupa podataka
(x_train, _), (x_test, _) = cifar100.load_data()

# Normalizacija podataka
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Spljoštavanje podataka
x_train_flat = x_train.reshape((len(x_train), -1))
x_test_flat = x_test.reshape((len(x_test), -1))

# Dimenzija ulaznih podataka
input_dim = x_train_flat.shape[1]

# Dodavanje šuma slikama
noise_factor = 0.1
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Spljoštavanje podataka sa šumom
x_train_noisy_flat = x_train_noisy.reshape((len(x_train_noisy), -1))
x_test_noisy_flat = x_test_noisy.reshape((len(x_test_noisy), -1))

# Definiranje enkodera
input_layer = Input(shape=(input_dim,))
encoder = Dense(512, activation='relu')(input_layer)
encoder = Dense(256, activation='relu')(encoder)
encoder = Dense(128, activation='relu')(encoder)

# Definiranje dekodera
decoder = Dense(128, activation='relu')(encoder)
decoder = Dense(256, activation='relu')(decoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)  # Promjena ovdje

# Kreiranje autoenkodera s jasno definiranim izlazom
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Kompiliranje modela
autoencoder.compile(optimizer='adam', loss='mse')

# Treniranje modela
history = autoencoder.fit(x_train_noisy_flat, x_train_flat, epochs=10, batch_size=256, validation_data=(x_test_noisy_flat, x_test_flat), verbose=1)

## VJEŽBA 7 - neuronske mreže
##

import tensorflow as tf
from tensorflow import keras
tf.__version__

#Dohvacanje podataka
baza=keras.datasets.fashion_mnist
(X_train,y_train), (X_test,y_test)=baza.load_data()

X_train.shape
y_train
class_names=["T-shirt/top", "Trouser", "Pullover","Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
y_train[0]
class_names[y_train[0]]
X_train[0].shape

import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='gray')

X_train=X_train/255
X_valid=X_train[:5000]
X_train=X_train[5000:]
y_valid=y_train[:5000]
y_train=y_train[5000:]

for i in range(0,5):
  plt.subplot(151+i)
  plt.imshow(X_valid[i], cmap='gray')
  plt.title(class_names[y_valid[i]])
plt.show()

#Definiranje modela
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.layers

#Prikaz arhitekture stvorenog modela
model.summary()
keras.utils.plot_model(model, show_shapes=True)
hidden1=model.layers[1]
weights, bias= hidden1.get_weights()
weights

#Kompailiranje modela
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"]) 

#Treniranje
history=model.fit(X_train,y_train,batch_size=32, epochs=20, validation_data=(X_valid, y_valid))
history.history.keys()

#Vizualizacija
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.ylim(0,1)
plt.show()

model.evaluate(X_test,y_test)
predikcije=model.predict(X_test)
predikcije

import numpy as np
y_pred=np.argmax(predikcije,axis=-1)
y_pred

np.array(class_names)[y_pred]

for i in range(0,5):
  plt.subplot(151+i)
  plt.imshow(X_test[i], cmap='gray')
  plt.title(class_names[y_pred[i]])   #što je predvidjelo
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred, target_names=class_names))

from sklearn import metrics
matrica_konfuzije = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrica_konfuzije, display_labels = class_names)
fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(ax=ax)
plt.show()

##
##VJEŽBA 8 - konvolucijske nm
##

import tensorflow as tf
tf.__version__
from tensorflow import keras
from keras.datasets import cifar10

#Dohvačanje podataka
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train.shape
y_train.shape
y_train
x_train[0]

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
print('Oznaka prve slike:', y_train[0])

#One hot encoding
y_train_one_hot=keras.utils.to_categorical(y_train, 10)
y_test_one_hot=keras.utils.to_categorical(y_test, 10)

#Normalizacija
y_train_one_hot[0]

x_train=x_train/255
x_test=x_test/255
x_train[0]

#Kreiranje CNN
model=keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3,3), activation='relu',padding='same' ,input_shape=(32,32,3)))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'))

#Sažimanje
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

#Odbacivanje
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=512, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=10, activation='softmax'))

#Treniranje CNN-a
model.compile(loss = 'categorical_crossentropy', # Corrected the typo in the loss function name
              optimizer = 'adam',
              metrics = ['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3,
                               restore_best_weights=True)

povijest_modela=model.fit(x_train, y_train_one_hot, batch_size=32,
                          epochs=20, validation_split=0.2,
                          callbacks = [early_stopping])

import pandas as pd
pd.DataFrame(povijest_modela.history).plot(figsize=(8,5))
plt.grid(True)
plt.xlabel('Epochs')
plt.show

model.save('my_cifar_model.keras')

#Testiranje
model.evaluate(x_test, y_test_one_hot)
predikcije_one_hot = model.predict(x_test)
predikcije_one_hot

import numpy as np
y_pred=np.argmax(predikcije_one_hot,axis=-1)

broj_u_klasu = ['avion', 'automobil', 'ptica', 'mačka', 'jelen', 'pas', 'zaba', 'konj', 'brod', 'kamion']

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=broj_u_klasu))

from sklearn import metrics
matrica_konfuzije = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrica_konfuzije, display_labels = broj_u_klasu)
fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(ax=ax)
plt.show()

#TESTIRANJE NA VLASTITIM SLIKAMA
!curl -o macka.jpg https://www.kucni-ljubimci.com/wp-content/uploads/2017/04/Sretna-maca.jpg

slika = plt.imread('macka.jpg')
slika.shape

from skimage.transform import resize
slika_resized = resize(slika, (32,32,3))
slika_resized.shape

import numpy as np
vjerojatnosti = model.predict(np.array([slika_resized]))

vjerojatnosti[0]

index = np.argsort(vjerojatnosti[0:])

for i in range(9,5,-1):
  print(broj_u_klasu[index[i]], ":", vjerojatnosti[0,index[i]])

##
##VJEŽBA 9 - autoenkoderi
##
##Učitavanje podataka
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

#Definiranje enkodera i dekodera
encoder=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(30, activation="relu")   #bottleneck
])

decoder=keras.models.Sequential([
    keras.layers.Dense(100, activation="relu",input_shape=[30]),
    keras.layers.Dense(28*28, activation="sigmoid"),
    keras.layers.Reshape([28,28])
])

#Spajanje dvije mreže autoenkodera
stacked_autoencoder=keras.models.Sequential([encoder,decoder])

#Treniranje modela
stacked_autoencoder.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
history=stacked_autoencoder.fit(x_train,x_train, epochs=10,validation_data=[x_test,x_test])

#Prikaz rezultata
plt.figure(figsize=(20,5))

for i in range(8):
  plt.subplot(2,8,i+1)
  plt.imshow(x_test[i], cmap="binary")

  plt.subplot(2,8,8+1+i)
  pred=stacked_autoencoder.predict(x_test[i].reshape(1,28,28))
  plt.imshow(pred.reshape(28,28),cmap="binary")

  plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.imshow(x_test[0], cmap="binary")

plt.subplot(1,3,2)
latent_vector=encoder.predict(x_test[0].reshape(1,28,28))
plt.imshow(latent_vector, cmap="binary")

plt.subplot(1,3,3)
pred=decoder.predict(latent_vector)
plt.imshow(pred.reshape(28,28),cmap="binary")

1-30/(28*28)

#Uklanjanje šuma
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(x_test[0], cmap="binary")

plt.subplot(1,2,2)
noise=np.random.random((28,28))/4
plt.imshow(x_test[0]+noise, cmap="binary")

encoder=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(100, activation="relu"), ######
    keras.layers.Dense(30, activation="relu")   #bottleneck
])

decoder=keras.models.Sequential([
    keras.layers.Dense(100, activation="relu",input_shape=[30]),
    keras.layers.Dense(100, activation="relu"),#####
    keras.layers.Dense(28*28, activation="sigmoid"),
    keras.layers.Reshape([28,28])
])
stacked_autoencoder=keras.models.Sequential([encoder,decoder])
stacked_autoencoder.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

#dodavanje šuma
x_train_noise=x_train+((np.random.random(x_train.shape))/4)
x_test_noise=x_test+((np.random.random(x_test.shape))/4)

history=stacked_autoencoder.fit(x_train_noise,x_train, epochs=10,validation_data=[x_test_noise,x_test])

plt.figure(figsize=(20,5))

for i in range(8):
  plt.subplot(2,8,i+1)
  plt.imshow(x_test_noise[i], cmap="binary")

  plt.subplot(2,8,8+1+i)
  pred=stacked_autoencoder.predict(x_test_noise[i].reshape(1,28,28))
  plt.imshow(pred.reshape(28,28),cmap="binary")


##KONVOLUCIJSKI AUTOENKODER
  encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2)
])
  
encoder.predict(x_test[0].reshape((1, 28, 28))).shape   #ovo nam treba za input shape napisat 1 je uspred jer je samo za jednu sliku tu gledano

decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding="valid",
                                 activation="relu",
                                 input_shape=[3, 3, 64]),
    keras.layers.Conv2DTranspose(16, kernel_size=(3, 3), strides=2, padding="same",
                                 activation="relu"),
    keras.layers.Conv2DTranspose(1, kernel_size=(3, 3), strides=2, padding="same",
                                 activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

stacked_autoencoder = keras.models.Sequential([encoder, decoder])
stacked_autoencoder.compile(loss="binary_crossentropy",
                            optimizer='adam', metrics=['accuracy'])

history = stacked_autoencoder.fit(x_train, x_train, epochs=5,   #stavi 10
                         validation_data=[x_test, x_test])

plt.figure(figsize=(20, 5))
for i in range(8):
  plt.subplot(2, 8, i+1)
  pred = stacked_autoencoder.predict(x_test[i].reshape((1, 28, 28)))
  plt.imshow(x_test[i], cmap="binary")

  plt.subplot(2, 8, i+8+1)
  plt.imshow(pred.reshape((28, 28)), cmap="binary")

plt.figure(figsize=(15,15))
for i in range(8 * 8):
  plt.subplot(8, 8, i+1)
  plt.imshow(encoder.layers[-2].weights[0][:, :, 0, i])
##
##VJEŽBA 10 - postojeće neuroneske mreže
##

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train = x_train[:2000]
y_train = y_train[:2000]
x_test = x_test[:100]
y_test = y_test[:100]

#dimenzije za Resnet
x_train = tf.image.resize(x_train, (224, 224))
x_test = tf.image.resize(x_test, (224, 224))

#Predobrada podataka
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Treniranje ResNet50
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

model = models.Sequential()
model.add(base_model)

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
povijest = model.fit(x_train, y_train, epochs=8, validation_split = 0.3)

base_model2 = ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model2.trainable = False

model2 = models.Sequential()
model2.add(base_model2)

model2.add(layers.GlobalAveragePooling2D())
model2.add(layers.Dense(10, activation='relu'))
model2.add(layers.Dense(10, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
povijest2 = model2.fit(x_train, y_train, epochs=8, validation_split = 0.3)

#Vizualizacija
import pandas as pd
import matplotlib.pyplot as plt


pd.DataFrame(povijest.history).plot(figsize=(8,5))
plt.grid(True)
plt.ylim(0, 1)
plt.xlabel('Epochs')
plt.show

import pandas as pd
import matplotlib.pyplot as plt


pd.DataFrame(povijest2.history).plot(figsize=(8,5))
plt.grid(True)
plt.ylim(0, 1)
plt.xlabel('Epochs')
plt.show

