# 1.2 import dependencies
# bibliotecas padroes
import cv2  # opencv
import os  # criar a estrutura de pastas
import random
import numpy as np  # arrays
import matplotlib.pyplot as plt  # visualizar as imagens
from colorama import Fore, Style
import uuid  # gerar nomes únicos para as imagens

# tensorflow - functional API
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# 1.3 set gpu growth - limitar o tanto de vram que o tensorflow poderá usar
# avoid OOM (out of memory) errors


try:
    gpus = tf.config.experimental.list_physical_devices(
        'GPU')  # pega todas as gpu's da maquina
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu, True)  # set memory growth
except:
    print(f'{Fore.RED}nenhuma gpu dedicada encontrada{Style.RESET_ALL}')

# 1.4 create folder structures


POS_PATH = os.path.join('data', 'positive')  # imagem igual
NEG_PATH = os.path.join('data', 'negative')  # imagem diferente
ANC_PATH = os.path.join('data', 'anchor')  # imagem da camera


try:
    os.makedirs(POS_PATH)
    os.makedirs(NEG_PATH)
    os.makedirs(ANC_PATH)
except:
    print(f'{Fore.RED}pastas já criadas{Style.RESET_ALL}')

# 2 collect positives and anchores
# 2.1 untar labelled faces in the Wild Dataset


# database de imagens para usar como negative: http://vis-www.cs.umass.edu/lfw/
# mover todas as imagens para data/negative
try:
    for directory in os.listdir('lfw'):
        for file in os.listdir(os.path.join('lfw', directory)):
            EX_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)
except:
    print(f'{Fore.RED}arquivos já movidos{Style.RESET_ALL}')

# 2.2 collect positive and anchor classes


# estabelecendo conexão com a webcam
# se der erro aqui, tente outros numeros pq pode variar de pc pra pc
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # cortar frame para 250x250px
    frame = frame[120:120+250, 200:200+250, :]

    # collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):  # aperte a para salvar a imagem
        # unique file path
        imgname = os.path.join(ANC_PATH, f'{uuid.uuid1()}.jpg')
        # salvando imagem
        cv2.imwrite(imgname, frame)

    # collect positive
    if cv2.waitKey(1) & 0XFF == ord('p'):
        imgname = os.path.join(POS_PATH, f'{uuid.uuid1()}.jpg')
        cv2.imwrite(imgname, frame)

    # mostrar imagem na tela
    cv2.imshow('Image Collection', frame)

    # stop
    if cv2.waitKey(1) & 0XFF == ord('q'):  # aperte q para sair
        break
# liberar a webcam
cap.release()
# fechar janela
cv2.destroyAllWindows()

# 3 load and preprocess images
# 3.1 get image directories


anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(300)

# 3.2 preprocessing - scale and resize


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img /= 255.0
    return img

# 3.3 create labelled dataset


positives = tf.data.Dataset.zip(
    (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip(
    (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# 3.4 build train and test partition


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# 4 model engineering
# 4.1 create embedding layer


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # first block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # final embedding layer
    c4 = Conv2D(256,  (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_embedding()

# 4.2 build distance layer


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# 4.3 make siamese model


def make_siamese_model():
    # anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image),
                              embedding(validation_image))

    # classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=[classifier], name='SiameseNetwork')


siamese_model = make_siamese_model()

# 5 training
# 5.1 setup loss and optimizer

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

# 5.2 establish checkpoints

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# 5.3 build train step function

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        # get anchor and positive/negative image
        X = batch[:2]
        # get label
        y = batch[2]

        # forward pass
        yhat = siamese_model(X, training=True)
        # calculate loss
        loss = binary_cross_loss(y, yhat)

    # calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    return loss