# bibliotecas padroes
import cv2  # opencv
import os  # para criar a estrutura de pastas
import random
import numpy as np  # para arrays
from matplotlib import pyplot as plt  # para visualizar as imagens
from colorama import Fore, Style
import uuid  # gerar nomes únicos para as imagens

# tensorflow - functional API
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# gpu growth - limitar o tanto de vram que o tensorflow poderá usar
# só fazer se for rodar o código localmente e com gpu dedicada
# avoid OOM (out of memory) errors
try:
    gpus = tf.config.experimental.list_physical_devices(
        'GPU')  # pega todas as gpu's da maquina
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu, True)  # set memory growth]
except:
    print(f'{Fore.RED}nenhuma gpu dedicada encontrada{Style.RESET_ALL}')

# setup paths

POS_PATH = os.path.join('data', 'positive')  # imagem igual
NEG_PATH = os.path.join('data', 'negative')  # imagem diferente
ANC_PATH = os.path.join('data', 'anchor')  # imagem da camera

# criando pastas

try:
    os.makedirs(POS_PATH)
    os.makedirs(NEG_PATH)
    os.makedirs(ANC_PATH)
except:
    print(f'{Fore.RED}pastas já criadas{Style.RESET_ALL}')

# collect negative images

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

# collect positive and anchor images

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

# get image directories

anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(300)

# preprocessing - scale and resize


def preProcess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img /= 255.0
    return img

# create labelled dataset


positives = tf.data.Dataset.zip(
    (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip(
    (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()

ex = samples.next()
print(ex)
