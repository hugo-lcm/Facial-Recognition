# bibliotecas padroes
import cv2  # opencv
import os  # para criar a estrutura de pastas
import random
import numpy as np  # para arrays
from matplotlib import pyplot as plt  # para visualizar as imagens
from colorama import Fore, Style

# tensorflow - functional API
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# gpu growth - limitar o tanto de vram que o tensorflow poderá usar
# só fazer se for rodar o código localmente e com gpu dedicada
# avoid OOM (out of memory) errors

# gpus = tf.config.experimental.list_physical_devices('GPU') # pega todas as gpu's da maquina
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True) # set memory growth

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
