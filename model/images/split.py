import os
import json
import tensorflow as tf
from PIL import Image

src_folder = "all"
train_folder = "train"
test_folder = "test"

train_ratio = 0.8
test_ratio = 0.2

if not os.path.exists(train_folder):
    os.makedirs(train_folder)

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Obter uma lista de todas as imagens na pasta de origem
images = [f for f in os.listdir(src_folder) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

# Determinar quantas imagens irão para cada conjunto
num_train = int(len(images) * train_ratio)
num_test = len(images) - num_train

# Separar as imagens em conjuntos de treinamento e teste
train_images = images[:num_train]
test_images = images[num_train:]

# Criar os arquivos JSON de cada conjunto
train_data = []
for image_name in train_images:
    with open(os.path.join(src_folder, image_name), 'rb') as f:
        image_data = f.read()
    image = Image.open(os.path.join(src_folder, image_name))
    width, height = image.size
    train_data.append({
        'filename': image_name,
        'gesture': 'gest1' if 'gest1' in image_name else 'gest2',
        'format': image.format.lower(),
        'size': [width, height],
        'data': tf.io.encode_base64(image_data).numpy().decode()
    })

test_data = []
for image_name in test_images:
    with open(os.path.join(src_folder, image_name), 'rb') as f:
        image_data = f.read()
    image = Image.open(os.path.join(src_folder, image_name))
    width, height = image.size
    test_data.append({
        'filename': image_name,
        'gesture': 'gest1' if 'gest1' in image_name else 'gest2',
        'format': image.format.lower(),
        'size': [width, height],
        'data': tf.io.encode_base64(image_data).numpy().decode()
    })

with open(os.path.join(train_folder, 'train.json'), 'w') as f:
    json.dump({'images': train_data}, f)

with open(os.path.join(test_folder, 'test.json'), 'w') as f:
    json.dump({'images': test_data}, f)