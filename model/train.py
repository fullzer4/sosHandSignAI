import tensorflow as tf
import numpy as np
import os
import json

with open('./images/train/train.json') as f:
    train_data = json.load(f)['images']

train_ds = tf.data.Dataset.from_tensor_slices({
    'filename': [image['filename'] for image in train_data],
    'gesture': [image['gesture'] for image in train_data],
    'data': [image['data'] for image in train_data]
})

def decode_image(image_data):
    image = tf.io.decode_base64(image_data)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

train_ds = train_ds.map(lambda x: (decode_image(x['data']), x['gesture']))

batch_size = 8
num_epochs = 50
learning_rate = 0.001

num_classes = 2
input_shape = (512, 512, 3)

train_dataset = tf.data.TFRecordDataset("train.tfrecord")

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "xmin": tf.io.VarLenFeature(tf.float32),
    "ymin": tf.io.VarLenFeature(tf.float32),
    "xmax": tf.io.VarLenFeature(tf.float32),
    "ymax": tf.io.VarLenFeature(tf.float32),
}

def _parse_example(example_proto):
    features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(features["image"], channels=3)
    label = tf.cast(features["label"], tf.int32)
    xmin = tf.sparse.to_dense(features["xmin"])
    ymin = tf.sparse.to_dense(features["ymin"])
    xmax = tf.sparse.to_dense(features["xmax"])
    ymax = tf.sparse.to_dense(features["ymax"])
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return image, {"bbox": boxes, "label": label}

train_dataset = train_dataset.map(_parse_example)

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, input_shape[:2])
    return image, label

train_dataset = train_dataset.map(preprocess_image).shuffle(buffer_size=1000).batch(batch_size).prefetch(1)

base_model = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
base_model.trainable = True
inputs = tf.keras.Input(shape=input_shape)
x = base_model(inputs, training=True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=num_epochs)