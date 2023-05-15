import tf2onnx
import torch
import onnx
import tensorflow as tf

model = torch.load('./best.pt')

dummy_input = torch.randn(16, 3, 1280, 720)

onnx_path = './modelo.onnx'
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

onnx_model = onnx.load('./modelo.onnx')

tf_model, _ = tf2onnx.convert.from_onnx(onnx_model)

saved_model_path = '../website/src/model'

tf.saved_model.save(tf_model, saved_model_path)