import tensorflow as tf

# To convert normal model to quant model
# convertor =tf.lite.TFLiteConverter.from_saved_model("/Users/himanshusharma/PycharmProjects/potato-disease/models/3")
# tflite_model = convertor.convert()
# with open("tflite_model.tflite", "wb") as f:
#     f.write(tflite_model)

# To convert normal model to lite and then quant model
print("TensorFlow version:", tf.__version__)
quantConvertor =tf.lite.TFLiteConverter.from_saved_model("/Users/himanshusharma/PycharmProjects/potato-disease/models/3")
quantConvertor.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = quantConvertor.convert()
with open("tflite_quant_model.tflite", "wb") as f:
    f.write(tflite_quant_model)