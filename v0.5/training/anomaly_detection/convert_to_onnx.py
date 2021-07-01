import keras2onnx
import tensorflow as tf
from IPython import embed

print(f"{tf.keras.__version__ = }")

# Load model
model_path = "trained_models/ad01.h5"
model = tf.keras.models.load_model(model_path)

# onnx_model = keras2onnx.convert_keras(model, "ad01")

# The current version of tensorflow.keras (2.5) is incompatible with keras2onnx
tf.saved_model.save(model, "trained_models/tmp_model")

# python3 -m tf2onnx.convert --saved-model trained_models/tmp_model --output trained_models/ad01.onnx
# convert_ad01.log

embed()
