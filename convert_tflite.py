import os
import numpy as np
import tensorflow as tf

MODEL_KERAS = "models/mlp48.keras"
SAVEDMODEL_DIR = "models/mlp48_savedmodel"
OUT_TFLITE = "models/mlp48_int8.tflite"

# 1) load model
model = tf.keras.models.load_model(MODEL_KERAS, compile=False)

# 2) export SavedModel (ổn định hơn khi convert)
if os.path.exists(SAVEDMODEL_DIR):
    # optional: bạn có thể xoá folder này thủ công nếu muốn
    pass
model.export(SAVEDMODEL_DIR)

# 3) representative dataset (tốt nhất dùng dữ liệu thật)
def rep_data():
    # load vài vector thật từ data_vec (nếu có)
    # nếu không có thì random vẫn chạy được nhưng quant kém hơn
    import glob
    files = glob.glob("data_vec/*/*.npy")
    if len(files) >= 200:
        for f in files[:200]:
            v = np.load(f).astype(np.float32).reshape(1, 48)
            yield [v]
    else:
        for _ in range(200):
            yield [np.random.uniform(-1, 1, (1, 48)).astype(np.float32)]

# 4) convert from SavedModel -> INT8
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print("Saved:", OUT_TFLITE)
