import tensorflow as tf

# GPUの確認
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUが利用可能です:")
    for gpu in gpus:
        print(f"- {gpu}")
else:
    print("GPUが利用できません。")
