import tensorflow as tf

# Load the model in the original environment
model = tf.keras.models.load_model("trained_model.h5")

# Save it in the new format
model.save("new_trained_model.h5")
