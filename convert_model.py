import tensorflow as tf

print("[1/3] Loading heavy model...")
# Load your existing heavy model
model = tf.keras.models.load_model('satellite-imagery.keras', compile=False)

print("[2/3] Compressing to Lite format...")
# Initialize the Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization: This tells TF to optimize the math for your CPU
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

print("[3/3] Saving optimized asset...")
# Save the new lightweight file
with open('satellite_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

print("SUCCESS: Model converted to 'satellite_optimized.tflite'")