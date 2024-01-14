import tensorflow as tf
import tensorflow_hub as hub

# Load a pre-trained model (e.g., SSD MobileNet V2)
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Function to run detection
def detect_objects(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)  # Add batch dimension

    detector_output = detector(img)
    return detector_output

# Example usage
image_path = 'Object_detection\\image.jpg'
detection_result = detect_objects(image_path)
print(detection_result)
