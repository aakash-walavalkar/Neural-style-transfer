import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_image(image_path, image_size=(256, 256)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image = image[tf.newaxis, :]
    return image

def stylize_image(content_image, style_image):
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image

#set your files path here
content_path = '/content/IMG_20230408_163146.jpg'
style_path = '/content/abstract-art.jpeg'

# Enable GPU acceleration
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

content_image = load_image(content_path)
style_image = load_image(style_path)

stylized_image = stylize_image(content_image, style_image)
stylized_image = tensor_to_image(stylized_image)

plt.subplot(1, 3, 1)
plt.imshow(load_image(content_path)[0])
plt.title('Content Image')

plt.subplot(1, 3, 2)
plt.imshow(load_image(style_path)[0])
plt.title('Style Image')

plt.subplot(1, 3, 3)
plt.imshow(stylized_image)
plt.title('Stylized Image')

plt.show()
