# https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras

import tensorflow as tf
from tensorflow import keras

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)


class ReconstructImage(keras.callbacks.Callback):
    def __init__(self, img_tag, log_dir, image):
        super().__init__() 
        self.tag = img_tag
        self.image = image
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs={}):
    
        # Reconstruct the image
        img = self.model.predict(self.image)

        image = make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(summary, epoch)
        writer.close()

        return