import tensorflow as tf
from tensorflow.keras import layers
import numpy as np



# Patch Embedding--------------------------
# https://keras.io/examples/vision/image_classification_with_vision_transformer/

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# Random Sampling-------------------------
class RandomSampling(layers.Layer):    
    def __init__(self, num_patches, mask_ratio=0.75):
        super(RandomSampling, self).__init__()
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio

        self.num_mask = int(mask_ratio * num_patches)
        self.un_masked_indices = None
        self.mask_indices = None

    def call(self, patches):
        """
            returns:
                unmasked patches, [mask indices, unmasked indices]
        """
        self.mask_indices = np.random.choice(self.num_patches, size=self.num_mask,
                            replace=False)
        self.un_masked_indices = np.delete(np.array(range(self.num_patches)), self.mask_indices)
        return tf.gather(patches, self.un_masked_indices, axis=1)


# Mask Token---------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class MaskToken(layers.Layer):
    """Append a mask token to encoder output."""
    def __init__(self, mask_indices, un_masked_indices):
        super(MaskToken, self).__init__()
        self.mask_indices = np.array([index for index in mask_indices])
        self.un_masked_indices = np.array([index for index in un_masked_indices])
        self.indices = None


    def build(self, input_shape):
        mst_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        out_init = tf.zeros_initializer()
        self.mst = tf.Variable(
            name="mst",
            initial_value=mst_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )
        

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # num_patches = self.mask_indices.shape[0] + self.un_masked_indices.shape[0]
        mask_num = self.mask_indices.shape[0]
        
        # broadcast mask token for batch
        mst_broadcasted = tf.cast(
                            tf.broadcast_to(self.mst, [batch_size, mask_num, self.hidden_size]),
                            dtype=inputs.dtype,
                        )
        
        # concat
        self.indices = np.concatenate([self.mask_indices, self.un_masked_indices], axis=0)
        updates = tf.concat([mst_broadcasted, inputs], axis=1)
                       
        out = tf.gather(updates, self.indices, axis=1, batch_dims=0)
        return out

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Positional Encoding---------------------------------------
# https://www.tensorflow.org/text/tutorials/transformer#positional_encoding

def pos_encode(pos, d_model):
    def get_angle(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    angle_rads = get_angle(np.arange(pos)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)    


# Patch Encoder--------------------------
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
# edited positional encoding part 

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = pos_encode(
            pos=num_patches, d_model=projection_dim
        )

    def call(self, patch):
        encoded = self.projection(patch) + self.position_embedding
        return encoded


# Transformer Block--------------------------------------
# https://github.com/faustomorales/vit-keras/blob/master/vit_keras/layers.py

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Implements a Transformer Encoder block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=input_shape[-1] // self.num_heads,  #input_shape[-1] = d_model
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.att(inputs, inputs)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        x = x + y
        x = self.layernorm1(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)