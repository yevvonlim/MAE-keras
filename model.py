from layers import PatchEncoder, Patches, PatchesToImage, TransformerBlock, RandomSampling, MaskToken, pos_encode
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def mae_model(input_shape, img_h, img_w, patch_size=40, d_model=32, dff=128, num_heads=4, drop=0.1, N_e=3, N_d=1):
# --------------- Embeddnig -------------------
    input = Input(shape=input_shape)
    x = Patches(patch_size) (input)
    num_patches = int(img_h*img_w//(patch_size*patch_size))
    x = PatchEncoder(num_patches, d_model) (x)
    
    random_sampler = RandomSampling(num_patches)
    x = random_sampler (x)
    
    mask_indices = random_sampler.mask_indices
    un_masked_indices = random_sampler.un_masked_indices
# ----------------- Encoder -------------------
    for _ in range(N_e):
        x = TransformerBlock(num_heads=num_heads, mlp_dim=dff, dropout=drop) (x) 
    mst = MaskToken(mask_indices, un_masked_indices)
    x = mst (x)

# -------------- Positional Embedding _-------------------
    x = x + pos_encode(x.shape[1], d_model) 

# ----------------- Decoder -------------------
    for _ in range(N_d):
        x = TransformerBlock(num_heads=num_heads, mlp_dim=dff, dropout=drop) (x)
    x = Dense(units=patch_size*patch_size*3) (x)
    x = PatchesToImage(img_h, img_w, 3, patch_size) (x)
    model = Model(inputs=input, outputs=x)
    return model

model = mae_model((1200, 800, 3), img_h=800, img_w=1200, d_model=768, dff=3072, N_e=12, N_d=1, num_heads=12)
model.summary()