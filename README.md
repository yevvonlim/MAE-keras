# MAE-keras
Unofficial keras(tensorflow) implementation of MAE model described in 'Masked Autoencoders Are Scalable Vision Learners'.
This work has been done on the basis of https://keras.io/examples/vision/image_classification_with_vision_transformer/,
                                        https://www.tensorflow.org/text/tutorials/transformer#positional_encoding,
                                        https://keras.io/examples/vision/image_classification_with_vision_transformer/

Currently, only pre-training mode is supported. But you can easily fine-tune the model using its encoder part. 
It is not complete, flawless implementation, so its performance could be different from the paper.
If there is anything to modify, please make it right :) Thanks.

✔️ Supported (DONE)  
▫️ Sinusoidal positional encoding at both encoder and decoder inputs  
▫️ (Random)Mask Token, Patch, PatchesToImages Layer  
▫️ ImageReconstruction callback  


✖️ Not Supported yet (TO DO)  
▫️ Pre-trained model  
▫️ Model test  

Contact on my bio: ga06033@yonsei.ac.kr
