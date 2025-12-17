# Sketch_to_Image_cLDM
Creating a Conditional Latent Diffusion Model to convert sketches to high resolution real images. We transfer our high resolution images to a lower dimension latent space on which we train our U-Net. 

### Architecture

We use multi-scale spatial conditioning to reinforce fine sketch details and to ensure the structural integrity of the sketch is intact.


### Features
1. Unconditional generation - we apply a 10% DropOut rate, whereby the entire sketch is dropped. This allows our model to learn unconditional generation, ie generating an image without a sketch.  
2. High Resolution - our model produces high resolution 256x256 real images. 

### Hardware Requirements
This notebook has been implemented in a Colab Environment. We recommend using an A100 or L4 GPU for training.
