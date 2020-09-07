# oct-patchbased-cgan
Code for the paper "Constructing synthetic chorio-retinal patches using generative adversarial networks"

# Dependencies
* Python 3.6.4
* Keras 2.2.4
* tensorflow-gpu 1.8.0
* h5py

1. Train a conditional GAN using 32x32 patches using *cgan_32x32_patchbased.py*. Load data using the *load_data* function.
2. Evaluate trained generators of a GAN, using the Frechet Inception Distance (FID), using *evalfid_32x32_patchbased.py* and by specifying the folder containing the generators as *load_path*
3. Construct synthetic patches using *cgan_32x32_patchbased_genfid.py*. Save the generated images as required.
