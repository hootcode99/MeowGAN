# MeowGAN
An optimized DCGAN that can generate realistic faces of cats

## Objective
My goal was to learn more about generative deep learning models, PyTorch, & Tensorboard by implementing and optimizing the Deep Convolutional Generative Adversarial Network described by Alec Radford et al. in the original 2016 [paper](https://arxiv.org/abs/1511.06434) 

##  Dataset 
This dataset is an aggregate of 5 other public pet (most commonly cat) color image datasets. The author of this dataset has cropped the images from the other datasets down to the faces of the cats and resized all images to 64x64 pixels. It contains many various colors and breeds of cats. There are 29,843 samples in total. This dataset is distributed as three tarball files on a public GitHub [repository](https://github.com/fferlito/Cat-faces-dataset). I manually downloaded and integrated this dataset using a Python script along with a PyTorch dataset sub-class to process and retrieve samples. Since the dataset is rather small, I manually apply an augmentation using a random horizontal flip with 50% probability. I find this dataset ideal for a training a generative image model. The small images should make the training process fast and prevent GPU memory issues during the training process. The data subjects are familiar and will make it easier to visually determine whether the model is progressing. Below is a sample set of images from the dataset.

![A sample set from the data](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/image_grids/cat_real_grid.png)

## Model

The GAN is a deep learning architecture influenced by the classical AI concepts of Game Playing, the Minimax algorithm, and Actor-Critic methods native to reinforcement learning. 
This model puts two networks, a generator and discriminator, in an adversarial environment with the end-goal of refining an unsupervised deep learning representation of the data 
to be able to generate new samples. The generator network takes in a variable-sized noise vector generated from some random distribution and transforms it to produce a convincing 
artificial sample of the data. The discriminator network takes in real or artificial samples and makes decisions on whether the sample is genuine. The generator is trained using the 
discriminator’s loss on artificial samples. The discriminator is trained as a standard supervised binary classification network. The formula representation of the model from the 
original GAN paper [(Goodfellow et al., 2014)](https://arxiv.org/abs/1511.06434) appears as follows:

![GAN Equation](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/gan_equation.png)

Where D(x) represents the disciminator’s classification of a real sample x, G(z) represents an artificial sample created from noise vector z by the generator, and D(G(z)) rep-
resents the discriminator’s classification of artificial samples. As there are only two classification options, the Binary Cross Entropy loss function is used for both the networks.
A more pragmatic representation (similar to my actual implementation) would be to separate out the losses for discriminator and generator.

### Discriminator Loss
![Discriminator Loss](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/practical_discriminator_loss.png)
### Generator Loss
![Generator Loss](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/practical_generator_loss.png)

Since my task is image generation, I will be implementing and optimizing the Deep Convolutional architecture (DCGAN) introduced in the aformentioned 2016 [paper](https://arxiv.org/abs/1511.06434). 
Rather than fully connected layers, this paper encourages the exclusive use of convolutional layers in discriminator and convolutional transpose layers 
in the generator. The paper advises the use of the Adam optimizer and a learning rate of 0.0002 for both networks. It also recommends avoiding pooling 
layers and instead to simply use strided convolutions. The vanilla structure of the both networks are represented below.

### Generator
![generator](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/generator.png)

### Discriminator
![discriminator](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/discriminator.png)

## Challenges
The GAN and it’s DCGAN variation are particularly unstable models. They are difficult to train because of their adversarial nature. If either the discriminator or the generator 
obtain too much of an advantage, the losses will diverge and the GAN will collapse. This model failure is called Mode Collapse. 
### Example of Mode Collapse (Generator Overpowers Discriminator)
![Mode Collapse](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/image_grids/GAN_mode_collapse.png)

Another issue that I dealt with was non-convergence. The model can stagnate and stop making meaningful progress towards convergence. Another related difficulty in training a DCGAN is 
that there are not many meaningful metrics to determine progress aside from visual inspection of the outputs of the model. The generator and discriminator losses can be helpful when 
determining if things have gone wrong. However, they don’t tell you much of anything (after perhaps the first few epochs) about how the GAN is progressing.

## Optimizations
- Hyperparameter Tuning
- LR Scheduling [(Kun Li and Dae-Ki Kang, 2022)](https://www.mdpi.com/2076-3417/12/3/1191)
- One-sided Label Smoothing [(Tim Salimans et al., 2016)](https://arxiv.org/abs/1606.03498))
- Dropout Regularization [(Phillip Isola et al., 2016)](https://arxiv.org/abs/1611.07004)
- Gaussian Noise [(Martin Arjovsky and Leon Bottou, 2016)](https://arxiv.org/abs/1701.04862) and [(Tim Salimans et al., 2016)](https://arxiv.org/abs/1606.03498))

## Results
With the DC-GAN, determining success is simply a qualitative visual metric. Do the generated images seem convincing? Are they improving as training progresses? Can they be distinguished from the real images? I
would say that with this dataset, I was successful. There is a visible improvement of the optimized model over the vanilla DCGAN implementation. The only distinguishing feature between the optimized model’s 
output and the actual images is some contrast loss. This stems from using approximate means of 0.5 during the (-1, 1) normalization processing on the images. Calculating and leveraging the true means would 
yield more accurate contrast. Despite this, presented in a vacuum, it would still be very difficult to tell the difference. 

### Vanilla DCGAN from Paper
![Vanilla Results](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/image_grids/GAN_cat_vanilla.png)
### Optimized DCGAN
![Optimized Results](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/image_grids/GAN_cat_best.png)
### Real Images (for reference)
![Real Images](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/image_grids/cat_real_grid.png)

## Potential Improvements
- Implement PyTorch's checkpointing feature to improve training progress
- Calculate the true means for the entire image dataset to fix contrast loss
- Convert training code to PyTorch Lightning rather than leveraging Automatic Mixed Precision
- Leverage a larger dataset (either a different subject or aggregate more cat faces on my own)
