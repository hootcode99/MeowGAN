# MeowGAN
An optimized DCGAN that can generate realistic faces of cats

## Objective
My goal was to learn more about generative deep learning models, PyTorch, & Tensorboard by implementing and optimizing the Deep Convolutional Generative Adversarial Network described by Alec Radford et al. in the original 2016 [paper](https://arxiv.org/abs/1511.06434) 

##  Dataset 
This dataset is an aggregate of 5 other public pet (most commonly cat) color image datasets. The author of this dataset has cropped the images from the other datasets down to the faces of the cats and resized all images to 64x64 pixels. It contains many various colors and breeds of cats. There are 29,843 samples in total. This dataset is distributed as three tarball files on a public GitHub [repository](https://github.com/fferlito/Cat-faces-dataset). I manually downloaded and integrated this dataset using a Python script along with a PyTorch dataset sub-class to process and retrieve samples. Since the dataset is rather small, I manually apply an augmentation using a random horizontal flip with 50% probability. I find this dataset ideal for a training a generative image model. The small images should make the training process fast and prevent GPU memory issues during the training process. The data subjects are familiar and will make it easier to visually determine whether the model is progressing. Below is a sample set of images from the dataset.

![A sample set from the data](https://github.com/hootcode99/MeowGAN/blob/main/GAN/imgs/image_grids/cat_real_grid.png)

## Model

The GAN is a deep learning architecture influenced by the classical AI concepts of Game Playing, the Minimax algorithm, and Actor-Critic methods native to reinforcement learning. 
This model puts two networks, a generator and discriminator, in an adversarial environment with the endgoal of refining an unsupervised deep learning representation of the data 
to be able to generate new samples. The gen-erator network takes in a variable-sized noise vector generated from some random distribution and transforms it toproduce a convincing 
artificial sample of the data. The discriminator network takes in real or artificial samples andmakes decisions on whether the sample is genuine. The generator is trained using the 
discriminator’s loss on artifi-cial samples. The discriminator is trained as a standard supervised binary classification network. The formula representation of the model from the 
original GAN paper (Goodfellow et al., 2014) appears as follows:


Where D(x) represents the disciminator’s classification of a real sample x, G(z) represents an artificial sample created from noise vector z by the generator, and D(G(z)) rep-
resents the discriminator’s classification of artificial samples. As there are only two classification options, the Binary Cross Entropy loss function is used for both the networks.
A more pragmatic representation (similar to our actual implementation) would be to separate out the losses for discriminator and generator.



Since our task is image generation, we will be implementing and optimizing the Deep Convolutional architecture (DC-GAN) introduced in the 2016 paper. 
Rather than fully connected layers, this paper encourages the exclusive use of convolutional layers in discriminator and convolutional transpose layers 
in the generator. The paper recommends the use of the Adam optimizer and a learning rate of 0.0002 for both networks. It also recommends avoiding pooling 
layers and instead to simply use strided convolutions. The vanilla structure of the both networks are represented below.

