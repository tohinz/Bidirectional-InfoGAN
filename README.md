# Bidirectional-InfoGAN

For further details on the network architectures and the hyperparameters used during the experiments see: [Architectures and Hyperparameters](./architectures/network-architectures.pdf)

Contents:
* [Additional Results](#additional-results): Additional random samples from the respective test sets according to certain characteristics represented by the values in c.
    * [MNIST Data Set](#mnist-data-set)
    * [CelebA Data Set](#celeba-data-set)
    * [SVHN Data Set](#svhn-data-set)
* [Images from the Paper](#images-from-the-paper): The original images used in the paper.
    * [MNIST Images](#mnist-images)
    * [CelebA Images](#celeba-images)
    * [SVHN Images](#svhn-images)

## Additional Results
### MNIST Data Set
Additional samples according to the individual values of the categorical variable c<sub>1</sub>:
![](./imgs/mnist/mnist_cat_c1.png)

Additional samples from the continuous variable c<sub>2</sub> (stroke width):
uneven rows show the samples according to the minimum values, even rows show samples according to the maximum values of c<sub>2</sub> of each categorical value
![](./imgs/mnist/mnist_cont_c2.png)

Additional samples from the continuous variable c<sub>3</sub> (digit rotation):
uneven rows show the samples according to the minimum values, even rows show samples according to the maximum values of c<sub>3</sub> of each categorical value
![](./imgs/mnist/mnist_cont_c3.png)

### CelebA Data Set
#### Categorical Variables
Here we show images that are sampled from the CelebA test set according to a categorical value of the variables c<sub>1</sub>, ..., c<sub>4</sub>.

Images with high confidence in the presence of glasses:
![](./imgs/celeba/categorical/celeba_glasses.png)

Images with high confidence in the presence of hats:
![](./imgs/celeba/categorical/celeba_hats.png)

Images with high confidence in blond hair:
![](./imgs/celeba/categorical/celeba_blond.png)

Images with high confidence in a person looking to the right:
![](./imgs/celeba/categorical/celeba_looking_right.png)

Images with high confidence in a person looking to the left:
![](./imgs/celeba/categorical/celeba_looking_left.png)

Images with high confidence in a person with a darker skin tone:
![](./imgs/celeba/categorical/celeba_dark_skin.png)

Images with high confidence in a person with their mouth open:
![](./imgs/celeba/categorical/celeba_mouth_open.png)

Images with high confidence in blue background:
![](./imgs/celeba/categorical/celeba_blue.png)

Images with high confidence in red background:
![](./imgs/celeba/categorical/celeba_red.png)

#### Continuous Variables
Here we show images that are sampled from the CelebA test set according to their value (minimum and maximum) for the continuous variables c<sub>5</sub>, ..., c<sub>8</sub>. In each image the first two rows show images where the value of the given continuous variable is small, while the second two rows show images where the continuous variable's value is large.

From not smiling to smiling:  
![](./imgs/celeba/continuous/celeba_mood.png)

From light hair to darker hair:  
![](./imgs/celeba/continuous/celeba_haircolor.png)

From light background to darker background:
![](./imgs/celeba/continuous/celeba_background.png)


### SVHN Data Set

## Images from the Paper
### MNIST Images
![](./imgs/imgs_paper/mnist/mnist_cat.png)
![](./imgs/imgs_paper/mnist/mnist_cont.png)

### CelebA Images
![](./imgs/imgs_paper/celeba/celeba_cat.png)

### SVHN Images
![](./imgs/imgs_paper/svhn/svhn_cat.png)
