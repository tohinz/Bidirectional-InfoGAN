# Bidirectional-InfoGAN

The work presented here was done at the [Knowledge Technology Research Group](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/ "Knowledge Technology Research Group") at the University of Hamburg and submitted as a conference contribution to the [ESANN 2018](https://www.elen.ucl.ac.be/esann/).

Contents:
* [Architectures and Hyperparameters](#architectures-and-hyperparameters)
* [Additional Results](#additional-results)
    * [MNIST Data Set](#mnist-data-set)
        * [Categorical Variables](#categorical-variables)
        * [Continuous Variables](#continuous-variables)
    * [CelebA Data Set](#celeba-data-set)
        * [Categorical Variables](#categorical-variables-1)
        * [Continuous Variables](#continuous-variables-1)
    * [SVHN Data Set](#svhn-data-set)
* [Images from the Paper](#images-from-the-paper)
     * [MNIST Images](#mnist-images)
     * [CelebA Images](#celeba-images)
     * [SVHN Images](#svhn-images)


## Architectures and Hyperparameters
For further details on the network architectures and the hyperparameters used during the experiments see [Architectures and Hyperparameters](./architectures/network-architectures.pdf).

## Additional Results
### MNIST Data Set
#### Categorical Variables
Additional samples according to the individual values of the categorical variable c<sub>1</sub>:
![](./imgs/mnist/categorical/mnist_cat_c1.png)

#### Continuous Variables
Additional samples from the continuous variable c<sub>2</sub> (stroke width):
uneven rows show the samples according to the minimum values, even rows show samples according to the maximum values of c<sub>2</sub> of each categorical value
![](./imgs/mnist/continuous/mnist_cont_c2.png)

Additional samples from the continuous variable c<sub>3</sub> (digit rotation):
uneven rows show the samples according to the minimum values, even rows show samples according to the maximum values of c<sub>3</sub> of each categorical value
![](./imgs/mnist/continuous/mnist_cont_c3.png)

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
![](./imgs/celeba/continuous/celeba_smile.png)

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
