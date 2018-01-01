# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of "good" driving behavior.
* Build, a convolution neural network in Keras that predicts steering angles from images.
* Train and validate the model with a training and validation set.
* Test that the model successfully drives around track one without leaving the road.
* Summarize the results with a written report.


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model.
* drive.py for driving the car in autonomous mode.
* model.h5 containing a trained model.
* writeup_report.md summarizing the results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a Keras implementation of [NVIDIA's self-driving car model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with slight modifications. The base model was adopted from [Alex Hagiopol's implementation](https://github.com/alexhagiopol/end-to-end-deep-learning/blob/master/architecture.py).

The model starts with cropping the images from the top and the bottom, in order to make sure that the hood of the car and the brids in the sky don't affect the results of our model. The model then proceeds as follows:

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model then adds 5 convolutional layers, preceeded and succeeded with a 1x1 convolutional layer on each side. After that, the model flattens the results, applies a 50% dropout and begins a series of 5 fully-connected layers to get to our single output (there's also another dropout there somewhere).

#### 2. Attempts to reduce overfitting in the model

I added a couple of dropout layers, and limited the number of training epochs to 4.

#### 3. Model parameter tuning

The model used Keras's Adam optimizer, and the learning rate was not tuned manually. I found that a batch size of 512 was appropriate, and that training for more than 4 epochs doesn't yeild better results.

I experimented with a number of different filter sizes and strides for the convolutional layers, but I found that sticking with the default &mdash; 5x5 filters with a stride of 2x2 for the first 3 layers and 3x3 with a 1x1 stride for the next 2 layers &mdash; to be the best.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (in both directions), a bit of recovering from the left and right sides of the road, and a some extra corner driving data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In this project I wanted to see how a car would learn to drive itself using only regression network and nothing else.

My first step was to use the NVIDIA self-driving model with no modifications. I quickly found out that the model required some tuning, a bit more data, and some luck.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the longer the model spent trainig (more epochs) the lower the mean squared error gets on the training set, but there's always a point in time where the validation set results would start becoming worse, which indicates overfitting.

To combat the overfitting, I modified the model so that it uses less epochs, but I also added dropout layers in order to combat the rising loss values in the validation set.

Then I added 1x1 convolutional layers, before and after the original convolutional layers in the model. The first layer's job is to reduce the 3 color channels (RGB) into a single, smooth channel. The latter's job is to slightly increase the dimensionality of the hidden features before the fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. It took a little while to convince the car to stay on track, but adding extra data, tuning my parameters, and doing a bit of data augmentation did the trick.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py: train_nvidia()) consisted of a convolution neural network with the following layers and layer sizes:

<b>FIXME</b>



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior (as much a mouse would allow me to), I first recorded two laps on track one using center lane driving, one in each direction of the track, and I kept adding laps, corners, and recovery cases on top of that data every time I notice that the car is missing an important bit of learning.

I didn't touch track 2, simply because it's difficult to drive on it, and I worried that I might contaminate my data with lousy driving.

To augment the data sat, I also flipped images and angles thinking that this would teach the car that the world isn't always biased to the left.

After the collection process, I had 10020 data points (before mirroring, and considering only the middle camera).


I finally randomly shuffled the data set and put 10% of the data into a validation set. Generally, 20-30% is the better ratio, but my real validation set was driving on the actual track. In reality I don't have a test set, since my track options are limited.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by basic trial and error, where the validation set accuracy remains constant, or increases after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
