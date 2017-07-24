# **Behavioral Cloning** 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/steering_hist.png "Distribution of Steering Angles in Training Data"
[image2]: ./examples/train_val_loss.png "Training and Validation loss"

---

### Summary of Training Data

The driving log data consists of 8036 rows, each row has 3 images recorded by 3 virtual cameras on the vehicle: center, left and right.
Here is a histogram to see the distribution of the steering angles. It is a little bit imbalanced because the vehicle drives
counter-clockwise on the track.

![alt text][image1]

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My initial model is LeNet but with continuous output. The initial data was collected by myself driving in the simulator
for 2 laps using only keyboard. The result wasn't very good because as some fellow students pointed out, keyboard's 
abrupted movements are not smooth and suitable for the model to learn from. The steering should be as smooth as possible, so
a game controller is recommended over the keyboard. But since I don't have a game controller at this time, I tried the sample
data instead. LeNet performed pretty well on the sample data. It was not bad for the straight or slightly curved lanes but when 
it reached sharper curves or the bridge where the ground has a different texture, it drove the car onto the curb and got stuck.

My next attempts were to augment the data as suggested by the lecture, and adopt a well-tested architecture in the literature,
which is NVIDIA's "End to End Learning for Self-Driving Cars" 
[paper] (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (model.py lines 19-60) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 20). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 64-70).
The number of training samples for each epoch is 20000, and the number of validation samples is set to 6400. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 61). It was trained for
4 epochs because further training didn't reduce the loss by a noticeable amount.

#### 4. Appropriate training data

Initially the data I collected was via keyboard. It was quite abrupt and jerky because it was hard to maintain a smooth input
with the keyboard. As expected, the result was also jerky. The car frequently adjusted steering angles even on straight lanes.

This is a regression task using convolutional neural networks,
hence there is an important note for training these models - "garbage in, garbage out". With the sample data and data augmentation,
I was able to improve the model output by a lot. I trained for 4 epochs since further training didn't appear to be very helpful. 
The final validation loss was 0.0102. 

The following diagram shows the training and validation losses in the training process over the number of epochs,

![alt text][image2]

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Following the suggestion of the lecture, I tried the LeNet model first and then moved to the NVidia model.

In order to gauge how well the model was working, I implemented the image batch generator to populate the training and 
validation sets. I found that my first model had a low mse loss on the training set but a high loss on the validation set. 
This implied that the model was overfitting. So I added data augmentation to generate more training samples. Since the car 
drives counter-clockwise on the track, the left and right steering data are not balanced. So I augmented the image data by flipping them
to horizontally symmetric ones randomly at a probability of 0.5. I also cropped and resized the images to only focus on
the road instead of the irrelevant parts.

The final step was to run the simulator to see how well the car was driving around track one. Without data augmentation, the car tended
to steer to the right more because of the imbalance left and right steering angles in the data. With random flipping, the car drove significantly 
better. 

On the model side, the NVidia paper provided a powerful solution to this problem. The paper described the architecture as follows.
The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers.
The first layer of the network performs image normalization. Performing normalization in the network allows the 
normalization scheme to be altered with the network architecture and to be accelerated via GPU processing.
The convolutional layers were designed to perform feature extraction and were chosen empirically through a series of 
experiments that varied layer configurations. The model used strided convolutions in the first three convolutional layers with a 
2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.
It follows the five convolutional layers with three fully connected layers leading to an output control value which is 
the inverse turning radius. The fully connected layers are designed to function as a controller for steering.

The final model appeared to be easy to train and effective. At the end of the process, 
the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The architecture is shown below.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 32, 32, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 32, 32, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 31, 31, 24)    0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 16, 16, 36)    21636       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 16, 16, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 15, 15, 36)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 8, 48)      43248       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 8, 8, 48)      0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 7, 7, 48)      0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 7, 7, 64)      27712       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 7, 7, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 6, 6, 64)      0           activation_4[0][0]               
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 6, 6, 64)      36928       maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 6, 6, 64)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 5, 5, 64)      0           activation_5[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1600)          0           maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          1863564     flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      activation_6[0][0]               
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        activation_7[0][0]               
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         activation_8[0][0]               
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 10)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          activation_9[0][0]               
====================================================================================================
Total params: 2,116,983
Trainable params: 2,116,983
Non-trainable params: 0
```

#### 3. End Result

I recorded the final result in autonomous mode into a mp4 file and uploaded it 
[here](https://www.youtube.com/watch?v=pDdN28Bdm-o&feature=youtu.be).


