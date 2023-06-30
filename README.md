# ERA-S8

# Custom CNN model to classify images of Cifar10 Dataset

Sample images: 

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/122eba70-a598-4add-a267-946aaff9da97)


# Model with Group Norm

Using group = 4

### The model network summary :

we have made use of padding to keep the channel size consistent which allows to have the skip connections.There are 2 skip connections in this network.The total number of parameters are 48552

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/a4f37903-1503-483c-8b6a-32aa4362f65f)


### Results: 
Train set: Accuracy : 78.36%
Test set:  Accuracy: 7784/10000 (77.84%)

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/c599a620-b093-44de-bc02-b3d12cadc006)


### Incorrect predictions samples :

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/4fcb8290-bc22-40e5-9220-cdc503bd3bae)



# Model with Layer Norm

### The model network summary :

we have made use of padding to keep the channel size consistent which allows to have the skip connections.There are 2 skip connections in this network.The total number of parameters are 48072

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/facb418c-ac6f-4a71-8de8-dcb8b428c6f0)



### Results: 
Train set: Accuracy : 77.25%
Test set:  Accuracy: 7764/10000 (77.64%)

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/30975dd2-12d4-4b7b-8ca8-da63aaadaa25)


### Incorrect predictions samples :

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/da173db8-3917-4415-b848-9c7ac455e30c)



# Model with Batch Norm

### The model network summary :

we have made use of padding to keep the channel size consistent which allows to have the skip connections.There are 2 skip connections in this network.The total number of parameters are 48552

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/16a625bc-07af-4857-ab03-45d0f7fdd014)




### Results: 
Train set: Accuracy : 78.85%
Test set:  Accuracy: 8066/10000 (80.66%)

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/8bdcff9e-e175-46b6-b83e-074d5b6c9186)


### Incorrect predictions samples :

![image](https://github.com/amitdoda1983/ERA-S8/assets/37932202/28b43c28-dac7-4bf4-924c-1e8be5876e41)


# Conclusion on Normaliztion:
As observed, the layer and group norm both gave almost similar convergence with a final test accuracy in the range of 77 %
The batch norm is a clear winner with faster convergence rate and final test accuracy of 80 %
Between layer norm and group norm, group norm seemed better as it had better and consistent results.
