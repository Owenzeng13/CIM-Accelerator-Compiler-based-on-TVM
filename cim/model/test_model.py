# Define your own pytorch model in this file following the pattern as shown

import torch
import torch.nn as nn

# We use modified Yolov3_tiny model as example
# Just simply define the model with ops that our device supports(conv, maxpool, relu, concat and upsample)
# and no quantization step should be defined in this model for we just use it to dump json format.


# !!!Notice!!!: the yolov3_tiny's definition is only a part of the model.
# This is because our device has limited storage, and can only store the activations and weights of that part.

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        # Define the supported ops
        self.conv0 = self.Conv_layer(32, 64, kernel_size=3, padding=1)
        self.conv1 = self.Conv_layer(64,64,kernel_size=3,padding=1)
        self.conv2 = self.Conv_layer(64,64,kernel_size=3,padding=1)
        self.conv3 = self.Conv_layer(64,128,kernel_size=3,padding=1)
        self.conv4 = self.Conv_layer(128,128,kernel_size=3,padding=1)
        self.conv5 = self.Conv_layer(128,256,kernel_size=1,padding=0)
        self.conv6 = self.Conv_layer(256,128,kernel_size=1,padding=0)
        self.conv7 = self.Conv_layer(256,128,kernel_size=3,padding=1)  
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    
    
    def forward(self, x):
        '''
        x = self.conv0(x)
        x = self.maxpool(x)
        
        x = self.conv1(x)
        x = self.maxpool(x)
        '''
        x = self.conv2(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        concat0 = self.maxpool(x)

        x = self.conv4(concat0)
        x = self.conv5(x)
        concat1 = self.conv6(x)
        
        cat_result = torch.cat((concat1,concat0), dim=1)
        result = self.upsample(cat_result)
        '''
        result = self.conv7(result)
        '''
        return result
    
    # for the activation function is always relu after conv, so we mix it together
    def Conv_layer(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size = kernel_size, padding = padding, stride = stride),
                             nn.ReLU()
        )