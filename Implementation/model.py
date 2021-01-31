import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def make_layers():
    vgg16_bn = models.vgg16_bn(pretrained=True)
    features = list(vgg16_bn.features.children())
    classifier = list(vgg16_bn.classifier.children())
    
    conv1 = nn.Sequential(*features[:6])
    conv2 = nn.Sequential(*features[7:13])
    conv3 = nn.Sequential(*features[14:23])
    conv4 = nn.Sequential(*features[24:33])
    conv5 = nn.Sequential(*features[34:43])
    
    conv6 = nn.Conv2d(512, 4096, kernel_size=(7, 7))
    conv7 = nn.Conv2d(4096, 4096, kernel_size=(1, 1))
    
    w_conv6 = classifier[0].state_dict()
    w_conv7 = classifier[3].state_dict()
    
    conv6.load_state_dict({'weight':w_conv6['weight'].view(4096, 512, 7, 7), 'bias':w_conv6['bias']})
    conv7.load_state_dict({'weight':w_conv7['weight'].view(4096, 4096, 1, 1), 'bias':w_conv7['bias']})

    return [conv1, conv2, conv3, conv4, conv5, conv6, conv7]


class DeconvNet(nn.Module):
    def __init__(self, num_classes, init_weights):
        super(DeconvNet, self).__init__()
        
        layers = make_layers()
        
        self.conv1 = layers[0]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.conv2 = layers[1]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.conv3 = layers[2]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.conv4 = layers[3]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.conv5 = layers[4]
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.conv67 = nn.Sequential(layers[5], nn.BatchNorm2d(4096), nn.ReLU(),
                                    layers[6], nn.BatchNorm2d(4096), nn.ReLU())
        
        self.deconv67 = nn.Sequential(nn.ConvTranspose2d(4096, 512, kernel_size=7, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU())
        
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=1, stride=1, padding=0))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        original = x
        
        x = self.conv1(x)
        x, p1 = self.pool1(x)
        
        x = self.conv2(x)
        x, p2 = self.pool2(x)
        
        x = self.conv3(x)
        x, p3 = self.pool3(x)
        
        x = self.conv4(x)
        x, p4 = self.pool4(x)
        
        x = self.conv5(x)
        x, p5 = self.pool5(x)
        
        
        x = self.conv67(x)
        x = self.deconv67(x)
        
        x = self.unpool5(x, p5)
        x = self.deconv5(x)
        
        x = self.unpool4(x, p4)
        x = self.deconv4(x)
        
        x = self.unpool3(x, p3)
        x = self.deconv3(x)
        
        x = self.unpool2(x, p2)
        x = self.deconv2(x)
        
        x = self.unpool1(x, p1)
        x = self.deconv1(x)
        
        return x

    def _initialize_weights(self):
        targets = [self.conv67, self.deconv67, self.deconv5, self.deconv4, self.deconv3, self.deconv2, self.deconv1]
        for layer in targets:
            for module in layer:
                if isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)