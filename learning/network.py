import torch
import torch.nn as nn
from functools import partial

class MLP(nn.Module):
    def __init__(self, dropout, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features)
        self.norm = bn_act_drop(out_features, dropout)

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        return x

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)     

def conv_bn(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(conv3x3(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

def conv_bn_relu(in_channels, out_channels, downsampling, *args, **kwargs):
    return nn.Sequential(conv3x3(in_channels, out_channels, stride=downsampling, *args, **kwargs), nn.BatchNorm2d(out_channels), nn.ReLU())

def bn_act_drop(num_channels, dropout):
    if dropout > 0:
        return nn.Sequential(nn.BatchNorm1d(num_channels), nn.ReLU(), nn.Dropout(dropout))
    else:
        return nn.Sequential(nn.BatchNorm1d(num_channels), nn.ReLU())

class EncoderSimple(nn.Module):
    def __init__(self, img_size=256, latent_size=128):
        super().__init__()
        self.block1 = conv_bn_relu(3,                 latent_size // 8,   downsampling=2)
        self.block2 = conv_bn_relu(latent_size // 8,  latent_size // 4,   downsampling=2)
        self.block3 = conv_bn_relu(latent_size // 4,  latent_size // 2,   downsampling=2)
        self.block4 = conv_bn_relu(latent_size // 2,  latent_size,        downsampling=2)
        self.gap = nn.AvgPool2d(img_size // 2**4)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_size=128, dropout=0.5, n_features=[128, 32], apply_sigmoid=False):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.latent_size = latent_size
        self.norm = bn_act_drop(latent_size * 2, dropout)
        self.mlp  = MLP(dropout=dropout, in_features=latent_size * 2, out_features=n_features[0])
        self.mlp2 = MLP(dropout=0.0, in_features=n_features[0],   out_features=n_features[1])
        self.mlp3 = nn.Linear(n_features[1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, x1, normalize=True):
        x0 = x0.reshape((-1, self.latent_size))
        x1 = x1.reshape((-1, self.latent_size))
        x = torch.cat([x0, x1], dim=1)
        if normalize:
            x = self.norm(x)
        x = self.mlp(x)        
        x = self.mlp2(x)
        x = self.mlp3(x)
        if self.apply_sigmoid:
            return self.sigmoid(x)
        else:
            return x

class GoodnessScoreDecoder(nn.Module):
    def __init__(self, latent_size=128, dropout=0.5, n_features=[128, 32]):
        super().__init__()
        self.latent_size = latent_size
        self.norm = bn_act_drop(latent_size, dropout)
        self.mlp  = MLP(dropout=dropout, in_features=latent_size, out_features=n_features[0])
        self.mlp2 = MLP(dropout=0.0, in_features=n_features[0],   out_features=n_features[1])
        self.mlp3 = nn.Linear(n_features[1], 1)

    def forward(self, x):
        x = x.reshape((-1, self.latent_size))
        x = self.norm(x)
        x = self.mlp(x)        
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x

class ViewModel(nn.Module):
    def __init__(self, mlp_layers, dropout, sigmoid, decoder='binary') -> None:
        super().__init__()
        self.encoder = EncoderSimple()
        if decoder == 'binary':
            self.decoder = Decoder(latent_size=128, dropout=dropout, n_features=mlp_layers, apply_sigmoid=sigmoid)
        elif decoder == 'goodness':
            self.decoder = GoodnessScoreDecoder(latent_size=128, dropout=dropout, n_features=mlp_layers)
    
    def forward(self, x0, x1=None):
        x0 = self.encoder(x0)
        if x1 is None:
            return self.decoder(x0)
        else:
            x1 = self.encoder(x1)
            return self.decoder(x0, x1)

if __name__ == '__main__':
    encoder = EncoderSimple(latent_size=128, img_size=128)
    decoder = GoodnessScoreDecoder(latent_size=128, n_features=[128, 32])
    img = torch.rand((16, 3, 128, 128))
    latent0 = encoder(img)
    latent1 = encoder(img)
    print(latent0.shape)
    pred = decoder(latent0)
    print(pred.shape)