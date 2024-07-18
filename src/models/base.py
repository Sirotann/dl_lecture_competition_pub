import torch
from torch import nn

class build_resnet_block(nn.Module):
    """
    a resnet block which includes two general_conv2d
    """
    def __init__(self, channels, layers=2, do_batch_norm=False, dropout_prob=0.5,bottleneck=False):
        super(build_resnet_block,self).__init__()
        self._channels = channels
        self._layers = layers
        self.dropout = nn.Dropout(p=dropout_prob)###
        self.bottleneck = bottleneck###
        
        #self.res_block = nn.Sequential(*[general_conv2d(in_channels=self._channels,
                                             #out_channels=self._channels,
                                             #strides=1,
                                             #do_batch_norm=do_batch_norm) for i in range(self._layers)])
        if self.bottleneck:
            self.res_block = nn.Sequential(
                nn.Conv2d(channels, channels // 4, kernel_size=1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, kernel_size=1),
                nn.BatchNorm2d(channels)
            )
        else:
            self.res_block = nn.Sequential(
                *[general_conv2d(in_channels=self._channels, out_channels=self._channels, strides=1, do_batch_norm=do_batch_norm) for _ in range(self._layers)]
            )


    def forward(self,input_res):
        inputs = input_res.clone()
        input_res = self.res_block(input_res)
        input_res = self.dropout(input_res)###
        return input_res + inputs

class upsample_conv2d_and_predict_flow(nn.Module):
    """
    an upsample convolution layer which includes a nearest interpolate and a general_conv2d
    """
    def __init__(self, in_channels, out_channels, ksize=3, do_batch_norm=False):
        super(upsample_conv2d_and_predict_flow, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._ksize = ksize
        self._do_batch_norm = do_batch_norm

        self.general_conv2d = general_conv2d(in_channels=self._in_channels,
                                             out_channels=self._out_channels,
                                             ksize=self._ksize,
                                             strides=1,
                                             do_batch_norm=self._do_batch_norm,
                                             padding=0)
        
        self.pad = nn.ReflectionPad2d(padding=(int((self._ksize-1)/2), int((self._ksize-1)/2),
                                        int((self._ksize-1)/2), int((self._ksize-1)/2)))

        self.predict_flow = general_conv2d(in_channels=self._out_channels,
                                           out_channels=2,
                                           ksize=1,
                                           strides=1,
                                           padding=0,
                                           activation='tanh')

    def forward(self, conv):
        shape = conv.shape
        conv = nn.functional.interpolate(conv,size=[shape[2]*2,shape[3]*2],mode='nearest')
        conv = self.pad(conv)
        conv = self.general_conv2d(conv)

        flow = self.predict_flow(conv) * 256.
        
        return torch.cat([conv,flow.clone()], dim=1), flow

def general_conv2d(in_channels,out_channels, ksize=3, strides=2, padding=1, do_batch_norm=False, activation='relu', dilation=1):
    """
    a general convolution layer which includes a conv2d, a relu and a batch_normalize
    """
    if activation == 'relu':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding, dilation=dilation),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding, dilation=dilation),
                nn.ReLU(inplace=True)
            )
    elif activation == 'tanh':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding, dilation=dilation),
                nn.Tanh(),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding, dilation=dilation),
                nn.Tanh()
            )
    return conv2d

class GeneralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, strides=2, padding=1, do_batch_norm=False, activation='relu'):
        super(GeneralConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=strides, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if do_batch_norm else nn.Identity()
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Tanh()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class UpsampleConv2dAndPredictFlow(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, do_batch_norm=False):
        super(UpsampleConv2dAndPredictFlow, self).__init__()
        self.conv = GeneralConv2d(in_channels, out_channels, ksize, strides=1, padding=1, do_batch_norm=do_batch_norm)
        self.predict_flow = GeneralConv2d(out_channels, 2, ksize=1, strides=1, padding=0, activation='tanh')

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        flow = self.predict_flow(x) * 256.0
        return x, flow

class UNetWithMultiScaleLoss(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithMultiScaleLoss, self).__init__()
        self.encoder1 = GeneralConv2d(in_channels, 64, ksize=3, strides=2, padding=1)
        self.encoder2 = GeneralConv2d(64, 128, ksize=3, strides=2, padding=1)
        self.encoder3 = GeneralConv2d(128, 256, ksize=3, strides=2, padding=1)
        
        self.decoder1 = UpsampleConv2dAndPredictFlow(256, 128, ksize=3)
        self.decoder2 = UpsampleConv2dAndPredictFlow(128, 64, ksize=3)
        self.decoder3 = UpsampleConv2dAndPredictFlow(64, out_channels, ksize=3)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        dec1, flow1 = self.decoder1(enc3)
        dec2, flow2 = self.decoder2(dec1)
        _, flow3 = self.decoder3(dec2)
        
        return flow1, flow2, flow3
