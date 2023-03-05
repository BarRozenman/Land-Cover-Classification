import segmentation_models_pytorch as smp
from torch import nn



class Net(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=7,  # model output channels (number of classes in your dataset)
            activation="softmax",
        ).to(device)

    def forward(self, x):
        res = self.model(x.float())
        return res
