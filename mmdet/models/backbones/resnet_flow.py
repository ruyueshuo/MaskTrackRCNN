from ..registry import BACKBONES
from .resnet import ResNet


@BACKBONES.register_module
class ResNetFlow(ResNet):

    def __init__(self):
        super(ResNetFlow, self).__init__()
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out0 = x
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0], out0
        else:
            return tuple(outs), out0