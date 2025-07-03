import timm
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBackbone(nn.Module):
    def __init__(self, pretrained=True, num_features=512, dropout=0, fp16=False):
        super().__init__()
        # Load ConvNeXt-V2 base
        self.backbone = timm.create_model("convnextv2_base", pretrained=pretrained, num_classes=0, global_pool="")
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Maps to 512-D embedding space, normalized (as expected by ArcFace)
        self.fc = nn.Linear(self.backbone.num_features, num_features)
        self.bn = nn.BatchNorm1d(num_features)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.dropout_layer(x)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x)
        return x




