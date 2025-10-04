##### model.py

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


# 1. Define the custom model
class ResNet18(nn.Module):
    def __init__(self, descriptor_size=193):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet-18

        # Freeze the ResNet-18 feature layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Get the number of features from the final convolutional block
        num_ftrs = self.resnet.fc.in_features  # 512 for ResNet-18

        # Remove the final fully connected layer from the ResNet model
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # A separate fc_layers that takes the combined features
        self.fc_layers = nn.Sequential(
            nn.Linear(num_ftrs + descriptor_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x, descriptors):
        resnet_features = self.resnet(x)  # The output 'out' shape [batch_size, num_ftrs, 1, 1]
        resnet_features = resnet_features.view(resnet_features.size(0),
                                               -1)  # Flatten the features to [batch_size, num_ftrs]
        combined_features = torch.cat((resnet_features, descriptors), dim=1)  # Concatenate descriptors
        outputs = self.fc_layers(combined_features)

        return outputs


# # Check the shape
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNet18()
# model.to(device)
# print(model)
# # e.g.
# x = torch.randn(2, 3, 224, 224).to(device)
# d = torch.randn(2, 193).to(device)
# print("Output shape:", model(x, descriptors=d).shape)

def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate and print the total parameters
if __name__ == '__main__':
    my_deep_model = ResNet18()
    total_params = count_parameters(my_deep_model)
    print(f"Total trainable parameters: {total_params:,}")