from transformers import AutoImageProcessor, ResNetForImageClassification
import torch.nn as nn

class Custom_resnet(nn.Module):
    def __init__(self, pretrained):
        super(Custom_resnet, self).__init__()
        self.pretrained = pretrained
        new_layers = nn.Linear(1000, 2)
        pretrained.classifier = nn.Sequential(pretrained.classifier, new_layers)

    def forward(self, x):
        x = self.pretrained(x)
        return x

if __name__ == "__main__":
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    new_model = Custom_resnet(model)
    for param in new_model.parameters():
        param.requires_grad = False
    for name, param in new_model.named_parameters():
        if name.split('.')[1] == 'classifier':
            param.requires_grad = True
    for name, param in new_model.named_parameters():
        print(name, param.requires_grad)
