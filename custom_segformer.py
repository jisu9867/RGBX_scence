from transformers import AutoImageProcessor, SegformerForImageClassification, SegformerForSemanticSegmentation
import torch.nn as nn

class Custom_segformer(nn.Module):
    def __init__(self, pretrained):
        super(Custom_segformer, self).__init__()
        self.pretrained = pretrained
        new_layers = nn.Conv2d(1000, 2, kernel_size=(1,1), stride=(1,1))
        pretrained.decode_head.classifier = nn.Sequential(pretrained.decode_head.classifier, new_layers)

    def forward(self, x):
        x = self.pretrained(x)
        return x

if __name__ == "__main__":
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b4")
    new_model = Custom_segformer(model)
    print(new_model)
    for param in new_model.parameters():
        param.requires_grad = False
    for name, param in new_model.named_parameters():
        if name.split('.')[2] == 'classifier':
            param.requires_grad = True

    for name, param in new_model.named_parameters():
        print(name, param.requires_grad)