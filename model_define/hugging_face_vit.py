from transformers import ViTConfig, ViTModel
import torch.nn as nn
# Initializing a ViT vit-base-patch16-224 style configuration
configuration = ViTConfig()

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
vit_model = ViTModel(configuration)

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=10):
        super(ViTForImageClassification, self).__init__()
        self.vit = vit_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels
    def forward(self, pixel_values, interpolate_pos_encoding=False):
        outputs = self.vit(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)
        return logits