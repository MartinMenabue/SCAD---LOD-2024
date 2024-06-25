import torch
from prompt import Prompt
from vit_prompt import vit_base_patch16_224_class_prompt

class PromptClass:
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.prompt = Prompt(num_classes)
        self.net = vit_base_patch16_224_class_prompt(pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.net(x, prompt=self.prompt)
