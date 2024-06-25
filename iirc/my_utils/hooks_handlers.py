import re
class HooksHandlerViT:
    def __init__(self, model) -> None:
       self.model = model
       self.block_outputs = []
       self.attentions = []
       self.hooks = []
 
       for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'VisionTransformer':
                self.hooks.append(module.register_forward_hook(self.process_outputs))
            if len(re.findall(r'^blocks.\d+$', name)) > 0:
                self.hooks.append(module.register_forward_hook(self.get_block_outputs))
 
    def get_attention(self, module, input, output):
        self.attentions.append(input)
    
    def get_block_outputs(self, module, input, output):
        self.block_outputs.append(output['block_output'])
    
    def reset(self):
        self.attentions = []
        self.block_outputs = []

    def process_outputs(self, module, input, output):
        self.reset()
        return output
        res = {
            # 'attention_masks_heads': [x for x in self.attentions],
            'block_outputs': [x for x in self.block_outputs],
            'output': output
        }
        self.reset()
        return res
