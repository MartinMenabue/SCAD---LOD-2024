import torch

class Prompt:
    def __init__(self, embed_dim, num_classes) -> None:
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.layers = [0]

        for e in self.layers:
            setattr(self, f'e_p_{e}', torch.nn.Parameter(self.tensor_prompt((num_classes, 2, self.embed_dim))))

        self.prompt_tokens = torch.nn.Parameter(torch.zeros((num_classes, 2, 768)))
        torch.nn.init.uniform_(self.prompt_tokens)
    
    def get_prompts(self, l, labels):
        if labels is None:
            return None
        p = getattr(self, f'e_p_{l}', None)
        if p is None:
            return None
        return p[labels]
    
    def tensor_prompt(self, shape):
        p = torch.FloatTensor(shape)
        torch.nn.init.uniform_(p)
        return p
