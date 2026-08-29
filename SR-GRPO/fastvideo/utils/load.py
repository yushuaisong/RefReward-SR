#This code file is from [https://github.com/hao-ai-lab/FastVideo], which is licensed under Apache License 2.0.

from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock)


def get_no_split_modules(transformer):
    if isinstance(transformer, FluxTransformer2DModel):
        return (FluxTransformerBlock, FluxSingleTransformerBlock)
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")
