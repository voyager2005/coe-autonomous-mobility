from .validators import Validator
from .einops_utils import check_extra_arguments, get_additional_args
from .transformations import input_based_transformation, update_input_tokens_mapping, Output_Transformations
import torch

def rearrange(tensor, pattern, **kwargs):
    """
    Native PyTorch bypass for einops.rearrange.
    Keeps everything on the GPU using native reshaping.
    """
    clean_pattern = pattern.replace(' ', '')
    
    # ---------------------------------------------------------
    # Pattern 1: Flattening the Image to Tokens
    # Example: 'b c h w -> b (h w) c'
    # ---------------------------------------------------------
    if clean_pattern == 'bchw->b(hw)c':
        b, c, h, w = tensor.shape
        # Move channel 'c' to the end, then flatten 'h' and 'w'
        return tensor.permute(0, 2, 3, 1).view(b, h * w, c)

    # ---------------------------------------------------------
    # Pattern 2: Reconstructing the Image from Tokens
    # Example: 'b (h w) c -> b c h w'
    # ---------------------------------------------------------
    elif clean_pattern == 'b(hw)c->bchw':
        # This requires passing h and w as kwargs
        h = kwargs.get('h')
        w = kwargs.get('w')
        if h is None or w is None:
            raise ValueError("rearrange: 'h' and 'w' must be provided for reconstruction.")
        b, hw, c = tensor.shape
        # Unflatten to h/w, then move channel 'c' back to dimension 1
        return tensor.view(b, h, w, c).permute(0, 3, 1, 2)
        
    # ---------------------------------------------------------
    # Pattern 3: Attention Head Splitting/Merging
    # Used inside the Multi-Head Attention blocks
    # ---------------------------------------------------------
    elif clean_pattern == 'bn(hd)->bnhd':
        # Splitting the embedding dimension 'd' into heads 'h'
        h_heads = kwargs.get('h')
        b, n, hd = tensor.shape
        d = hd // h_heads
        return tensor.view(b, n, h_heads, d)
        
    elif clean_pattern == 'bnhd->bn(hd)':
        # Merging heads back into the embedding dimension
        b, n, h_heads, d = tensor.shape
        return tensor.reshape(b, n, h_heads * d)
        
    elif clean_pattern == 'bnhd->bhnd':
        # Transposing sequence 'n' and heads 'h'
        return tensor.permute(0, 2, 1, 3)
# ---------------------------------------------------------
    # Pattern 4: Cross-Attention Key/Value Reshaping
    # Example: 'b (head c) h w -> b head c (h w)'
    # ---------------------------------------------------------
    elif clean_pattern == 'b(headc)hw->bheadc(hw)':
        head = kwargs.get('head')
        if head is None:
            raise ValueError("rearrange: 'head' must be provided for Key/Value reshaping.")
        
        b, channels, h, w = tensor.shape
        c = channels // head
        
        # We use .reshape() to split the channels and flatten the spatial dimensions
        return tensor.reshape(b, head, c, h * w)
    # ---------------------------------------------------------
    # Pattern 5: Cross-Attention Output Merging
    # Example: 'b head c (h w) -> b (head c) h w'
    # ---------------------------------------------------------
    elif clean_pattern == 'bheadc(hw)->b(headc)hw':
        h = kwargs.get('h')
        w = kwargs.get('w')
        if h is None or w is None:
            raise ValueError("rearrange: 'h' and 'w' must be provided for output merging.")
        
        b, head, c, hw = tensor.shape
        
        # We merge the 'head' and 'c' dimensions by multiplying them, 
        # and unroll the flattened 'hw' back into 'h' and 'w'.
        return tensor.reshape(b, head * c, h, w)
    # ---------------------------------------------------------
    # The Fail-Fast Trap
    # ---------------------------------------------------------
    else:
        raise NotImplementedError(
            f"Native Bypass Trap: The rearrange pattern '{pattern}' was not recognized. "
            f"Please share this pattern so we can add the PyTorch equivalent!"
        )

def repeat(tensor, pattern, **kwargs):
    """
    Native PyTorch bypass for einops.repeat.
    Handles the specific tensor expansions needed for OneRestore.
    """
    # Strip spaces for easier matching
    clean_pattern = pattern.replace(' ', '')
    
    # ---------------------------------------------------------
    # Pattern 1: The ViT Batch Expansion (Most Common)
    # Examples: '1 n d -> b n d' or '() n d -> b n d'
    # ---------------------------------------------------------
    if clean_pattern in ['1nd->bnd', '()nd->bnd', '1cd->bcd', '1hwc->bhwc', '1chw->bchw']:
        b = kwargs.get('b')
        if b is None:
            raise ValueError("Custom repeat: Batch size 'b' must be provided.")
        
        # We use .expand() instead of .repeat() because it is 
        # infinitely more memory efficient. It creates a "view" 
        # without duplicating the actual data in your VRAM.
        return tensor.expand(b, *tensor.shape[1:])

    # ---------------------------------------------------------
    # Pattern 2: Channel Expansion
    # Example: 'b 1 h w -> b c h w' (Expanding grayscale to RGB)
    # ---------------------------------------------------------
    elif clean_pattern == 'b1hw->bchw':
        c = kwargs.get('c')
        return tensor.expand(-1, c, -1, -1)
# ---------------------------------------------------------
    # Pattern 3: Multi-Head Cross-Attention Expansion
    # Example: 'b l -> b head c l'
    # ---------------------------------------------------------
    elif clean_pattern == 'bl->bheadcl':
        # Expanding a 2D tensor (batch, length) into a 4D multi-head tensor
        head = kwargs.get('head')
        c = kwargs.get('c')
        if head is None or c is None:
            raise ValueError("repeat: 'head' and 'c' must be provided for attention expansion.")
        
        # Add new dimensions for 'head' and 'c' (at index 1 and 2), 
        # then expand them to the required sizes.
        return tensor.unsqueeze(1).unsqueeze(2).expand(-1, head, c, -1)
    # ---------------------------------------------------------
    # The "Fail-Fast" Trap
    # ---------------------------------------------------------
    else:
        # If OneRestore uses a weird pattern we didn't predict, 
        # this trap will catch it and tell us exactly what it is 
        # so we can write the 1-line PyTorch fix for it.
        raise NotImplementedError(
            f"Native Bypass Trap: The pattern '{pattern}' was not recognized. "
            f"Please share this pattern so we can add the PyTorch equivalent!"
        )