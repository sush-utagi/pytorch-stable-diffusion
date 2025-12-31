"""
Attention Capture Module for Visualizing Cross-Attention to Text Tokens

This module provides utilities to capture and aggregate cross-attention weights
from the U-Net during diffusion model generation. By capturing these weights,
we can visualize which text tokens the model is focusing on at different
stages of the denoising process.

Architecture:
- CrossAttention layers attend from image features (queries) to text embeddings (keys/values)
- Attention weights shape: (Batch, Heads, Seq_Len_Image, Seq_Len_Text)
- Seq_Len_Text = 77 (CLIP's fixed sequence length)
- We aggregate across heads and spatial dimensions to get per-token importance

Author: Honours Research Project
"""

import torch
from contextlib import contextmanager
from typing import Dict, List, Optional
import numpy as np


# Global storage for attention weights
_ATTENTION_STORE = {}
_CAPTURE_ENABLED = False


@contextmanager
def capture_attention():
    """
    Context manager to enable attention weight capture.

    Usage:
        with capture_attention():
            # Run model forward pass
            output = model(input)

        # Get captured attention
        attn_weights = get_attention_maps()
    """
    global _CAPTURE_ENABLED, _ATTENTION_STORE

    _CAPTURE_ENABLED = True
    _ATTENTION_STORE = {}

    try:
        yield
    finally:
        _CAPTURE_ENABLED = False


def is_capture_enabled() -> bool:
    """Check if attention capture is currently enabled."""
    return _CAPTURE_ENABLED


def store_attention(layer_name: str, attention_weights: torch.Tensor):
    """
    Store attention weights from a layer.

    Args:
        layer_name: Identifier for the layer
        attention_weights: Attention weights tensor (Batch, Heads, Seq_Q, Seq_KV)
    """
    if not _CAPTURE_ENABLED:
        return

    if layer_name not in _ATTENTION_STORE:
        _ATTENTION_STORE[layer_name] = []

    # Store as numpy for easier manipulation
    _ATTENTION_STORE[layer_name].append(attention_weights.detach().cpu().numpy())


def get_attention_maps() -> Dict[str, np.ndarray]:
    """
    Get all captured attention maps.

    Returns:
        Dictionary mapping layer names to attention weight arrays
    """
    return _ATTENTION_STORE.copy()


def clear_attention():
    """Clear all stored attention weights."""
    global _ATTENTION_STORE
    _ATTENTION_STORE = {}


def aggregate_attention(
    attention_maps: Dict[str, List[np.ndarray]],
    aggregate_heads: bool = True,
    aggregate_layers: bool = True,
    aggregate_spatial: bool = False,
) -> np.ndarray:
    """
    Aggregate attention weights across heads and layers.

    Args:
        attention_maps: Dictionary of layer_name -> list of attention arrays
        aggregate_heads: Whether to average across attention heads
        aggregate_layers: Whether to average across layers
        aggregate_spatial: Whether to average across spatial dimensions (needed when layers have different resolutions)

    Returns:
        Aggregated attention weights of shape:
        - If all True: (Seq_KV,)
        - If aggregate_heads and aggregate_layers True: (Seq_Q, Seq_KV) - may fail if spatial dims differ
        - If aggregate_heads only: (Num_Layers, Seq_Q, Seq_KV) - may fail if spatial dims differ
        - If aggregate_layers only: (Num_Heads, Seq_Q, Seq_KV)
        - If both False: (Num_Layers, Num_Heads, Seq_Q, Seq_KV)
    """
    all_weights = []

    for layer_name, weight_list in attention_maps.items():
        for weights in weight_list:
            # weights shape: (Batch, Heads, Seq_Q, Seq_KV)
            # Take first batch element
            weights = weights[0]  # (Heads, Seq_Q, Seq_KV)

            if aggregate_heads:
                # Average across heads
                weights = weights.mean(axis=0)  # (Seq_Q, Seq_KV)

            if aggregate_spatial:
                # Average across spatial dimensions (queries)
                # This is necessary when layers have different spatial resolutions
                if len(weights.shape) == 3:
                    # (Heads, Seq_Q, Seq_KV) -> (Heads, Seq_KV)
                    weights = weights.mean(axis=1)
                elif len(weights.shape) == 2:
                    # (Seq_Q, Seq_KV) -> (Seq_KV,)
                    weights = weights.mean(axis=0)

            all_weights.append(weights)

    if not all_weights:
        return np.array([])

    # Check if all weights have the same shape
    shapes = [w.shape for w in all_weights]
    if len(set(shapes)) > 1:
        # Shapes differ - need to handle this
        # If we haven't already aggregated spatial dims, do it now
        if not aggregate_spatial:
            # Average across spatial dimension for each weight
            all_weights = [
                w.mean(axis=-2) if len(w.shape) >= 2 else w
                for w in all_weights
            ]

    all_weights = np.stack(all_weights, axis=0)  # (Num_Layers, ...)

    if aggregate_layers:
        # Average across layers
        all_weights = all_weights.mean(axis=0)

    return all_weights


def get_token_attention(
    attention_maps: Dict[str, List[np.ndarray]],
    aggregate_spatial: bool = True,
) -> np.ndarray:
    """
    Get attention weights per text token, aggregated across spatial dimensions.

    This gives us a single attention score per token, showing how much the
    model is attending to each word in the prompt overall.

    Args:
        attention_maps: Dictionary of layer_name -> list of attention arrays
        aggregate_spatial: Whether to average across spatial dimensions (queries)

    Returns:
        Token attention weights of shape (Seq_KV,) if aggregate_spatial=True,
        otherwise (Seq_Q, Seq_KV)
    """
    # First aggregate across heads and layers
    attn = aggregate_attention(
        attention_maps,
        aggregate_heads=True,
        aggregate_layers=True
    )

    if len(attn) == 0:
        return np.array([])

    # Check the shape of returned attention
    # If it's already 1D, spatial dimension was already aggregated due to shape mismatch handling
    # If it's 2D, we have (Seq_Q, Seq_KV) where Seq_Q is spatial (H*W)

    if aggregate_spatial and len(attn.shape) == 2:
        # Average across all spatial positions to get overall token importance
        token_attn = attn.mean(axis=0)  # (Seq_KV,)
    elif len(attn.shape) == 0:
        # Scalar - this shouldn't happen, return empty array
        return np.array([])
    else:
        # Already 1D or user doesn't want spatial aggregation
        token_attn = attn

    return token_attn


def decode_tokens(tokenizer, token_ids: torch.Tensor) -> List[str]:
    """
    Decode CLIP token IDs to text strings.

    Args:
        tokenizer: CLIP tokenizer
        token_ids: Token ID tensor (Batch, Seq_Len)

    Returns:
        List of token strings
    """
    # Get first sequence
    token_ids = token_ids[0].cpu().numpy()

    # Decode each token
    tokens = []
    for token_id in token_ids:
        # CLIP tokenizer decode
        token_text = tokenizer.decode([token_id])
        tokens.append(token_text)

    return tokens


def visualize_token_attention(
    token_attention: np.ndarray,
    tokens: List[str],
    top_k: int = None,
) -> List[tuple]:
    """
    Create a sorted list of (token, attention_score) pairs.

    Args:
        token_attention: Attention scores per token (Seq_Len,)
        tokens: List of token strings
        top_k: If provided, return only top-k attended tokens

    Returns:
        List of (token, score) tuples sorted by attention score
    """
    # Pair tokens with their attention scores
    token_scores = list(zip(tokens, token_attention))

    # Sort by attention score (descending)
    token_scores.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        token_scores = token_scores[:top_k]

    return token_scores
