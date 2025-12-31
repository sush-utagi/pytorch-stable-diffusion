"""
Attention Visualization Utilities

Functions for creating visualizations of cross-attention weights during
diffusion model generation, showing how the model's focus on different
prompt tokens evolves throughout the denoising process.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple
import seaborn as sns


def visualize_attention_evolution(
    attention_data: List[Tuple[int, float, np.ndarray]],
    tokens: List[str],
    title: str = "Token Attention Evolution",
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "YlOrRd",
) -> plt.Figure:
    """
    Create a heatmap showing how attention to each token evolves over time.

    Args:
        attention_data: List of (step_idx, timestep, token_attention) tuples
        tokens: List of token strings from tokenizer
        title: Figure title
        figsize: Figure size (width, height)
        cmap: Colormap for heatmap

    Returns:
        matplotlib Figure object
    """
    if not attention_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No attention data available",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Extract attention matrices
    num_steps = len(attention_data)
    num_tokens = len(tokens)

    # Create attention matrix: rows=steps, cols=tokens
    attn_matrix = np.zeros((num_steps, num_tokens))
    step_labels = []

    for i, (step_idx, timestep, token_attn) in enumerate(attention_data):
        if len(token_attn) > 0:
            attn_matrix[i, :len(token_attn)] = token_attn[:num_tokens]
        step_labels.append(f"Step {step_idx}\nt={int(timestep)}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(attn_matrix, aspect='auto', cmap=cmap, interpolation='nearest')

    # Set ticks
    ax.set_yticks(range(num_steps))
    ax.set_yticklabels(step_labels, fontsize=8)
    ax.set_xticks(range(num_tokens))
    ax.set_xticklabels(tokens, rotation=90, ha='right', fontsize=7)

    # Labels
    ax.set_xlabel("Tokens", fontsize=11, fontweight='bold')
    ax.set_ylabel("Denoising Steps", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=10)

    # Grid
    ax.grid(False)

    plt.tight_layout()
    return fig


def plot_top_tokens_per_step(
    attention_data: List[Tuple[int, float, np.ndarray]],
    tokens: List[str],
    top_k: int = 5,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Create bar plots showing top-k attended tokens at each step.

    Args:
        attention_data: List of (step_idx, timestep, token_attention) tuples
        tokens: List of token strings
        top_k: Number of top tokens to show per step
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    num_steps = len(attention_data)

    # Calculate grid layout
    ncols = min(3, num_steps)
    nrows = (num_steps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_steps == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (step_idx, timestep, token_attn) in enumerate(attention_data):
        ax = axes[i]

        if len(token_attn) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.axis('off')
            continue

        # Get top-k tokens
        top_indices = np.argsort(token_attn)[-top_k:][::-1]
        top_tokens = [tokens[idx] for idx in top_indices]
        top_values = token_attn[top_indices]

        # Create bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
        bars = ax.barh(range(top_k), top_values, color=colors)

        # Labels
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_tokens, fontsize=8)
        ax.set_xlabel('Attention', fontsize=9)
        ax.set_title(f'Step {step_idx} (t={int(timestep)})', fontsize=10, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, top_values)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left', va='center', fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Hide unused subplots
    for i in range(num_steps, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Top Attended Tokens Per Step', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    return fig


def plot_token_attention_trajectory(
    attention_data: List[Tuple[int, float, np.ndarray]],
    tokens: List[str],
    selected_tokens: List[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot attention trajectories for selected tokens over denoising steps.

    Args:
        attention_data: List of (step_idx, timestep, token_attention) tuples
        tokens: List of all token strings
        selected_tokens: Specific tokens to track (if None, use top-k overall)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if not attention_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No attention data", ha='center', va='center')
        return fig

    # If no tokens selected, find top tokens across all steps
    if selected_tokens is None:
        # Average attention across all steps
        all_attn = np.stack([attn for _, _, attn in attention_data if len(attn) > 0])
        mean_attn = all_attn.mean(axis=0)
        top_indices = np.argsort(mean_attn)[-8:][::-1]  # Top 8
        selected_tokens = [tokens[idx] for idx in top_indices]

    # Find indices of selected tokens
    token_indices = {token: idx for idx, token in enumerate(tokens)}
    selected_indices = [token_indices.get(token, -1) for token in selected_tokens]
    selected_indices = [idx for idx in selected_indices if idx >= 0]

    if not selected_indices:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Selected tokens not found", ha='center', va='center')
        return fig

    # Extract trajectories
    steps = [step_idx for step_idx, _, _ in attention_data]
    trajectories = {tokens[idx]: [] for idx in selected_indices}

    for step_idx, timestep, token_attn in attention_data:
        for idx in selected_indices:
            if idx < len(token_attn):
                trajectories[tokens[idx]].append(token_attn[idx])
            else:
                trajectories[tokens[idx]].append(0.0)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each trajectory
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
    for i, idx in enumerate(selected_indices):
        token = tokens[idx]
        ax.plot(steps, trajectories[token], marker='o', linewidth=2,
               markersize=6, label=token, color=colors[i], alpha=0.8)

    # Labels and legend
    ax.set_xlabel('Denoising Step', fontsize=11, fontweight='bold')
    ax.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
    ax.set_title('Token Attention Trajectories', fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def create_comprehensive_attention_report(
    attention_data: List[Tuple[int, float, np.ndarray]],
    tokens: List[str],
    prompt: str,
    save_path: str = None,
) -> List[plt.Figure]:
    """
    Create a comprehensive multi-figure report of attention analysis.

    Args:
        attention_data: List of (step_idx, timestep, token_attention) tuples
        tokens: List of token strings
        prompt: Original text prompt
        save_path: Optional directory to save figures

    Returns:
        List of created figures
    """
    figures = []

    # Figure 1: Attention evolution heatmap
    fig1 = visualize_attention_evolution(
        attention_data, tokens,
        title=f'Token Attention Evolution\nPrompt: "{prompt}"'
    )
    figures.append(fig1)

    # Figure 2: Top tokens per step
    fig2 = plot_top_tokens_per_step(attention_data, tokens, top_k=5)
    figures.append(fig2)

    # Figure 3: Token trajectories
    fig3 = plot_token_attention_trajectory(attention_data, tokens)
    figures.append(fig3)

    # Save if requested
    if save_path:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        fig1.savefig(save_path / "attention_heatmap.png", dpi=150, bbox_inches='tight')
        fig2.savefig(save_path / "top_tokens.png", dpi=150, bbox_inches='tight')
        fig3.savefig(save_path / "token_trajectories.png", dpi=150, bbox_inches='tight')

    return figures
