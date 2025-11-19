"""
Replication Package: Two-Faced Social Agents
Produces Results 1-4 and 8 from the paper

Run: python analysis_sat.py
Output: results/ directory with all figures and tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Plotting style
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
sns.set_palette("husl")


def result_1_accuracy_by_ses():
    """Result 1: SAT Math Accuracy by SES Background (Figure 1)"""

    # Load data
    df = pd.read_csv('data/accuracy_results.csv')
    anova_df = pd.read_csv('data/anova_results.csv')

    # Calculate accuracy percentages
    accuracy = df.groupby(['model', 'scenario', 'ses_3level'])[
        'correct'].mean() * 100
    accuracy_df = accuracy.reset_index()
    accuracy_df.columns = ['model', 'scenario', 'ses_3level', 'accuracy']

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scenarios = ['optimal_conditions',
                 'moderate_stress', 'challenging_conditions']
    titles = ['Optimal', 'Moderate Stress', 'Challenging']
    colors = {'Low': '#e74c3c', 'Mid': '#f39c12', 'High': '#27ae60'}

    for idx, (scenario, title) in enumerate(zip(scenarios, titles)):
        ax = axes[idx]
        data = accuracy_df[accuracy_df['scenario'] == scenario]

        models = data['model'].unique()
        x = np.arange(len(models))
        width = 0.25

        for i, ses in enumerate(['Low', 'Mid', 'High']):
            ses_data = data[data['ses_3level'] == ses]
            ses_values = [ses_data[ses_data['model'] == m]['accuracy'].values[0]
                          if len(ses_data[ses_data['model'] == m]) > 0 else 0
                          for m in models]
            ax.bar(x + width * (i - 1), ses_values, width,
                   label=f'{ses} SES', color=colors[ses], alpha=0.8)

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(85, 102)
        if idx == 0:
            ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('SAT Math Performance by Scenario and SES Background',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figure1_accuracy_by_ses.png',
                bbox_inches='tight')
    plt.close()

    # Save summary
    accuracy_df.to_csv(
        f'{RESULTS_DIR}/table1_accuracy_summary.csv', index=False)
    anova_df.to_csv(f'{RESULTS_DIR}/table2_anova_results.csv', index=False)


def result_2_preference_analysis():
    """Result 2: Preference Task Analysis (Figure 2)"""

    # Load preference statistical results
    pref_df = pd.read_csv('data/preference_statistical.csv')

    # Pivot for heatmaps
    effect_pivot = pref_df.pivot(index='preference_item',
                                 columns='model',
                                 values='effect_size')
    pval_pivot = pref_df.pivot(index='preference_item',
                               columns='model',
                               values='p_value')

    # Create dual heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Effect size heatmap
    sns.heatmap(effect_pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=ax1, cbar_kws={'label': 'Effect Size (ε² / Cramer\'s V)'})
    ax1.set_title('Effect Sizes of SES on Preferences\n(Darker red = larger effect)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Preference Dimension', fontsize=12)

    # P-value heatmap (green = significant)
    pval_colors = pval_pivot.applymap(lambda x: x < 0.05)
    sns.heatmap(pval_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax2, cbar_kws={'label': 'p-value'}, vmin=0, vmax=0.10)
    ax2.set_title('Statistical Significance of SES Effects on Preferences\n(Green = more significant)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figure2_preference_heatmaps.png',
                bbox_inches='tight')
    plt.close()

    # Save summary
    pref_df.to_csv(
        f'{RESULTS_DIR}/table3_preference_statistics.csv', index=False)


def result_3_embedding_analysis():
    """Result 3: Semantic Embedding Analysis (Figure 3)"""

    # Load reasoning data
    df = pd.read_csv('data/reasoning_data.csv')

    # Load sentence transformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Create figure with subplots for each model
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {'Low': '#e74c3c', 'Mid': '#f39c12', 'High': '#27ae60'}

    for idx, model_name in enumerate(df['model'].unique()):
        ax = axes[idx]
        model_data = df[df['model'] == model_name]

        # Generate embeddings
        embeddings = model.encode(model_data['reasoning_text'].tolist(),
                                  batch_size=32, show_progress_bar=False)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30,
                    random_state=42, n_iter=1000)
        coords = tsne.fit_transform(embeddings)

        # Plot
        for ses in ['Low', 'Mid', 'High']:
            mask = model_data['ses_3level'].values == ses
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[ses], label=f'{ses} SES',
                       alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax.set_title(f'{model_name}\nReasoning Embeddings by SES',
                     fontsize=12, fontweight='bold')
        if idx == 2:
            ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.2)

    plt.suptitle('Semantic Structure of SAT Reasoning (Correct Answers Only)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figure3_tsne_embeddings.png',
                bbox_inches='tight')
    plt.close()


def result_4_linguistic_analysis():
    """Result 4: Linguistic Feature Analysis (Figure 4)"""

    # Load linguistic data
    features_df = pd.read_csv('data/linguistic_features.csv')
    results_df = pd.read_csv('data/linguistic_results.csv')

    # Focus on Claude (most variation)
    claude_results = results_df[
        (results_df['model'] == 'Claude Sonnet') &
        (results_df['significant'] == True)
    ].sort_values('cohens_d', key=abs, ascending=False).head(8)

    if len(claude_results) == 0:
        print("No significant linguistic differences found for Claude")
        return

    # Create violin plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    claude_data = features_df[features_df['model'] == 'Claude Sonnet']

    for idx, (_, row) in enumerate(claude_results.iterrows()):
        if idx >= 8:
            break

        ax = axes[idx]
        feature = row['feature']

        # Prepare data
        low_vals = claude_data[claude_data['ses_3level']
                               == 'Low'][feature].values
        high_vals = claude_data[claude_data['ses_3level']
                                == 'High'][feature].values

        plot_data = pd.DataFrame({
            'SES': ['Low'] * len(low_vals) + ['High'] * len(high_vals),
            'Value': np.concatenate([low_vals, high_vals])
        })

        # Violin plot
        sns.violinplot(data=plot_data, x='SES', y='Value', ax=ax,
                       palette={'Low': '#e74c3c', 'High': '#27ae60'})

        # Add mean markers
        ax.scatter([0, 1], [low_vals.mean(), high_vals.mean()],
                   color='black', s=100, zorder=10, marker='D')

        ax.set_title(f"{feature.replace('_', ' ').title()}\n"
                     f"d={row['cohens_d']:.2f}, p={row['p_value']:.3f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.suptitle('Claude Sonnet: Reasoning Quality & Style by SES',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figure4_linguistic_features.png',
                bbox_inches='tight')
    plt.close()

    # Save summary
    results_df[results_df['significant'] == True].to_csv(
        f'{RESULTS_DIR}/table4_linguistic_differences.csv', index=False)


def result_8_human_alignment():
    """Result 8: Human vs AI Alignment (Figure 5)"""

    # Load alignment data
    align_df = pd.read_csv('data/human_alignment.csv')

    # Human baseline (College Board 2007)
    human_low = 0.438
    human_high = 0.593
    human_gap = human_high - human_low

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Human reference line
    ax.plot([human_low, human_high], [human_low, human_high],
            'k--', linewidth=2, label='Perfect Alignment', alpha=0.5)

    # Human data point
    ax.scatter([human_low, human_high], [human_low, human_high],
               s=300, c='black', marker='s', zorder=10,
               label='Human (College Board 2007)')

    # Model data
    colors = {'Claude Sonnet': '#3498db', 'GPT-5': '#e74c3c',
              'Gemini 2.5 Flash': '#2ecc71'}

    for _, row in align_df.iterrows():
        model = row['model']
        low_acc = row['low_ses_accuracy']
        high_acc = row['high_ses_accuracy']
        correlation = row['correlation']

        # Determine pattern
        if pd.isna(correlation):
            pattern = 'Suppressed (r=undef)'
        elif correlation > 0:
            pattern = f'Aligned (r={correlation:.2f})'
        else:
            pattern = f'Inverted (r={correlation:.2f})'

        # Plot model line
        ax.plot([human_low, human_high], [low_acc, high_acc],
                'o-', linewidth=2, markersize=12,
                color=colors.get(model, 'gray'),
                label=f'{model}\n{pattern}', alpha=0.7)

        # Add model name annotation
        ax.text(human_high + 0.01, high_acc, model.split()[0],
                fontsize=9, va='center')

    ax.set_xlabel('Human SAT Math Accuracy\n(Low SES → High SES)', fontsize=13)
    ax.set_ylabel('AI Model Accuracy\n(Low SES → High SES)', fontsize=13)
    ax.set_title('AI vs Human SAT Performance Patterns by SES\n' +
                 f'(Human gap: {human_gap:.1%})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.4, 0.65)
    ax.set_ylim(0.85, 1.02)

    # Add annotations
    ax.annotate('Low-SES students', xy=(human_low, human_low),
                xytext=(0.35, 0.50), fontsize=10, style='italic',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('High-SES students', xy=(human_high, human_high),
                xytext=(0.55, 0.65), fontsize=10, style='italic',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figure5_human_alignment.png',
                bbox_inches='tight')
    plt.close()

    # Save summary
    align_df.to_csv(f'{RESULTS_DIR}/table5_alignment_summary.csv', index=False)


def main():
    """Run all analyses"""
    result_1_accuracy_by_ses()
    result_2_preference_analysis()
    result_3_embedding_analysis()
    result_4_linguistic_analysis()
    result_8_human_alignment()

    print("\n" + "="*60)
    print("REPLICATION COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {RESULTS_DIR}/")
    print("\nGenerated files:")
    print("  Figures:")
    print("    - figure1_accuracy_by_ses.png")
    print("    - figure2_preference_heatmaps.png")
    print("    - figure3_tsne_embeddings.png")
    print("    - figure4_linguistic_features.png")
    print("    - figure5_human_alignment.png")
    print("  Tables:")
    print("    - table1_accuracy_summary.csv")
    print("    - table2_anova_results.csv")
    print("    - table3_preference_statistics.csv")
    print("    - table4_linguistic_differences.csv")
    print("    - table5_alignment_summary.csv")


if __name__ == '__main__':
    main()
