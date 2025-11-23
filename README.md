# Replication Package: Two-Faced Social Agents

Replication materials for Results:
**"Two-Faced Social Agents: Context Collapse in Role-Conditioned Large Language Models"**

https://arxiv.org/abs/2511.15573 Paper is available here.

Data can be swapped to other files in the directory and replicate all findings in the appendix and also the extended claude replication
## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run analysis:
```bash
python analysis_sat.py
```

3. View outputs in `results/` directory

## System Requirements

- Python 3.11+

## Outputs

### Figures (PNG, 600 DPI):
- `figure1_accuracy_by_ses.png` - SAT accuracy by SES and scenario
- `figure2_preference_heatmaps.png` - Preference task effect sizes
- `figure3_tsne_embeddings.png` - Semantic clustering of reasoning
- `figure4_linguistic_features.png` - Linguistic differences by SES
- `figure5_human_alignment.png` - Human vs AI performance patterns

### Tables (CSV):
- `table1_accuracy_summary.csv` - Accuracy statistics
- `table2_anova_results.csv` - ANOVA test results
- `table3_preference_statistics.csv` - Preference task statistics
- `table4_linguistic_differences.csv` - Significant linguistic features
- `table5_alignment_summary.csv` - Human-AI correlation results

## Data Files

Located in `data/`:
- `accuracy_results.csv` (N=7,290) - SAT response data
- `anova_results.csv` - Statistical test results
- `preference_statistical.csv` - Preference task analysis
- `reasoning_data.csv` (N=6,947) - Reasoning text for correct answers
- `linguistic_features.csv` - Extracted linguistic features
- `linguistic_results.csv` - Linguistic statistical tests
- `human_alignment.csv` - Human-AI comparison data
