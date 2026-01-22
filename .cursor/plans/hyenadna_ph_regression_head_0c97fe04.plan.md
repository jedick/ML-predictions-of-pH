---
name: HyenaDNA pH Regression Head
overview: Implement a standalone training script with a separate RegressionHead class for HyenaDNA model to predict continuous pH values from DNA sequences. The script will download data from HuggingFace, handle sequence concatenation with [SEP] tokens, and train using pretrained HyenaDNA weights.
todos:
  - id: create_regression_head
    content: Create separate RegressionHead class with architecture options (linear, mlp2, mlp3, mlp2_ln)
    status: pending
  - id: create_training_script
    content: Create standalone train_hyenadna_ph.py script in project root with all components
    status: pending
  - id: implement_data_loading
    content: Implement HuggingFace dataset loading and sequence concatenation with [SEP] tokens
    status: pending
  - id: implement_model_setup
    content: Implement model loading with pretrained weights and regression head attachment
    status: pending
  - id: implement_training_loop
    content: Implement PyTorch training loop with loss, optimizer, and basic evaluation metrics
    status: pending
  - id: add_hyperparameters
    content: Add configurable hyperparameters (model size, learning rate, batch size, etc.) with reasonable defaults
    status: pending
isProject: false
---

# HyenaDNA pH Regression Head Implementation Plan

## Overview

Create a standalone training script `train_hyenadna_ph.py` in the project root that:

- Implements a separate `RegressionHead` class (maintaining separation from HyenaDNA codebase)
- Downloads dataset from HuggingFace (`jedick/microbial-DNA-pH`)
- Uses pretrained HyenaDNA weights (configurable model size, default: tiny-1k-seqlen)
- Concatenates multiple DNA sequences per sample with [SEP] tokens up to model max_length
- Trains using plain PyTorch (no Lightning dependency)
- Computes basic evaluation metrics (MSE, MAE, R²)

## Current State

The HyenaDNA codebase in [`hyena-dna/standalone_hyenadna.py`](hyena-dna/standalone_hyenadna.py):

- Uses `SequenceDecoder` for classification (we will NOT modify this)
- `CharacterTokenizer` supports [SEP] token (ID=1) - **confirmed feasible for concatenation**
- Model uses causal attention with left padding
- Pretrained weights available on HuggingFace

The dataset structure (from [`create_hf_dataset.py`](create_hf_dataset.py)):

- DNA sequences stored as lists of strings per sample
- pH values are continuous floats (can be `None` for missing data)
- Each sample has multiple sequences (up to 1MB total DNA length)
- Available on HuggingFace: `jedick/microbial-DNA-pH`

## Architecture Design

### RegressionHead Class

Create a new `RegressionHead` class that:

- Takes hidden states from HyenaDNA backbone (shape: `(batch, seq_len, d_model)`)
- Applies pooling to get sequence-level representation (similar to `SequenceDecoder`)
- Maps pooled features to single pH value via MLP
- Supports multiple architecture options

**Architecture Options**:

1. **`linear`**: Simple linear layer
   ```python
   nn.Linear(d_model, 1)
   ```

2. **`mlp2`** (Default): 2-layer MLP
   ```python
   nn.Sequential(
       nn.Linear(d_model, d_model // 2),
       nn.ReLU(),
       nn.Dropout(0.1),
       nn.Linear(d_model // 2, 1)
   )
   ```

3. **`mlp3`**: 3-layer MLP
   ```python
   nn.Sequential(
       nn.Linear(d_model, d_model),
       nn.ReLU(),
       nn.Dropout(0.1),
       nn.Linear(d_model, d_model // 2),
       nn.ReLU(),
       nn.Dropout(0.1),
       nn.Linear(d_model // 2, 1)
   )
   ```

4. **`mlp2_ln`**: 2-layer MLP with LayerNorm
   ```python
   nn.Sequential(
       nn.LayerNorm(d_model),
       nn.Linear(d_model, d_model // 2),
       nn.ReLU(),
       nn.Dropout(0.1),
       nn.Linear(d_model // 2, 1)
   )
   ```


**Pooling Modes** (reuse logic from `SequenceDecoder`):

- `pool`: Mean pooling (default) - works well for variable-length sequences
- `last`: Use last token representation
- `first`: Use first token representation
- `sum`: Sum all tokens

**Output**: Single scalar pH value per sample `(batch,)` or `(batch, 1)`

### Model Integration Strategy

Since we're keeping separation from HyenaDNA codebase:

- Load pretrained `HyenaDNAModel` with `use_head=False` (embeddings only)
- Attach our `RegressionHead` separately
- Only the regression head will be trained (backbone can be frozen or fine-tuned)

## Implementation Details

**Important**: We will NOT modify any files in the `hyena-dna/` directory. All implementation will be in a new standalone script `train_hyenadna_ph.py` in the project root, maintaining complete separation from the HyenaDNA codebase.

### New File: `train_hyenadna_ph.py`

The script will contain all components needed for training. See "File Structure" section below for details.

### Files to Modify (DEPRECATED - for reference only)

1. **`hyena-dna/standalone_hyenadna.py`**:

   - Modify `SequenceDecoder` class to support regression mode
   - Add head architecture options
   - Update `HyenaDNAModel` to support `task_type` parameter

2. **`hyena-dna/huggingface_wrapper.py`**:

   - Update `HyenaDNAPreTrainedModel.from_pretrained` to accept `task_type` and `head_architecture`

### Key Changes

1. **SequenceDecoder modifications**:
   ```python
   def __init__(
       self, 
       d_model, 
       d_output=None, 
       l_output=None, 
       use_lengths=False, 
       mode="last",
       task_type="classification",  # NEW
       head_architecture="linear"   # NEW
   ):
   ```


   - Build `output_transform` based on `task_type` and `head_architecture`
   - For regression: ensure `d_output=1` and use appropriate architecture

2. **HyenaDNAModel modifications**:
   ```python
   def __init__(
       ...,
       use_head=False,
       task_type="classification",      # NEW
       n_classes: int = 2,              # For classification
       head_architecture="linear",     # NEW
       ...
   ):
   ```


   - Determine `d_output` based on `task_type`
   - Pass `task_type` and `head_architecture` to `SequenceDecoder`

3. **Output shape**:

   - Classification: `(batch, n_classes)` 
   - Regression: `(batch, 1)` or `(batch,)` after squeeze

### HyenaDNA Model Configuration

Based on the [HyenaDNA repository](https://github.com/HazyResearch/hyena-dna), needed configurations:

**Model Size Options** (with max_lengths):

- `hyenadna-tiny-1k-seqlen`: max_length=1024 (default for testing)
- `hyenadna-small-32k-seqlen`: max_length=32768
- `hyenadna-medium-160k-seqlen`: max_length=160000
- `hyenadna-medium-450k-seqlen`: max_length=450000
- `hyenadna-large-1m-seqlen`: max_length=1_000_000

**Tokenizer Configuration**:

- Characters: `["A", "C", "G", "T", "N"]` (N for uncertain nucleotides)
- `model_max_length = max_length + 2` (account for special tokens)
- `padding_side = "left"` (causal model)

**Model Loading**:

- Use `HyenaDNAPreTrainedModel.from_pretrained()` from `hyena-dna/huggingface_wrapper.py`
- Set `use_head=False` to get embeddings only
- Download weights if not present locally (configurable)

### Default Hyperparameters

**Model**:

- Model size: `hyenadna-tiny-1k-seqlen` (default, configurable)
- Head architecture: `mlp2` (2-layer MLP)
- Pooling mode: `pool` (mean pooling)
- Freeze backbone: `False` (fine-tune entire model)

**Training**:

- Batch size: `4` (small for long sequences, adjust based on GPU memory)
- Learning rate: `1e-4` (typical for fine-tuning)
- Num epochs: `10` (configurable)
- Loss function: `SmoothL1Loss` (beta=1.0)
- Optimizer: `AdamW` with weight decay `0.01`
- Learning rate scheduler: Optional `ReduceLROnPlateau` (patience=3)

**Data**:

- Train/Val/Test split: `0.8/0.1/0.1`
- Filter samples with missing pH values
- Random seed: `42` (for reproducibility)

### Sequence Concatenation Details

**Confirmed Feasibility**: [SEP] token (ID=1) exists in `CharacterTokenizer` and can be used for concatenation.

**Implementation Strategy**:

1. For each sample with multiple sequences:

   - Tokenize each sequence: `tokenizer(seq)["input_ids"]`
   - Remove special tokens from individual sequences (keep only DNA tokens)
   - Join sequences with [SEP] token: `seq1_tokens + [1] + seq2_tokens + [1] + ...`
   - Truncate if total length > `max_length`: keep last `max_length` tokens
   - Add [CLS] at start if desired (though causal models may not need it)

2. Edge cases:

   - Empty sequence list: return empty or single [SEP] token
   - Single sequence: no [SEP] needed, just tokenize and truncate
   - All sequences too long: take last sequence and truncate to max_length

3. Padding:

   - Pad to `max_length` with [PAD] token (ID=4) on the left
   - Create attention mask to ignore padding tokens in loss

### File Structure

```
train_hyenadna_ph.py
├── Imports (torch, datasets, transformers, etc.)
├── RegressionHead class
├── Data loading functions
│   ├── load_dataset_from_hf()
│   ├── concatenate_sequences()
│   ├── create_dataloader()
│   └── collate_fn() for batching
├── Model setup functions
│   ├── setup_model()
│   └── get_model_config()
├── Training functions
│   ├── train_epoch()
│   ├── evaluate()
│   └── save_checkpoint()
└── main() with argument parsing
```

### Dependencies

Add to project requirements (if not already present):

- `torch` (PyTorch)
- `datasets` (HuggingFace datasets)
- `transformers` (for tokenizer compatibility)
- `numpy`
- `scikit-learn` (for metrics: R², MSE, MAE)
- `tqdm` (optional, for progress bars)

### Output Files

The script should save:

- `best_model.pt`: Best model checkpoint (based on validation loss)
- `training_log.txt`: Training history (loss, metrics per epoch)
- `test_predictions.csv`: Test set predictions with ground truth
- `config.json`: Training configuration used

### Testing Considerations

1. **Sequence concatenation**: Verify [SEP] tokens are correctly inserted and model can process them
2. **Truncation**: Test with samples that exceed max_length
3. **Output shape**: Ensure regression head outputs `(batch,)` shape
4. **pH value range**: Verify predictions are reasonable (typically 0-14, but check actual data)
5. **Memory usage**: Monitor GPU memory, especially with longer sequences
6. **Training stability**: Check for NaN values, gradient clipping if needed

### Known Considerations

1. **Causal attention**: Model uses causal (autoregressive) attention, so [SEP] tokens will be part of the context but won't have bidirectional information
2. **Sequence order**: Order of sequences in concatenation may matter - consider shuffling as data augmentation
3. **Variable length**: Different samples will have different numbers of sequences - handle in batching
4. **Memory constraints**: Long sequences (32k+) require significant GPU memory - start with tiny model

### Future Enhancements (Out of Scope)

- Multi-task learning (classification + regression)
- Uncertainty quantification (e.g., variance prediction)
- Quantile regression (predict pH distribution)
- Attention-based pooling instead of simple pooling
- Sequence-level attention mechanisms
- Data augmentation (reverse complement, etc.)