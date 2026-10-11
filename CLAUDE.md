# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Code from Andrej Karpathy's "Neural Networks: Zero To Hero" nanoGPT lecture. It is deliberately kept close to what was written in the video, and `git log` is meant to be walked as a teaching history. Prefer small, readable changes over refactors or abstractions.

## Running

There is no build system, package manifest, test suite, or linter. The only dependency is PyTorch. Each script is self-contained and trains end-to-end when run:

```
python bigram.py   # tiny bigram baseline, runs quickly on CPU
python gpt.py      # ~10.8M-param char-level Transformer; realistically needs a GPU
```

Both scripts read `input.txt` (Tiny Shakespeare) from the **current working directory**, so run them from the repo root. They print train/val loss every `eval_interval` steps and then print 500 sampled characters. Nothing is saved to disk (no checkpoints). `more.txt` is a pre-generated 10k-character sample from `gpt.py`; the commented-out last line of `gpt.py` regenerates it.

To get a fast smoke run of `gpt.py`, lower `max_iters`, `eval_iters`, `block_size`, and `n_embd`/`n_layer` in the hyperparameter block at the top of the file. There is no CLI or config.

## Architecture

- **Two parallel scripts, not a library.** `bigram.py` and `gpt.py` share copy-pasted code for data loading, the char tokenizer (`stoi`/`itos`, `encode`/`decode`), the 90/10 train/val split, `get_batch`, `estimate_loss`, and the training loop. If you change one of these shared pieces, decide whether the other script should match.
- **Hyperparameters are module-level globals** (`n_embd`, `block_size`, `dropout`, `device`, ...) that the `nn.Module` classes in `gpt.py` read directly instead of taking them as constructor arguments. `estimate_loss` also uses the global `model`. Changing a hyperparameter means editing the top of the file.
- **Model in `gpt.py`:** token + learned position embeddings → `n_layer` pre-LayerNorm `Block`s (residual `MultiHeadAttention` built from separate `Head` modules, plus a `FeedFoward` MLP; the misspelling is the real class name) → final `ln_f` → `lm_head`. Causal masking uses a registered `tril` buffer sized `block_size`, and `generate` crops the context to the last `block_size` tokens.
- **Weight init:** `GPTLanguageModel._init_weights` (normal with std 0.02, zero biases) was added after the lecture. The README note saying init is missing predates this commit.
