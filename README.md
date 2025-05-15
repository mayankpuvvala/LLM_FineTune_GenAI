# 🔧 PyTorch GitHub Issue Body Generator

Automatically generate GitHub issue bodies from issue titles using a fine-tuned T5-small model. This project streamlines the documentation process in large-scale projects like PyTorch by suggesting detailed issue descriptions.

## 🔍 Problem Statement

Documenting issues in large projects like PyTorch is time-consuming and often inconsistent. This tool automates the generation of issue descriptions based on the issue title, saving developer time and ensuring better documentation.

## 🎯 Objective

Train a sequence-to-sequence model that takes a GitHub issue title as input and generates a plausible issue body. The model is based on the T5 architecture and fine-tuned on a custom dataset of PyTorch issues.

## 🧰 Technical Stack

- **Model**: T5-small (Text-to-Text Transfer Transformer)
- **Fine-tuning**: PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation)
- **Framework**: PyTorch
- **Deployment**: Gradio on Hugging Face Spaces
- **Dataset**: Custom dataset built using GitHub API (issue titles and bodies)

## 📊 Performance

- **ROUGE Precision**: 53.12%
- **ROUGE F1 Score**: 49.8%
- **Epochs**: 3

These results demonstrate strong performance, especially given the use of PEFT + LoRA for efficient training with reduced computational load.

## 🚀 Demo & Models

**👉 Try the model live**:  
[Hugging Face Space](https://huggingface.co/spaces/mayankpuvvala/pytorch_issues_deployment)

**🧠 Model Checkpoints**:
- [LoRA Adapter Training Model](https://huggingface.co/mayankpuvvala/lora-t5-pytorch-issues)
- [Merged Inference Model](https://huggingface.co/mayankpuvvala/peft_lora_t5_merged_model_pytorch_issues)

**📂 Dataset**:  
[GitHub-PyTorch-Issues Dataset](https://huggingface.co/datasets/mayankpuvvala/github-pytorch-issues)  
(Visit the data card for preprocessing and structure details.)

## 🧪 Usage Example

**Input (Issue Title)**: Memory leak when using DataLoader with num_workers > 0

**Output (Generated Body)**: When using DataLoader with num_workers greater than 0, the memory usage increases over time, leading to a memory leak.
