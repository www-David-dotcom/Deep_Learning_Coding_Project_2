## Deep Learning Coding Project 2: Tiny ImageNet Image Classification

### 1. Project Overview

In this project, you will design and train a deep learning model for image classification on the [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) dataset using PyTorch. The dataset contains **200 classes** of real-world objects (e.g., animals, vehicles, household items), with 64×64 RGB images — a scaled-down but challenging variant of the full ImageNet benchmark.

Image classification is a foundational task in computer vision and deep learning. By building a model from scratch (without pre-trained weights), you will gain hands-on experience with convolutional neural network architecture design, training loop engineering, hyperparameter tuning, and data augmentation — all essential skills in modern deep learning practice.

### 2. Learning Objectives

By the end of this project, you will be able to:

- Design and implement a custom convolutional neural network architecture for image classification using PyTorch.
- Write a complete training loop including optimizer configuration, loss computation, and checkpoint saving.
- Apply training techniques such as data augmentation, learning rate scheduling, dropout, and weight decay to improve model performance.
- Evaluate a trained model on a held-out validation set and interpret accuracy metrics.
- Work within practical constraints (parameter budgets, no pre-trained models) that mirror real-world deployment considerations.

### 3. Prerequisites and Environment Setup

**Prerequisites:** Familiarity with Python, PyTorch, and basic deep learning concepts (CNNs, backpropagation, gradient descent).

**Environment Setup:**

1. Install [uv](https://docs.astral.sh/uv/):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   If you encounter network issues, use the mirror:

   ```bash
   curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

**Data Preparation:**

Download and extract the Tiny ImageNet dataset:

```bash
bash download_data.sh
```

#### Dataset Details

| Property | Value |
|---|---|
| Training images | 100,000 (500 per class) |
| Validation images | 10,000 (50 per class) |
| Image shape | `(3, 64, 64)` |
| Number of classes | 200 |
| Data directory | `data/tiny-imagenet-200/` |

### 4. Codebase Structure

Familiarize yourself with the starter code. Do not modify files marked as read-only.

```
deep-learning-coding-project-2/
├── datasets.py            # [Read-only] TinyImageNetDataset class (PyTorch Dataset)
├── evaluate.py            # [Read-only] Model evaluation script (accuracy computation)
├── download_data.sh       # [Read-only] Shell script to download & extract dataset
├── modules.py             # [TODO] CustomModel class — implement your model here
├── train.py               # [TODO] Training loop — implement the train() function here
├── report.md              # [TODO] Write your report here
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock                # Dependency lock file
├── checkpoints/           # Directory for saved model checkpoints
└── data/                  # Dataset directory (created by download_data.sh)
    └── tiny-imagenet-200/
        ├── train/         # 200 subdirectories, 500 images each
        ├── val/           # 10,000 validation images with annotations
        └── test/          # Unlabeled test images
```

### 5. Implementation Tasks

**Task 1: Model Architecture (`modules.py`)**

Implement the `CustomModel` class by filling in the `__init__` and `forward` methods. Your model must:

- Accept input images of shape `(batch_size, 3, 64, 64)`.
- Output logits of shape `(batch_size, 200)` — one score per class.
- Contain **no more than 20 million parameters**.
- **Not** use any pre-trained weights (e.g., ImageNet-pretrained ResNet is strictly prohibited).

You are free to choose any architecture (e.g., custom CNN, ResNet-style, etc.) as long as these constraints are met.

**Task 2: Training Loop (`train.py`)**

Implement the `train(model, dataset)` function. You are free to choose your own:

- Optimizer (e.g., SGD, Adam, AdamW)
- Loss function (e.g., CrossEntropyLoss)
- Learning rate and learning rate scheduler
- Batch size and number of epochs
- Data augmentation strategy
- Regularization techniques (e.g., dropout, weight decay)

The `main()` function (already provided) handles argument parsing, dataset creation, model instantiation, and checkpoint saving. You only need to implement the `train()` function.

**Task 3: Report (`report.md`)**

Write a report with the following sections:

1. **Cover Information** — Name and student ID.
2. **Generative AI Usage Disclosure** — State "None" if you did not use AI. Otherwise, describe which tool(s) you used and how (e.g., brainstorming, code generation, debugging, concept explanation).
3. **Architecture Details** — Describe your `CustomModel` architecture.
4. **Hyperparameters** — Document learning rate, batch size, optimizer, number of epochs, etc.
5. **Training Techniques** — Describe techniques used (e.g., data augmentation, learning rate scheduler, dropout, weight decay).
6. **Training and Validation Curves** — Include loss and accuracy curves.

### 6. Testing and Debugging

**Training your model:**

```bash
uv run python train.py checkpoints/best.pth
```

**Evaluating your model on the validation set:**

```bash
uv run python evaluate.py checkpoints/best.pth
```

The evaluation script reports top-1 accuracy on the validation split. Use this to track your progress and debug your model. A validation accuracy of **45% or higher** earns full marks for the performance component.

If your model exceeds 20 million parameters, `train.py` will log an error and exit. Verify your parameter count before lengthy training runs.

### 7. Submission Guidelines

Follow these exact steps to ensure your project is successfully received and graded.

1. Ensure `modules.py`, `train.py`, and `report.md` are finalized.
2. Create a ZIP archive named `submission.zip`:

   ```bash
   zip -r submission.zip modules.py train.py report.md
   ```

3. Verify the archive structure matches the following layout:

   ```
   submission.zip
   ├── modules.py
   ├── train.py
   └── report.md
   ```

4. Submit `submission.zip` through the designated course submission platform.

### 8. Grading Rubric

Your project will be evaluated based on the following criteria:

| Criteria | Weight | Description |
| --- | --- | --- |
| Model Performance | 70% | Based on validation accuracy. Given your validation accuracy *X*, your score is: `(min(X, H) - 0.1) / (H - 0.1) × 7`, where *H* = 0.45. Achieving **45% accuracy** earns full marks for this component. |
| Report | 30% | Completeness and clarity of the report, covering architecture details, hyperparameters, training techniques, AI usage disclosure, and training/validation curves. |
| Bonus | +1 point | The submission with the **highest validation accuracy** receives **1 bonus point** toward the **final course grade**. |

**Grading Environment:**

Your submission will be executed on the grading platform which at least meets the following specifications.

| Resource | Specification |
|---|---|
| GPU VRAM | 16 GB |
| System RAM | 32 GB |
| Training Time | 30 minutes |

### 9. Academic Integrity and AI Policy

**Plagiarism in any form will result in an F for the course.**

You must disclose any use of generative AI tools in the **Generative AI Usage Disclosure** section of your report. If you did not use AI, state "None." If you did, describe which tool(s) you used and how (e.g., brainstorming, code generation, debugging, concept explanation). Undisclosed use of AI tools is a violation of academic integrity.

All submitted code must be your own work. You may discuss high-level ideas with classmates, but sharing or copying code is strictly prohibited. Use of pre-trained models is not permitted for this assignment.
