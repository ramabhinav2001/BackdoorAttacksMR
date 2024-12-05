## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2. Set Up the Conda Environment

Ensure you have Conda (or Mamba) installed. To create and activate the environment from the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate your-environment-name
```

*Replace `your-environment-name` with the name specified in `environment.yaml`.*

### 3. Dataset Preparation

Make sure the dataset you want to use (e.g., CIFAR-10) is available in the path specified by `--data_path`. By default, `data/` is used. The script will attempt to download or load the dataset automatically if supported by the code. 

If you are using custom datasets, follow the instructions in `get_dataset()` to ensure they are placed or referenced correctly.

### 4. Running the Code

**Basic Command:**
```bash
python DC/main.py --dataset CIFAR10 --model ConvNet --ipc 10 --data_path /path/to/data --save_path /path/to/results
```

**Common Arguments:**
- `--method {DC, DSA}`: Select the distillation method.
- `--dataset {CIFAR10, GTSRB, ...}`: Choose the dataset.
- `--model {ConvNet, ResNet, ...}`: Model architecture.
- `--ipc {int}`: Images per class for the synthetic dataset.
- `--Iteration {int}`: Number of training iterations for dataset distillation.
- `--ori {float}`: Fraction of the original dataset to use.
- `--portion {float}`: Proportion of the dataset to poison.

**Backdoor Attack Options:**
- `--doorping`: Use a doorping-style backdoor trigger.
- `--invisible`: Use an invisible (image-based) trigger.
- `--simple_trigger`: Insert a static, non-updatable trigger pattern.
- `--relaxed_trigger`: Insert a gradient-updatable trigger (requires `--kip`).
- `--kip`: Enable KIP-based approach (necessary for relaxed triggers).

**Example Commands:**

- **Simple Trigger (No KIP)**:
  ```bash
  python main.py --dataset CIFAR10 --model ConvNet --ipc 10 --simple_trigger
  ```

- **Relaxed Trigger with KIP**:
  ```bash
  python main.py --dataset CIFAR10 --model ConvNet --ipc 10 --kip --relaxed_trigger --portion 0.01
  ```

- **Doorping Trigger**:
  ```bash
  python main.py --dataset CIFAR10 --model ConvNet --ipc 10 --doorping --portion 0.01
  ```

- **Invisible Trigger**:
  ```bash
  python main.py --dataset CIFAR10 --model ConvNet --ipc 10 --invisible --portion 0.01
  ```

### 5. Results and Outputs

- **Logs**: The script prints training progress, losses, and evaluation metrics (Clean Test Accuracy, Attack Success Rate) to the terminal.
- **Synthetic Data and Triggers**: Saved in the specified `--save_path`. Each experiment folder contains:
  - `vis_*` images and `.pth` files with synthetic data at certain iterations.
  - Trigger images when applicable.
  - Models saved periodically for evaluation.


### 6. Troubleshooting

- **Environment Issues**: If packages are missing or conflicts arise, update `environment.yaml` and re-create the environment.
- **Dataset Errors**: Check that the dataset path is correct, or modify `get_dataset()` to handle custom data formats.
- **Trigger Not Appearing or Not Updating**: Verify that the correct flags are used (`--simple_trigger` vs. `--relaxed_trigger` and `--kip`). For relaxed triggers, ensure `--kip` is enabled.

---

This README should give users a clear starting point for setting up the conda environment, running experiments, and understanding what the code does and how to customize it.