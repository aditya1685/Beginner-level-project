# Convolutional Neural Network for Image Classification

This repository contains an implementation of a Convolutional Neural Network (CNN) designed to classify images into three categories: `pizza`, `steak`, and `sushi`. The project is built using PyTorch and incorporates custom dataset handling, model architecture, and training scripts.

---

## Features
- **Custom Dataset Handling**: Supports structured directories for image classification tasks.
- **CNN Architecture**: A custom-designed CNN with five convolutional blocks and fully connected layers.
- **Training and Testing**: Includes scripts to train and evaluate the model.
- **GPU Support**: Utilizes CUDA for faster training when available.

---

## Directory Structure

```
project_root/
├── model.py         # Defines the CNN architecture
├── engine.py        # Contains training and testing logic
├── data_setup.py    # Handles dataset loading and preprocessing
├── train.py         # Script to train and evaluate the model
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
```

---

## Requirements

Ensure you have the following installed:

- Python 3.8 or higher
- PyTorch 1.12 or higher
- Torchvision
- PIL
- tqdm

You can install all dependencies using:

```
pip install -r requirements.txt
```

---

## Dataset Preparation
The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/dietzschenostoevsky/pizza-steak-sushi).
The dataset should be structured as follows:

```
data/
├── train/
│   ├── pizza/
│   ├── steak/
│   └── sushi/
├── test/
│   ├── pizza/
│   ├── steak/
│   └── sushi/
```

Each class (e.g., `pizza`, `steak`, `sushi`) should have its own folder containing the respective images.

---

## How to Run

### Training the Model

1. Clone the repository:
   ```
git clone <repository_url>
cd project_root
```

2. Run the training script:
   ```
python train.py --model Model --batch_size 32 --lr 0.001 --num_epochs 30
```
   
   Arguments:
   - `--model`: Name of the model class (default: `Model`).
   - `--batch_size`: Batch size for training and testing (default: `32`).
   - `--lr`: Learning rate (default: `0.001`).
   - `--num_epochs`: Number of training epochs (default: `30`).

### Outputs

During training, the script will output the training and testing loss and accuracy for each epoch. Model checkpoints and results can be saved if configured.

---

## Model Architecture

The CNN model consists of five convolutional blocks, each containing:
- Convolutional layers
- Batch normalization
- ReLU activation
- Max-pooling layers

The fully connected classifier has:
- Two dense layers with 4096 neurons each
- Dropout layers for regularization
- Output layer with three neurons (one for each class)

---

## Custom Functions

### Data Setup
- `find_classes(directory)`: Identifies class names and maps them to indices.
- `ImageFolderCustom`: Custom dataset loader to handle structured image datasets.
- `create_dataloaders`: Creates PyTorch dataloaders for training and testing.

### Engine
- `train_step`: Handles a single training iteration.
- `test_step`: Handles a single testing iteration.
- `train`: Orchestrates the training and testing process across epochs.

---

## Example Results

After training for 30 epochs with the provided dataset, the model achieved the following results:

- **Training Accuracy**: ~95%
- **Test Accuracy**: ~92%

---

## Future Work

- Implement data augmentation for improved generalization.
- Add functionality to save and load trained models.
- Experiment with different optimizers and learning rates.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

- PyTorch: [https://pytorch.org](https://pytorch.org)
- Torchvision: [https://pytorch.org/vision/](https://pytorch.org/vision/)

---

## Contact

For any questions or suggestions, please open an issue or contact the repository maintainer.

