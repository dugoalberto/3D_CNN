# A 3D UNet for Brats2020 challenge

Let's see the [`paper`](https://github.com/dugoalberto/Tumor_Segmentation/blob/main/egpaper_final.pdf).

## Overview

This repository contains code to preprocess, train, and evaluate a 3D U-Net model on the BraTS (Brain Tumor Segmentation) dataset. The main script, `3dunet.py`, implements data loading, preprocessing (cropping, normalization), model definition, training loop with Dice loss, validation, learning rate scheduling, checkpointing, and uploading to Hugging Face.

## Requirements

* Python 3.10 or higher
* NVIDIA GPU with CUDA (for training)

### Python packages

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

**requirements.txt** should include:

```
torch>=1.8.0
numpy
pandas
scikit-learn
nibabel
opencv-python
matplotlib
tqdm
huggingface_hub
kagglehub
kaggle-secrets
tensorboard
```

## Setup

1. **Kaggle authentication**: Ensure you have a Kaggle API token (`kaggle.json`) in `~/.kaggle/` or set up via `kaggle-secrets`:

   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_key
   ```
2. **Hugging Face**: To upload checkpoints, set your token:

   ```bash
   export HUGGING_FACE_TOKEN=hf_xxx
   ```

## Usage

Open `preprocessing.py` in your editor or notebook environment. Adjust configuration variables at the top of the file:

```python
data_dir         = 'data/BraTS20_Training'
batch_size       = 2
num_epochs       = 300
slice_range      = (40, 190)
modalities       = ['flair', 't1', 't1ce', 't2']
learning_rate    = 4e-5
weight_decay     = 5e-6
use_amp          = True       # Mixed-precision training
checkpoint_dir   = 'models/'
```

Then run:

```bash
python 3dunet.py
```

This will:

1. Download the BraTS dataset from Kaggle (via `kagglehub`).
2. Preprocess each patient scan (crop, z-score normalization).
3. Create PyTorch `Dataset` and `DataLoader` objects.
4. Train a 3D U-Net with Combined Dice + Cross-Entropy loss.
5. Validate on hold-out set and apply early stopping.
6. Save model checkpoints every 5 epochs and upload to Hugging Face.

## Preprocessing Details

The script defines the `BrainTumorDataset` class:

```python
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, modalities, slice_range, is_train=True):
        # data_dir: directory with patient subfolders
        # modalities: list of ['flair','t1','t1ce','t2']
        # slice_range: tuple(start, end) along axial axis
        # is_train: whether to load segmentation masks
```

* **Loading**: Reads `.nii` files via `nibabel`.
* **Cropping**: Selects slices in `slice_range` along z-axis.
* **Normalization**: Z-score per scan.
* **Mask processing**: Converts raw segmentation masks into three channels (NCR/Net, ED, ET).
* **Visualization**: `visualize_sample(idx)` plots each modality and overlayed masks.

Example:

```python
from torch.utils.data import DataLoader
from preprocessing import BrainTumorDataset

dataset = BrainTumorDataset(
    data_dir='data/BraTS20_Training',
    modalities=['flair','t1','t1ce','t2'],
    slice_range=(40,190),
    is_train=True
)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
for imgs, masks in loader:
    # imgs: (batch, 4, H, W, D)
    # masks: (batch, 3, H, W, D)
    break
```

## Training & Evaluation

* **Model**: `ImprovedUNet3D` defined at the top of the script.
* **Loss**: `CombinedLoss` (Dice + CrossEntropy).
* **Optimizer**: AdamW with scheduler (`ReduceLROnPlateau`).
* **Metrics**: Dice coefficient per class.
* **EarlyStopping**: stops after 10 epochs without improvement.
* **Checkpointing**: Saves every 5 epochs to `models/` and uploads to Hugging Face.

To visualize training metrics:

```bash
tensorboard --logdir logs/
```


## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

* [BraTS Challenge](https://www.med.upenn.edu/sbia/brats2018.html) for the dataset.
* [U-Net 3D implementations](https://arxiv.org/abs/1606.06650).
* Kaggleforum and PyTorch community for examples and support.
