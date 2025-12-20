#%%
import argparse
from pathlib import Path
import logging
#%%
import sys
sys.path.append('../packages')
sys.path.append('../wrappers')
from dataset_wrapper import initialise_dataset
from wrapped_transformations import N4BiasFieldCorrectionCustomd
#%%
import numpy as np
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader
from monai.losses import DiceLoss
from monai.data import PersistentDataset
import torch
import torch.optim as optim
#%%
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ToTensord,
    AdjustContrastd,
    HistogramNormalized,
    NormalizeIntensityd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CenterSpatialCropd,
    SpatialPadd,
)
#%%
from monai.transforms import (
    RandFlipd,
    RandAffined,
    Rand3DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd
)
#%%
deterministic_transformations = Compose([
    LoadImaged(keys=["image", "seg_mask"]),  # loads NIfTI files
    EnsureChannelFirstd(keys=["image"]),
    EnsureChannelFirstd(keys="seg_mask"),
    EnsureTyped(keys="seg_mask", dtype=np.uint8),

# ----------------- Personal Transformations ----------------- #

    Orientationd(keys=["image", "seg_mask"], axcodes="RAS"),
    Spacingd(
        keys=["image", "seg_mask"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest")
    ),
    N4BiasFieldCorrectionCustomd(keys=["image"]),
    CenterSpatialCropd(keys=["image", "seg_mask"], roi_size=(160, 192, 160)),
    SpatialPadd(keys=["image","seg_mask"], spatial_size=(160,192,160)),  # pad to multiples of 16
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=3000,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    HistogramNormalized(keys=["image"], num_bins=256),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    AdjustContrastd(keys=["image"], gamma=1.2),

# ------------------------------------------------------------ #

    ToTensord(keys=["image", "seg_mask"])  # convert both to torch tensors
])
#%%
# Edit to change which transformations are used when augmenting training dataset
augmentation_transformations = Compose([

    # ---------------- Spatial (image + mask) ---------------- #

    # Left-right flip (safe for brain)
    RandFlipd(
        keys=["image", "seg_mask"],
        spatial_axis=0,
        prob=0.5
    ),

    # Small rotations, translations, and scaling
    RandAffined(
        keys=["image", "seg_mask"],
        prob=0.3,
        rotate_range=(0.1, 0.1, 0.1),     # ~±6°
        translate_range=(5, 5, 5),        # voxels
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest"),
        padding_mode="border"
    ),

    # Mild elastic deformation (simulates anatomy variance)
    Rand3DElasticd(
        keys=["image", "seg_mask"],
        prob=0.15,
        sigma_range=(5, 8),
        magnitude_range=(50, 100),
        mode=("bilinear", "nearest"),
        padding_mode="border"
    ),

    # ---------------- Intensity (image only) ---------------- #

    # Scanner noise
    RandGaussianNoised(
        keys=["image"],
        prob=0.2,
        mean=0.0,
        std=0.01
    ),

    # Slight smoothing (resolution variation)
    RandGaussianSmoothd(
        keys=["image"],
        prob=0.2,
        sigma_x=(0.5, 1.0),
        sigma_y=(0.5, 1.0),
        sigma_z=(0.5, 1.0)
    ),

    # Intensity scaling
    RandScaleIntensityd(
        keys=["image"],
        factors=0.1,
        prob=0.3
    ),

    # Intensity shift
    RandShiftIntensityd(
        keys=["image"],
        offsets=0.1,
        prob=0.3
    ),
])
#%%
modalities = ["flair", "t1", "t1ce", "t2"]

mod_dic = {}
for index, modality in enumerate(modalities):
    mod_dic[modality] = index
#%%
def main():

    parser = argparse.ArgumentParser(description='Converts .csv file of price data to a usable torch matrix (.pt)')

    parser.add_argument("-d", "--data_path", type=Path, default='../../data/', required=True, help="Path to BraTS2021 dataset")
    parser.add_argument("-a", "--cache", type=Path, default='../cache', required=True, help="Path to BraTS2021 dataset")

    parser.add_argument("-o", "--output", type=Path, default='../model/monai_3d_unet.pt', required=True, help="Path to model save file")
    parser.add_argument("-c", "--checkpoint", type=Path, default=None, help="Path to model checkpoint")

    parser.add_argument("-e", "--epochs", type=int, default=200, help="Number of training epochs")

    parser.add_argument("-l", "--log_output", type=Path, default='../log/train_baseline_unet_log.txt', help="Path to output log file")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    args = parser.parse_args()

    # set logging to desired verbosity
    if args.verbose:
        logging.basicConfig(
            filename=args.log_output,
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.info("Verbose mode activated")
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.warning("Script Running")

    dataset_access = initialise_dataset(args.data_path, modalities=modalities, transformations=deterministic_transformations)
    base_dataset = PersistentDataset(
        data=dataset_access.files,
        transform=dataset_access.transform,
        cache_dir=args.cache
    )

    train_dataset = Dataset(
        data=base_dataset,
        transform=augmentation_transformations
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=4,       # MRI modalities
        out_channels=1,      # 1 segmentation mask
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function = DiceLoss(to_onehot_y=False, softmax=True)
    optimiser = optim.Adam(model.parameters(), lr=1e-4)

    if args.checkpoint is not None:

        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

        logging.info("Checkpoint loaded")

    logging.warning("Beginning training")
    for epoch in range(args.epochs):

        model.train()
        epoch_loss = 0

        for batch_data in train_loader:

            inputs = batch_data["image"].to(device)
            labels = batch_data["seg_mask"].to(device)

            optimiser.zero_grad()

            outputs = model(inputs)   # logits

            loss = loss_function(outputs, labels)
            loss.backward()

            optimiser.step()

            epoch_loss += loss.item()

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict()
            }, args.output)

        logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        logging.info(f"Epoch {epoch+1} finished")

    logging.warning("Training finished")

#%%
if __name__ == "__main__":
    main()