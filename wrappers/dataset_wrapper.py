#%%
# DO NOT TOUCH
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord
import sys
sys.path.append('../packages')
from dataset import BraTSDataset
#%%
# DO NOT TOUCH
default_modalities = ["flair", "t1", "t1ce", "t2"]

default_transformations = Compose([
    LoadImaged(keys=["image", "seg_mask"]),      # loads NIfTI files
    EnsureChannelFirstd(keys=["image"]),              # ensures channel dimension exists
    ToTensord(keys=["image", "seg_mask"])  # convert both to torch tensors
])
#%%
# DO NOT TOUCH
def initialise_dataset(path, modalities=default_modalities, transformations=default_transformations):
    return BraTSDataset(path, modalities=modalities, transform=transformations)