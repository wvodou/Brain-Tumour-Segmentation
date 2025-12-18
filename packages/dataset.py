#%%
# DO NOT TOUCH
import os
#%%
# DO NOT TOUCH
import torch
from torch.utils.data import Dataset

#%%
# DO NOT TOUCH
class BraTSDataset(Dataset):

    def __init__(self, root,  modalities, transform):

        self.root = root
        self.modalities = modalities
        self.transform = transform
        self.files = self.__init_files__()

    def __init_files__(self):
        patients = [
            p for p in sorted(os.listdir(self.root))
            if not p.startswith(".") and os.path.isdir(os.path.join(self.root, p))
        ]

        files = []

        for patient in patients:

            patient_path = os.path.join(self.root, patient)

            flair = os.path.join(patient_path, f"{patient}_flair.nii.gz").replace("\\", "/")
            t1   = os.path.join(patient_path, f"{patient}_t1.nii.gz").replace("\\", "/")
            t1ce = os.path.join(patient_path, f"{patient}_t1ce.nii.gz").replace("\\", "/")
            t2   = os.path.join(patient_path, f"{patient}_t2.nii.gz").replace("\\", "/")
            seg  = os.path.join(patient_path, f"{patient}_seg.nii.gz").replace("\\", "/")

            if all(os.path.exists(f) for f in [flair, t1, t1ce, t2, seg]):
                files.append({
                    "image": [flair, t1, t1ce, t2],
                    "seg_mask": seg
                })

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        sample = self.files[idx]
        sample = self.transform(sample)

        modality_map = {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}
        selected_modalities = [modality_map[m] for m in self.modalities]

        image = torch.as_tensor(sample["image"])
        image = image[selected_modalities,:,:,:]

        seg_mask = torch.as_tensor(sample["seg_mask"])

        return image, seg_mask
