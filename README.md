# Compressed Training Data for Brain Tumour Segmentation
Brain Tumour Segmentation project for Deep Learning Class at ETH Zurich

## Folder Setup
Your local machine MUST have the following folder layout for the code to run:

- `data/`
    - `BraTS2021_00000/`
    - `BraTS2021_00001/`
    - `BraTS2021_00002/`
    - ...
- `your_project_folder_connected_to_github/`
    - ...

Each `BraTS2021_XXXXX/` folder should contain:

- `BraTS2021_XXXXX_flair.nii.gz`
- `BraTS2021_XXXXX_seg.nii.gz`
- `BraTS2021_XXXXX_t1.nii.gz`
- `BraTS2021_XXXXX_t1ce.nii.gz`
- `BraTS2021_XXXXX_t2.nii.gz`

**IMPORTANT:**  
The `data/` folder must **NOT** be connected to git, as patient data is confidential and must remain local.

## Initialising Dataset
As the BraTS2021 dataset is memory intensive, data loading and unloading is done dynamically. A wrapper to the dataset class is provided.

Example usage:

```python
dataset = initialise_dataset(
    path_to_data="path/to/data",
    modalities=['flair', 't1'],
    transformations=some_monai_transformations
)

# extract patient data
patient_data = dataset.__getitem__(patient_id)

