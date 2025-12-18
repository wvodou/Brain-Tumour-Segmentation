#%%
import SimpleITK as sitk
import numpy as np
from monai.transforms import MapTransform
#%%
class N4BiasFieldCorrectionCustomd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            corrected = []
            for c in range(img.shape[0]):
                sitk_img = sitk.GetImageFromArray(img[c])
                corrector = sitk.N4BiasFieldCorrection(sitk_img)
                corrected_img = sitk.GetArrayFromImage(corrector)
                corrected.append(corrected_img)
            d[key] = np.stack(corrected)
        return d
#%%
