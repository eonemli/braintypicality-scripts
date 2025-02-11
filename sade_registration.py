import os
import sys

import ants
import numpy as np
from absl import app, flags
from ml_collections.config_flags import config_flags
from sade.datasets.loaders import get_image_files_list
from sade.datasets.transforms import get_val_transform
from tqdm import tqdm

CACHE_DIR = "/codespace/braintypicality/dataset/template_cache/"

procd_ref_img_path = f"{CACHE_DIR}/cropped_niral_mni.nii.gz"

####
if not os.path.exists(procd_ref_img_path):
    T1_REF_IMG_PATH = os.path.join(
        CACHE_DIR, "mni_icbm152_09a/mni_icbm152_t1_tal_nlin_sym_09a.nrrd"
    )
    T2_REF_IMG_PATH = os.path.join(
        CACHE_DIR, "mni_icbm152_09a/mni_icbm152_t2_tal_nlin_sym_09a.nrrd"
    )
    MASK_REF_IMG_PATH = os.path.join(
        CACHE_DIR, "mni_icbm152_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nrrd"
    )

    ants_mni = ants.image_read(f"{CACHE_DIR}/croppedMni152.nii.gz")
    t1_ref_img = ants.image_read(T1_REF_IMG_PATH)
    t2_ref_img = ants.image_read(T2_REF_IMG_PATH)
    ref_img_mask = ants.image_read(MASK_REF_IMG_PATH)

    # Use ANTs' tighter cropping
    diff = np.array(t1_ref_img.shape) - np.array(ants_mni.shape)
    crop_idxs_start, crop_idxs_end = (
        1 + diff // 2,
        np.array(t1_ref_img.shape) - diff // 2,
    )

    t1_ref_img = ants.crop_indices(t1_ref_img, crop_idxs_start, crop_idxs_end)
    t2_ref_img = ants.crop_indices(t2_ref_img, crop_idxs_start, crop_idxs_end)
    ref_img_mask = ants.crop_indices(ref_img_mask, crop_idxs_start, crop_idxs_end)

    procd_ref_img = ants.merge_channels(
        (t1_ref_img * ref_img_mask, t2_ref_img * ref_img_mask)
    )

    procd_ref_img.to_filename(procd_ref_img_path)
####

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Sade configuration used to train the model.", lock_config=True)
flags.DEFINE_string("dataset", None, "The dataset for which to run the registration (should match names used in config and splits dir).")

def run_registration(argv):

    config = FLAGS.config
    dataset_name = FLAGS.dataset
    savedir = f"/DATA/Users/amahmood/braintyp/spacing_{int(config.data.spacing_pix_dim)}-ants"
    os.makedirs(savedir, exist_ok=True)

    img_loader = get_val_transform(config)
    procd_ref_img_path = f"{CACHE_DIR}/cropped_niral_mni.nii.gz"
    ref_img_tensor = img_loader({"image": procd_ref_img_path})["image"].numpy()
    ref_img_post_transform = (ants.from_numpy(ref_img_tensor[0]) + 1) / 2

    fnames = get_image_files_list(
        dataset_name, config.data.dir_path, config.data.splits_dir
    )

    for fname_dict in tqdm(fnames):
        img_tensor = img_loader(fname_dict)["image"].numpy()
        t1_img = (ants.from_numpy(img_tensor[0]) + 1) / 2

        fname = fname_dict['image']
        sampleid = os.path.basename(fname).split(".nii.gz")[0]

        if config.data.spacing_pix_dim == 1.0:
            # reg_iterations = [40, 20, 10, 0]
            reg_iterations = [500, 80, 40, 0]
        else:
            reg_iterations = [40, 20, 0]

        _ = ants.registration(
            fixed=ref_img_post_transform,
            moving=t1_img,
            # More info at https://github.com/ANTsX/ANTs/blob/master/Scripts/antsRegistrationSyN.sh
            type_of_transform="antsRegistrationSyN[s]",
            outprefix=f"{savedir}/{sampleid}",
            write_composite_transform=True,
            reg_iterations=reg_iterations,
            verbose=0,
        )

if __name__ == "__main__":
    app.run(run_registration)
