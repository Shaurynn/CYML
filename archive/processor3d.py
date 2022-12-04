import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nipype.interfaces.ants import RegistrationSynQuick
from nipype.interfaces.ants.segmentation import BrainExtraction
from ants import image_read

def preprocess3d(image, atlas):
    ants_image = image_read(image)
    slice_image = ants_image[50,:,:]
    plt.imshow(slice_image, cmap="gray")
    plt.savefig("./output/image.jpg")
    st.image("./output/image.jpg")
    st.success("Original MRI image read")

    reg = RegistrationSynQuick()
    reg.inputs.fixed_image = atlas
    reg.inputs.moving_image = os.path.join(".",image)
    reg.inputs.num_threads = 2
    reg.cmdline
    f"antsRegistrationSyNQuick.sh -d 3 -f {atlas} -r 32 -m {image} -n 2 -o ./ -p d"
    reg.run()
    ants_reg = image_read("./transformWarped.nii.gz")
    slice_ants_reg = ants_reg[:,:,50].T
    plt.imshow(slice_ants_reg, cmap="gray")
    plt.savefig("./output/reg_image.jpg")
    st.image("./output/reg_image.jpg")
    st.success("Brain registration complete")
    brainextraction = BrainExtraction()
    brainextraction.inputs.dimension = 3
    brainextraction.inputs.anatomical_image = os.path.join("./transformWarped.nii.gz")
    brainextraction.inputs.brain_template = os.path.join("./data/tpl-MNI305_desc-head_mask.nii.gz")
    brainextraction.inputs.brain_probability_mask = os.path.join("./data/tpl-MNI305_desc-brain_mask.nii.gz")
    brainextraction.cmdline
    f"antsBrainExtraction.sh -a ./transformWarped.nii.gz -m ./data/tpl-MNI305_desc-brain_mask.nii.gz -e ./data/tpl-MNI305_desc-head_mask.nii.gz -d 3 -o ./ -s nii.gz"
    brainextraction.run()
    ants_ss = image_read("./highres001_BrainExtractionBrain.nii.gz")
    slice_ants_ss = ants_ss[:,:,50].T
    plt.imshow(slice_ants_ss, cmap="gray")
    plt.savefig("./output/ss_image.jpg")
    st.image("./output/ss_image.jpg")
    st.success("Brain extraction complete")
    ants_ss[:,:,85]
    image_2d = ants_ss[:,:,85]
    np.save(f"./output/{image.split('/')[-1]}_2d", image_2d)
    return
