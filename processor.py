import SimpleITK as sitk
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from dltk.io import preprocessing
from nipype.interfaces import fsl
import tensorflow as tf
import os
import gzip
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from ants import image_read
from nipype.interfaces.ants import RegistrationSynQuick, BrainExtraction

import numpy as np
# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)

IMG_SHAPE = (78, 110, 86)
IMG_2D_SHAPE = (IMG_SHAPE[1] * 4, IMG_SHAPE[2] * 4)
#SHUFFLE_BUFFER = 5 #Subject to change
N_CLASSES = 3


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0]):
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def registrate(sitk_fixed, sitk_moving, bspline=False):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk_fixed)
    elastixImageFilter.SetMovingImage(sitk_moving)

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    if bspline:
        parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()
    return elastixImageFilter.GetResultImage()

def skull_strip_nii(original_img, destination_img, frac=0.2): #
    btr = fsl.BET()
    btr.inputs.in_file = original_img
    btr.inputs.frac = frac
    btr.inputs.out_file = destination_img
    btr.cmdline
    res = btr.run()
    return res

def gz_extract(zipfile):
    file_name = (os.path.basename(zipfile)).rsplit('.',1)[0]
    with gzip.open(zipfile,"rb") as f_in, open(f"{zipfile.split('/')[0]}/{file_name}","wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipfile)

def slices_matrix_2D(img):

  image_2D = np.empty(IMG_2D_SHAPE)

  TOP = 60
  BOTTOM = 30
  STEP = 2
  N_CUTS = 16
  cut_it = TOP
  row_it = 0
  col_it = 0

  for cutting_time in range(N_CUTS):

    cut = img[cut_it, :, :]
    cut_it -= STEP

    if cutting_time in [4, 8, 12]:
      row_it = 0
      col_it += cut.shape[1]

    for i in range(cut.shape[0]):
      for j in range(cut.shape[1]):
        image_2D[i + row_it, j + col_it] = cut[i, j]
    row_it += cut.shape[0]

  return np.repeat(image_2D[None, ...], 3, axis=0).T

def load_image_2D(abs_path): #, labels

  sitk_image = sitk.ReadImage(abs_path)
  img = sitk.GetArrayFromImage(sitk_image)
  img = preprocessing.whitening(img)
  img = slices_matrix_2D(img)

  return img

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


def write_tfrecords(x, y, filename):
    writer = tf.io.TFRecordWriter(filename)

    for image, label in zip(x, y):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(serialize_array(image)),
                'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

def _parse_image_function(example_proto):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.parse_tensor(features['image'], out_type=tf.double)
    image = tf.reshape(image, [344, 440, 3])

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(features['label'], 3)

    return image, label

def read_dataset(epochs, batch_size, filename):

    dataset = tf.data.TFRecordDataset(filename)

    dataset = dataset.map(_parse_image_function, num_parallel_calls=10)
    dataset = dataset.prefetch(batch_size)                      ##4
    dataset = dataset.repeat(epochs)                            ##2
    dataset = dataset.shuffle(buffer_size=10 * batch_size)      ##1
    dataset = dataset.batch(batch_size, drop_remainder=True)    ##3
    return dataset

def preprocess(image, atlas):
    sitk_image = sitk.ReadImage(image)
    arr_image = sitk.GetArrayFromImage(sitk_image)
    slice_image = arr_image[:,:,50].T
    plt.imshow(slice_image, cmap="gray")
    plt.savefig("./output/image.jpg")
    st.image("./output/image.jpg")
    st.success("Original MRI image read")

    res_image = resample_img(sitk_image)
    atlas_img = sitk.ReadImage(atlas)
    atlas_img = resample_img(atlas_img)

    registrated_image = registrate(atlas_img, res_image, bspline=False)
    sitk.WriteImage(registrated_image, f"./output/{image.split('/')[-1]}_registrated.nii")
    registrated_image = sitk.ReadImage(f"./output/{image.split('/')[-1]}_registrated.nii")
    registrated_array = sitk.GetArrayFromImage(registrated_image)
    slice_reg_image = registrated_array[50,:,:]
    plt.imshow(slice_reg_image, cmap="gray")
    plt.savefig("./output/image_reg.jpg")
    st.image("./output/image_reg.jpg")
    st.success("Brain registration complete")

    skull_strip_nii(f"./output/{image.split('/')[-1]}_registrated.nii", f"./output/{image.split('/')[-1]}_stripped.nii", frac=0.2)
    ss_image = sitk.ReadImage(f"./output/{image.split('/')[-1]}_stripped.nii")
    ss_array = sitk.GetArrayFromImage(ss_image)
    slice_ss_image = ss_array[50,:,:]
    plt.imshow(slice_ss_image, cmap="gray")
    plt.savefig("./output/image_ss.jpg")
    st.image("./output/image_ss.jpg")
    st.success("Brain extraction complete")

    gz_extract(f"./output/{image.split('/')[-1]}_stripped.nii.gz")
    image_2d = load_image_2D(f"./{image.split('/')[-1]}_stripped.nii")
    np.save(f"./output/{image.split('/')[-1]}_2d", image_2d)
    print("Image 2D conversion successfully completed")
    os.remove(f"./{image.split('/')[-1]}_stripped.nii")
    return

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


def predict(x, chosen_model):
    image_test_array = []
    label_test_array = []
    image_test_array.append(np.load(x))
    label_test_array.append(0)

    np_test_array = np.array(image_test_array)
    write_tfrecords(np_test_array, label_test_array, "./output/test.tfrecords")
    Test = read_dataset(10, 1, './output/test.tfrecords')
    print(Test)
    Test_array = list(Test.take(1).as_numpy_iterator())
    print(Test_array[0][0])
    print(chosen_model.summary())
    prediction = chosen_model.predict(Test_array[0][0])
    if np.argmax(prediction) == 2:
        st.info(f"Subject most likely **_has Alzheimer's disease_**.", icon="⚠️")
    elif np.argmax(prediction) == 1:
        st.warning(f"Subject most likely **_has mild cognitive impairment_**.", icon="⚠️")
    elif np.argmax(prediction) == 0:
        st.success(f"Subject most likely shows no cognitive impairment.", icon="✅")
    for infile in os.listdir("./input"):
        os.remove(os.path.join("./input", infile))
    for outfile in os.listdir("./output"):
        os.remove(os.path.join("./output", outfile))
    return
