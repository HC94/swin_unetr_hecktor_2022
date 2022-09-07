# Swin UNETR for tumor and lymph node delineation of multicentre oropharyngeal cancer patients with PET/CT imaging - [Hecktor 2022 challenge](https://hecktor.grand-challenge.org/)



This repository contains the code for the [Hecktor 2022 challenge](https://hecktor.grand-challenge.org/). It includes:
- Automatic extraction of bounding box coordinates (see `bounding_box.py`).
- Expanding the bounding box from size = 144 x 144 x 144 to size = 192 x 192 x 192 (see `resample_larger_nobrain.py`).
- Loading [pre-trained self-supervised Swin UNETR encoder](https://arxiv.org/abs/2111.14791) weights, and training and evaluation of the model using 
cross-validation on Hecktor 2022 challenge data (see `train.py`).
- Outputting segmentation map from each cross-validation model as well as an ensemble (see `ensemble.py`). 



The data augmentation and modeling were implemented using [Project MONAI 0.9](https://monai.io/) in
[PyTorch 1.10](https://pytorch.org/) with Python 3.8.



<h3>Abstract</h3>
<em>
Delineation of Gross Tumor Volume (GTV) is essential for the treatment of cancer with radiotherapy. 
GTV contouring is a time-consuming specialized manual task performed by radiation oncologists. 
Deep learning (DL) algorithms have shown potential in creating automatic delineations, reducing delineation time and 
inter-observer variation. The aim of this work was to create automatic delineations of primary tumors (GTVp) and 
pathological lymph nodes (GTVn) in oropharyngeal cancer patients with DL. Provided by the HECKTOR 2022 challenge, 
the data was 3D Computed Tomography (CT) and Positron Emission Tomography (PET) scans with ground-truth GTV delineations 
acquired from nine different centres were provided by the HECKTOR Challenge 2022. Our proposed DL algorithm inputs the 
3D image data, and is based on a Swin UNETR architecture. We applied transfer learning and used customized pre-processing 
functions to create an anatomic based region of interest. An average dice score of 0.656 was achieved on a test set of 359 
patients from the HECKTOR 2022 challenge.
</em>



<h3>References</h3>
1. Andrearczyk V, et al. ``Overview of the HECKTOR Challenge at MICCAI 2022: Automatic Head and 
Neck Tumor Segmentation and Outcome Prediction in PET/CT", in: Head and Neck Tumor Segmentation and Outcome Prediction, (2023).
2. Oreiller V., et al., ``Head and Neck Tumor Segmentation in PET/CT: The HECKTOR Challenge", 
Medical Image Analysis, 77:102336 (2022).