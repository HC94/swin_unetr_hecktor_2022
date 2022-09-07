"""
Ensembling of cross-validation models.
"""

import os
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from torchinfo import summary

from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.data import (
    DataLoader,
    Dataset,
)
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

from misc import create_folder_if_not_exists, sort_human

# Change working directory
root_path = '/data/pg-umcg_mii/'
os.chdir(root_path)

# Initialize variables
# Data
ct_a_min = -200
ct_a_max = 400
pt_a_min = 0
pt_a_max = 25
strength = 1  # Data aug strength
p = 0.5  # Data aug transforms probability
# Training
perform_test_run = False
n_channels = 2
n_classes = 3  # including background
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
seed = 0  # Seed for reproducibility
crop_samples = 2
input_size = [96, 96, 96]
max_epochs = 200
batch_size = 1
sliding_window_batch = 4
num_workers = 12
pin_memory = True if num_workers > 0 else False  # Do not change 
optimizer_name = 'adam_w'  # ['adam', 'adam_w', 'rmsprop', 'sgd', 'sgd_w', 'acc_sgd', 'ada_belief', 'ada_bound',
# 'ada_hessian', 'apollo', 'diff_grad', 'madgrad', 'novo_grad', 'pid', 'qh_adam', 'qhm', 'r_adam', 'ranger_qh',
# 'ranger_21', 'swats', 'yogi'].
lr = 1e-4
momentum=0 
weight_decay = 1e-5
T_0 = 40  # Cosine scheduler
epoch_early_stopping = 5
# Plotting
figsize = (18, 12)
nr_images = 8


# Paths
user_path = os.path.join(root_path, 'hung')
models_path = os.path.join(user_path, 'models')
predictions_path = os.path.join(user_path, 'predictions')
predictions_ensemble_path = os.path.join(user_path, 'predictions_ensemble')
data_path = os.path.join(root_path, 'data/hecktor2022_testing/resampled_larger')
cache_path = os.path.join(user_path, 'cache')
best_model_filename = 'best_model.pth'

# Variables for test run
if perform_test_run:
    n_train_testing = 3
    n_val_testing = 3
    crop_samples = 1
    max_epochs = 2
    batch_size = 1


# Data transforms
image_keys = ['ct', 'pt']  # Do not change
modes_3d = ['trilinear', 'trilinear']
modes_2d = ['bilinear', 'bilinear']

val_transforms = Compose(
    [
        LoadImaged(keys=image_keys),
        AddChanneld(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='RAS'),
        Spacingd(keys=image_keys, pixdim=(1, 1, 1), mode=modes_2d),
        ScaleIntensityRanged(keys=['ct'], a_min=ct_a_min, a_max=ct_a_max, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=['pt'], a_min=pt_a_min, a_max=pt_a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=image_keys, source_key='ct'),
        ToTensord(keys=image_keys),
    ]
)

def to_shape(a, shape):
    z_, y_, x_ = shape
    z, y, x = a.shape
    z_pad = (z_-z)
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a, ((z_pad//2, z_pad//2 + z_pad%2),
                      (y_pad//2, y_pad//2 + y_pad%2),
                      (x_pad//2, x_pad//2 + x_pad%2)),
                  mode='constant', constant_values=0)
    
# Dataset
all_files = sort_human(os.listdir(data_path))
patient_ids_ct = [x for x in all_files if 'CT' in x]
patient_ids_pt = [x.replace('CT', 'PT_nobrain') for x in patient_ids_ct]
patient_ids = [x.replace('__CT.nii.gz', '') for x in patient_ids_ct]  

# Generate segmentation maps of model from each fold
# Consider all data
val_ct = patient_ids_ct
val_pt = patient_ids_pt
val_patients = patient_ids

if perform_test_run:
    val_ct = val_ct[-n_val_testing:]
    val_pt = val_pt[-n_val_testing:]
    val_patients = val_patients[-n_val_testing:]

# Initialize DataLoader
val_dict = [{'ct': os.path.join(data_path, ct), 'pt': os.path.join(data_path, pt), 'patient_id': patient_id} for ct, pt, patient_id in zip(val_ct, val_pt, val_patients)]
# val_ds = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=1.0, num_workers=int(num_workers//2))
# val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(num_workers//2), pin_memory=pin_memory)
val_ds = Dataset(data=val_dict, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# Initialize  model
model = SwinUNETR(
    img_size=input_size,
    in_channels=n_channels,
    out_channels=n_classes,
    feature_size=48,
    use_checkpoint=True,
).to(device)

summary(model=model, input_size=[1, n_channels] + input_size, device=device)

# Get all models
model_folders = os.listdir(models_path)

print('model_folders:', model_folders)
for m in model_folders:
    print('Model:', model)
    # Load pretrained model
    model.load_state_dict(torch.load(os.path.join(models_path, m, best_model_filename), map_location=torch.device(device)))
    print('Trained model loaded!')

    # Create and save segmentation map predictions
    create_folder_if_not_exists(os.path.join(predictions_path, m))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            val_ct, val_pt, patient_id = (batch['ct'].to(device), batch['pt'].to(device), batch['patient_id'])
            print('patient_id:', patient_id)
            assert len(patient_id) == 1
            patient_id = patient_id[0]

            val_inputs = torch.concat([val_ct, val_pt], axis=1)
            val_outputs = sliding_window_inference(val_inputs, input_size, sliding_window_batch, model)  # output: (1, 3, X, X, X)
            torch.save(val_outputs, os.path.join(predictions_path, m, patient_id + '.pt'))

# Delete model to save memory
del model

# Create ensemble output
create_folder_if_not_exists(predictions_ensemble_path)

if perform_test_run:
    patient_ids = patient_ids[-n_val_testing:]

# Get all model predcitions
models = os.listdir(predictions_path)

nr_models = len(models)
print('Number of models: {}.'.format(nr_models))

for patient_id in patient_ids:

    val_outputs_ens = None
    # Load output of every model
    for i, m in enumerate(models):
        val_outputs_k = torch.load(os.path.join(predictions_path, m, patient_id + '.pt'))

        # Softmax
        val_outputs_k = torch.softmax(val_outputs_k, dim=1)

        if val_outputs_ens is None:
            val_outputs_ens = val_outputs_k
        else:
            val_outputs_ens += val_outputs_k

    # Take average of all model predictions (this is not necessary though, because we use argmax)
    val_outputs_ens = val_outputs_ens / nr_models

    # Argmax for final prediction
    pred_array = torch.squeeze(val_outputs_ens)
    pred_array = torch.argmax(pred_array, dim=0)
    pred_array = pred_array.detach().cpu().numpy()

    # save_array (for .nii.gz format)
    save_array = np.flip(np.flip(np.swapaxes(pred_array, 0, 2), axis=2), axis=1)

    # Prediction (ensemble)
    np_gtv = sitk.GetImageFromArray(save_array)
    sitk.WriteImage(np_gtv, os.path.join(predictions_ensemble_path, patient_id + '.nii.gz'))
