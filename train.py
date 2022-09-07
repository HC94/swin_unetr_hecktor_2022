"""
Training of Swin UNETR model.
"""
import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from monai.utils import set_determinism
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    PersistentDataset,
    decollate_batch,
)

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotated,
    ToTensord,
)

from misc import copy_file, create_folder_if_not_exists, sort_human

# Change working directory
root_path = '/data/pg-umcg_mii/'
os.chdir(root_path)
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

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
num_workers = 12
pin_memory = True if num_workers > 0 else False  # Do not change 
optimizer_name = 'adam_w'  # ['adam', 'adam_w', 'rmsprop', 'sgd', 'sgd_w', 'acc_sgd', 'ada_belief', 'ada_bound',
# 'ada_hessian', 'apollo', 'diff_grad', 'madgrad', 'novo_grad', 'pid', 'qh_adam', 'qhm', 'r_adam', 'ranger_qh',
# 'ranger_21', 'swats', 'yogi'].
lr = 1e-4
momentum = 0
weight_decay = 1e-5
T_0 = 40  # Cosine scheduler
epoch_early_stopping = max_epochs
# Plotting
figsize = (18, 12)
nr_images = 8

# Paths
user_path = os.path.join(root_path, 'hung')
exp_path = os.path.join(user_path, 'exp')
exp_folder = os.path.join(user_path, 'exp', '{}_'.format(datetime_str) + '{}')
tb_path = os.path.join(user_path, 'tb')
data_path = os.path.join(root_path, 'data/hecktor2022/resampled_larger/')
cache_path = os.path.join(user_path, 'cache')
folds_path = os.path.join(root_path, 'wei/data/train_folds7.csv')
pretrained_path = os.path.join(root_path, 'hung', 'model_swinvit.pt')  # if None: no pretraining
best_model_filename = 'best_model.pth'

# Create empty folders if they do not exist yet
for f in [user_path, exp_path, tb_path, cache_path]:
    create_folder_if_not_exists(f)

# Variables for test run
if perform_test_run:
    n_train_testing = 4
    n_val_testing = 2
    crop_samples = 1
    max_epochs = 2
    batch_size = 1
    pretrained_path = None

# Data transforms
image_keys = ['ct', 'pt', 'gtv']  # Do not change
modes_3d = ['trilinear', 'trilinear', 'nearest']
modes_2d = ['bilinear', 'bilinear', 'nearest']
train_transforms = Compose(
    [
        LoadImaged(keys=image_keys),
        AddChanneld(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='RAS'),
        Spacingd(
            keys=image_keys,
            pixdim=(1, 1, 1),
            mode=modes_2d,
        ),
        ScaleIntensityRanged(
            keys=['ct'],
            a_min=ct_a_min,
            a_max=ct_a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=['pt'],
            a_min=pt_a_min,
            a_max=pt_a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=image_keys, source_key='ct'),
        RandCropByPosNegLabeld(
            keys=image_keys,
            label_key='gtv',
            spatial_size=input_size,
            pos=1,
            neg=1,
            num_samples=crop_samples,
            image_key='ct',
            image_threshold=0,
        ),
        RandAffined(keys=image_keys, prob=p,
                    translate_range=(round(10 * strength), round(10 * strength), round(10 * strength)),
                    padding_mode='border', mode=modes_2d),
        RandAffined(keys=image_keys, prob=p, scale_range=(0.10 * strength, 0.10 * strength, 0.10 * strength),
                    padding_mode='border', mode=modes_2d),
        RandFlipd(
            keys=image_keys,
            spatial_axis=[0],
            prob=p / 3,
        ),
        RandFlipd(
            keys=image_keys,
            spatial_axis=[1],
            prob=p / 3,
        ),
        RandFlipd(
            keys=image_keys,
            spatial_axis=[2],
            prob=p / 3,
        ),
        RandShiftIntensityd(
            keys=['ct', 'pt'],
            offsets=0.10,
            prob=p,
        ),
        ToTensord(keys=image_keys),
    ]
)

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

# Training and validation process
def validation(epoch_iterator_val, mode, device):
    print('Evaluation on {}.'.format(mode))
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_ct, val_pt, val_label = (batch['ct'].to(device), batch['pt'].to(device), batch['gtv'].to(device))
            val_inputs = torch.concat([val_ct, val_pt], axis=1)
            
            # Note: y_outputs is expected to have binarized predictions and y_label should be in one-hot format
            val_outputs = sliding_window_inference(val_inputs, input_size, 4, model)
            val_label_list = decollate_batch(val_label)
            val_label_convert = [post_label(val_label_tensor) for val_label_tensor in val_label_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            
            # Compute metric          
            dice_metric(y_pred=val_output_convert, y=val_label_convert)
            epoch_iterator_val.set_description('Validate (%d / %d Steps)' % (epoch, 10.0))
            
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(epoch, train_loader, val_loader, scheduler, dice_val_best, epoch_best, writer, nr_images, device):
    train_num_iterations = len(train_loader)

    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc='Training (X / X Steps) (loss=X.X)', dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        
        ct, pt, y = (batch['ct'].to(device), batch['pt'].to(device), batch['gtv'].to(device))
        x = torch.concat([ct, pt], axis=1)
        
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        
        # Scheduler: step() called after every batch update
        # scheduler.step(epoch + (i + 1) / train_num_iterations) is specifically for 'cosine' scheduler
        # Normally just use `scheduler.step()`
        scheduler.step(epoch + (step / train_num_iterations))
            
        epoch_iterator.set_description('Training (%d / %d Steps) (loss=%2.5f)' % (epoch, max_epochs, loss))
    
    # Evaluate after completing epoch
    # DiceCE loss
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    
    # Dice metric
    epoch_iterator_2 = tqdm(train_loader, desc='Training (X / X Steps) (loss=X.X)', dynamic_ncols=True)
    epoch_iterator_val = tqdm(val_loader, desc='Validate (X / X Steps) (dice=X.X)', dynamic_ncols=True)
    dice_train = validation(epoch_iterator_2, 'training data', device)
    dice_val = validation(epoch_iterator_val, 'validation data', device)
    dice_metric_values.append(dice_val)
    writer.add_scalar('Dice/train', dice_train, epoch)
    writer.add_scalar('Dice/val', dice_val, epoch)
                 
    if dice_val > dice_val_best:
        dice_val_best = dice_val
        epoch_best = epoch
        torch.save(model.state_dict(), best_model_path)
        print('Model saved. Current best avg. Dice: {}. Current avg. Dice: {}.'.format(dice_val_best, dice_val))
    else:
        print('Model not saved. Current best avg. Dice: {}. Current avg. Dice: {}.'.format(dice_val_best, dice_val))

    if (epoch % 10) == 0:
        # Check best model output with the input image and label
        # model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
        # TRAINING
        model.eval()
        with torch.no_grad():
            batch = next(iter(train_loader))
            val_ct, val_pt, val_label = (batch['ct'].to(device), batch['pt'].to(device), batch['gtv'].to(device))
            val_inputs = torch.concat([val_ct, val_pt], axis=1)
            val_outputs = sliding_window_inference(val_inputs, input_size, 4, model)              
                    
            # Determine slices to be plotted
            # Make sure that nr_images that we want to plot is greater than or equal to the number of slices available
            nr_slices = val_inputs.cpu().numpy().shape[-1]
            if nr_slices < nr_images:
                nr_images = nr_slices
            slice_indices = np.linspace(0, nr_slices - 1, num=nr_images)
            # Only consider unique values
            slice_indices = np.unique(slice_indices.astype(int))
            
            # Create images
            instance = random.randint(0, val_inputs.shape[0] - 1)
            for i, idx in enumerate(slice_indices):
                j = i+1
                plt.figure('Instance = {}'.format(instance), figsize=figsize)
                plt.subplot(4, nr_images, j)
                plt.title('CT ({})'.format(idx))
                plt.imshow(val_inputs.cpu().numpy()[instance, 0, :, :, idx], cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                
                plt.subplot(4, nr_images, nr_images + j)
                plt.title('PET ({})'.format(idx))
                plt.imshow(val_inputs.cpu().numpy()[instance, 1, :, :, idx], cmap='hot', vmin=0, vmax=1)
                plt.axis('off')
                
                plt.subplot(4, nr_images, 2*nr_images + j)
                plt.title('GTV ({})'.format(idx))
                plt.imshow(val_label.cpu().numpy()[instance, 0, :, :, idx])
                plt.axis('off')
                
                plt.subplot(4, nr_images, 3*nr_images + j)
                plt.title('Prediction ({})'.format(idx))
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[instance, :, :, idx])
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder_i, 'output_epoch_{}_train.png'.format(epoch)))
            plt.close()
            
        # Check best model output with the input image and label
        # model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
        # VALIDATION
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            val_ct, val_pt, val_label = (batch['ct'].to(device), batch['pt'].to(device), batch['gtv'].to(device))
            val_inputs = torch.concat([val_ct, val_pt], axis=1)
            val_outputs = sliding_window_inference(val_inputs, input_size, 4, model)              
                    
            # Determine slices to be plotted
            # Make sure that nr_images that we want to plot is greater than or equal to the number of slices available
            nr_slices = val_inputs.cpu().numpy().shape[-1]
            if nr_slices < nr_images:
                nr_images = nr_slices
            slice_indices = np.linspace(0, nr_slices - 1, num=nr_images)
            # Only consider unique values
            slice_indices = np.unique(slice_indices.astype(int))
            
            # Create images
            instance = random.randint(0, val_inputs.shape[0] - 1)
            for i, idx in enumerate(slice_indices):
                j = i+1
                plt.figure('Instance = {}'.format(instance), figsize=figsize)
                plt.subplot(4, nr_images, j)
                plt.title('CT ({})'.format(idx))
                plt.imshow(val_inputs.cpu().numpy()[instance, 0, :, :, idx], cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                
                plt.subplot(4, nr_images, nr_images + j)
                plt.title('PET ({})'.format(idx))
                plt.imshow(val_inputs.cpu().numpy()[instance, 1, :, :, idx], cmap='hot', vmin=0, vmax=1)
                plt.axis('off')
                
                plt.subplot(4, nr_images, 2*nr_images + j)
                plt.title('GTV ({})'.format(idx))
                plt.imshow(val_label.cpu().numpy()[instance, 0, :, :, idx])
                plt.axis('off')
                
                plt.subplot(4, nr_images, 3*nr_images + j)
                plt.title('Prediction ({})'.format(idx))
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[instance, :, :, idx])
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder_i, 'output_epoch_{}_val.png'.format(epoch)))
            plt.close()
    
    epoch += 1
    return epoch, dice_val_best, epoch_best
    
    
# Dataset
all_files = sort_human(os.listdir(data_path))
patient_ids_ct = [x for x in all_files if 'CT' in x]  # ['HMR-040__CT.nii.gz', 'HMR-041__CT.nii.gz', 'HMR-042__CT.nii.gz', ...]
patient_ids_pt = [x.replace('CT', 'PT_nobrain') for x in patient_ids_ct]
patient_ids_gtv = [x.replace('CT', 'gtv') for x in patient_ids_ct]
patient_ids = [x.replace('__CT.nii.gz', '') for x in patient_ids_ct]  

# Train model
df_folds = pd.read_csv(folds_path)
folds = set(df_folds['fold_center'])

# K-Fold cross-validation
for k in range(7):
    # Set seed
    torch.manual_seed(seed=seed)
    set_determinism(seed=seed)
    random.seed(a=seed)
    np.random.seed(seed=seed)

    print('Fold: {}'.format(k))
    exp_folder_i = exp_folder.format('Fold_{}'.format(k))
    figures_folder_i = os.path.join(exp_folder_i, 'figures')
    for folder in [exp_folder_i, figures_folder_i]:
        create_folder_if_not_exists(folder)
    best_model_path = os.path.join(exp_folder_i, best_model_filename)
    # Copy files
    for filename in ['main.py']:
        copy_file(src=os.path.join(user_path, filename), dst=os.path.join(exp_folder_i, filename))

    # Training-validation fold
    val_ct = [x for x, y in zip(patient_ids_ct, patient_ids) if df_folds[df_folds['ID'] == y]['fold_center'].values[0] == k]
    val_pt = [x for x, y in zip(patient_ids_pt, patient_ids) if df_folds[df_folds['ID'] == y]['fold_center'].values[0] == k]
    val_gtv = [x for x, y in zip(patient_ids_gtv, patient_ids) if df_folds[df_folds['ID'] == y]['fold_center'].values[0] == k]

    train_ct = [x for x in patient_ids_ct if x not in val_ct]
    train_pt = [x for x in patient_ids_pt if x not in val_pt]
    train_gtv = [x for x in patient_ids_gtv if x not in val_gtv]

    assert len(train_ct) == len(train_pt) == len(train_gtv)
    assert len(val_ct) == len(val_pt) == len(val_gtv)

    if perform_test_run:
        train_ct = patient_ids_ct[:n_train_testing]
        train_pt = patient_ids_pt[:n_train_testing]
        train_gtv = patient_ids_gtv[:n_train_testing]

        val_ct = patient_ids_ct[-n_val_testing:]
        val_pt = patient_ids_pt[-n_val_testing:]
        val_gtv = patient_ids_gtv[-n_val_testing:]

    # Initialize DataLoader
    train_dict = [{'ct': os.path.join(data_path, ct), 'pt': os.path.join(data_path, pt), 'gtv': os.path.join(data_path, gtv)} for ct, pt, gtv in zip(train_ct, train_pt, train_gtv)]
    train_ds = CacheDataset(data=train_dict, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
    # train_ds = PersistentDataset(data=train_dict, transform=train_transforms, cache_dir=cache_path)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    val_dict = [{'ct': os.path.join(data_path, ct), 'pt': os.path.join(data_path, pt), 'gtv': os.path.join(data_path, gtv)} for ct, pt, gtv in zip(val_ct, val_pt, val_gtv)]
    val_ds = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=1.0, num_workers=int(num_workers//2))
    # val_ds = PersistentDataset(data=val_dict, transform=val_transforms, cache_dir=cache_path)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(num_workers//2), pin_memory=pin_memory)

    # Initialize  model
    model = SwinUNETR(
        img_size=input_size,
        in_channels=n_channels,
        out_channels=n_classes,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    summary(model=model, input_size=[1, n_channels] + input_size, device=device)

    # Load pretrained model
    if pretrained_path is not None:
        weight = torch.load(pretrained_path, map_location=torch.device(device))
        model.load_from(weights=weight)
        print('Using pretrained weights!')

    # Optimizer and loss function
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0,
                                                                     T_mult=1, eta_min=1e-8)

    # Initalize classes
    writer = SummaryWriter(os.path.join(tb_path, '{}_'.format(datetime_str) + 'fold_{}'.format(k)))
    # Note: y_outputs/y_preds is expected to have binarized predictions and y_label should be in one-hot format
    post_label = AsDiscrete(to_onehot=n_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=n_classes)
    dice_metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)

    epoch = 0
    dice_val_best = 0.0
    epoch_best = 0
    epoch_loss_values = []
    dice_metric_values = []
    while (epoch < max_epochs) and (epoch - epoch_best <= epoch_early_stopping):
        epoch, dice_val_best, epoch_best = train(
            epoch, train_loader, val_loader, scheduler, dice_val_best, epoch_best, writer, nr_images, device
        )
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))

    print(
        f'Train completed, best_metric: {dice_val_best:.4f} '
        f'at iteration: {epoch_best}'
    )

    # Close Tensorboard writer
    writer.close()







