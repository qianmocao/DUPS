# Adversarial Patch: patch_utils
# utils for patch initialization and mask generation
# Created by Junbo Zhao 2020/3/19
from typing import Union, Any

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import cv2
import gc

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Initialize the patch
def patch_initialization(args):
    device = args.device
    image_path = args.patch_init
    patch_size = args.patch_size
    log_path = args.log_path
    batch_size = args.batch_size

    # generate patch mask and tensor
    # with Image.open(image_path) as img:
    #     img = img.convert('RGBA')
    #     to_tensor = transforms.Compose([
    #         # transforms.Resize(patch_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.48145466, 0.4578275, 0.40821073, 0.0),
    #                              (0.26862954, 0.26130258, 0.27577711, 1.0))
    #     ])
    #     img_tensor = to_tensor(img)
    #     patch_tensor = img_tensor[:3, :, :]
    # r, g, b, a = img.split()
    # alpha_array = np.array(a)
    # alpha_tensor = torch.tensor(alpha_array, dtype=torch.float32)
    # mask = (alpha_tensor > 0).float()
    #
    # mask_np = mask.squeeze(0).cpu().numpy()
    # save_path = os.path.join(log_path, 'mask_numpy.npy')
    # np.save(save_path, mask_np)

    mask_numpy = np.load('./patch_init/mask_numpy.npy')
    mask = torch.tensor(mask_numpy, dtype=torch.float32).expand(3, -1, -1).unsqueeze(0)
    mask = F.interpolate(mask, size=(patch_size, patch_size), mode='bilinear', align_corners=False)

    height, width = mask_numpy.shape
    patch_numpy = np.zeros((height, width, 3), dtype=np.uint8)
    patch_numpy[mask_numpy == 1] = [255, 0, 0]

    plt.figure(figsize=(15, 15))
    plt.imshow(patch_numpy)
    plt.axis('off')
    save_path = './logs/training_patches/red_patch_image.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    patch_tensor = torch.tensor(patch_numpy, dtype=torch.float32)
    patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0)
    patch_tensor = F.interpolate(patch_tensor, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
    mask = mask.to(device)
    patch_tensor = patch_tensor.to(device)

    patch_tensor = patch_tensor / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406], device=patch_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=patch_tensor.device).view(1, 3, 1, 1)

    patch_tensor = (patch_tensor - mean) / std

    return patch_tensor, mask

def apply_patch(images, patch, mask, alpha, patch_position, patch_size,image_size):
    """
    Apply a patch at the specified position of the image
    """
    device = images.device
    applied_patch = torch.zeros(image_size).to(device)
    mask_tensor = torch.zeros(image_size).to(device)
    batch_size, C, H, W = image_size

    for i in range(batch_size):
        best_x_location = patch_position[i, 0]
        best_y_location = patch_position[i, 1]
        applied_patch[i, :, best_x_location:best_x_location + patch_size,
        best_y_location:best_y_location + patch_size] = patch
        mask_tensor[i, :, best_x_location:best_x_location + patch_size,
        best_y_location:best_y_location + patch_size] = mask

    perturbated_image = mask_tensor.float() * (alpha * applied_patch.float()) + \
                        (1 - mask_tensor.float()) * images.float() + \
                        mask_tensor.float() * ((1 - alpha) * images.float())
    perturbated_image = perturbated_image.to(device)

    return perturbated_image

def random_position (patch,mask,patch_size,image_size):

    device = patch.device
    batch_size, _, H, W = image_size
    mask = mask.to(device)

    x_tensor = torch.randint(0, W - patch_size, (batch_size,)).to(device)
    y_tensor = torch.randint(0, H - patch_size, (batch_size,)).to(device)
    position = torch.stack((x_tensor, y_tensor), dim=1).to(device)
    applied_patch = torch.zeros(image_size).to(device)
    mask_tensor = torch.zeros(image_size).to(device)
    for i in range(batch_size):
        best_x_location = 0 #position[i, 0]
        best_y_location = 0 #position[i, 1]
        applied_patch[i, :, best_x_location:best_x_location + patch_size,
        best_y_location:best_y_location + patch_size] = patch
        mask_tensor[i, :, best_x_location:best_x_location + patch_size,
        best_y_location:best_y_location + patch_size] = mask

    return applied_patch, mask_tensor, position

# Test the patch on dataset  random_position
def test_patch(target_idx, patch, mask, test_loader, model, epoch, args):
    model.eval()
    log_path = args.log_path
    device = args.device
    class_num = args.class_num
    patch_size = args.patch_size

    test_total = 0
    test_actual_total = 0
    acc_origin = 0
    acc_attack = 0
    alpha = args.alpha
    res_acc = dict()
    res_attack = dict()
    test_acc = os.path.join(log_path, 'train_log/test_acc.txt')
    test_attack = os.path.join(log_path, 'train_log/test_attack.txt')
    os.makedirs(os.path.dirname(test_acc), exist_ok=True)
    print('\n ******** Test ********')
    for batch in tqdm(test_loader):
        images = batch['img'].to(device)
        labels = batch['label'].to(device)
        batch_size, _, _, _ = images.size()
        test_total += labels.shape[0]

        with torch.no_grad():
            _, res_batch = model(images)
            _, pred_batch = torch.max(res_batch, 1)
        # pred = pred.item()
        gt_batch = labels.to(device)
        # for pred, gt in zip(pred, gt):
        for pred, gt in zip(pred_batch, gt_batch):
            if gt.item() in res_acc:
                res_acc[gt.item()].append(1 if torch.equal(pred, gt) else 0)
            else:
                res_acc[gt.item()] = [1 if torch.equal(pred, gt) else 0]

        applied_patch, applied_mask, best_positions = random_position(patch,mask,patch_size,image_size=images.size())
        perturbated_image = applied_mask * alpha * applied_patch + (1 - applied_mask) * images + (1 - alpha) * applied_mask * images
        perturbated_image = perturbated_image.to(device)

        with torch.no_grad():
            _, patchs_res = model(perturbated_image)
            _, patch_res_batch = torch.max(patchs_res, 1)

        for batch_idx in range(batch_size):
            pred = pred_batch[batch_idx]
            gt = gt_batch[batch_idx]
            patch_res = patch_res_batch[batch_idx]
            if pred.item() == gt.item():
                acc_origin += 1
            if pred.item() == gt.item() and pred.item() != target_idx:
                test_actual_total += 1
            if pred.item() == gt.item() and pred.item() != target_idx and patch_res.item() != gt.item():
                acc_attack += 1
                if gt.item() in res_attack:
                    res_attack[gt.item()].append(1)
                else:
                    res_attack[gt.item()] = [1]
            if pred.item() == gt.item() and pred.item() != target_idx and patch_res.item() == gt.item():
                if gt.item() in res_attack:
                    res_attack[gt.item()].append(0)
                else:
                    res_attack[gt.item()] = [0]

    acc_array = np.zeros(class_num)
    attack_array = np.zeros(class_num)

    # Fill the accuracy array
    for key, values in res_acc.items():
        acc_array[key] = np.mean(values)
    
    # Fill the attack rate array
    for key, values in res_attack.items():
        attack_array[key] = np.mean(values)
    
    print('\nTest Accuracy%: ', end='')
    acc_str = [f'Class {i}: {int(acc_array[i] * 100)}% ' for i in range(class_num) if acc_array[i] > 0]
    print(''.join(acc_str))
    with open(test_acc, 'a') as f:
        f.write(f'Epoch {epoch} Accuracy:\n')
        # acc_lines = [f'Class {i}: {int(acc_array[i] * 100)}%\n' for i in range(class_num) if acc_array[i] > 0]
        f.writelines(acc_str)

    print('\nTest Attack%: ', end='')
    attack_str = [f'Class {i}: {int(attack_array[i] * 100)}% ' for i in range(class_num) if attack_array[i] > 0]
    print(''.join(attack_str))
    with open(test_attack, 'a') as f:
        f.write(f'Epoch {epoch} Attack:\n')
        # attack_lines = [f'Class {i}: {int(attack_array[i] * 100)}%\n' for i in range(class_num) if
        #                 attack_array[i] > 0]
        f.writelines(attack_str)

    return acc_origin / test_total, acc_attack / test_actual_total

def compute_uncertainty(model, images):
    """
    Calculate the uncertainty at the current patch position
    :param model: target neural network
    :param image: input image with patch
    :return: uncertainty measure of the model
    """
    with torch.no_grad():
        _, logits = model(images)
    # loss = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=1)
    uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return uncertainty

# Generate the position and apply the patch
def generate_random_point(image_size, patch_size, bbox, num_candidates=100):
    batch_size, _, H, W = image_size  # image_size 已知，例如 (3, 224, 224)
    bx, by, bw, bh = bbox  # 排除区域

    center_min = patch_size // 2
    center_max_x = W - patch_size // 2
    center_max_y = H - patch_size // 2

    # (num_candidates, 2)
    xs = torch.randint(center_min, center_max_x + 1, (num_candidates, 1))
    ys = torch.randint(center_min, center_max_y + 1, (num_candidates, 1))
    candidates = torch.cat((xs, ys), dim=1)  # 每行格式为 [center_x, center_y]

    # Filter candidate points
    mask = (candidates[:, 0] < bx) | (candidates[:, 0] >= bx + bw) | \
           (candidates[:, 1] < by) | (candidates[:, 1] >= by + bh)
    valid_candidates = candidates[mask]

    if valid_candidates.numel() == 0:
        chosen_center=[random.randint(0, W - patch_size // 2),random.randint(0, H - patch_size // 2)]
    else:
        random_idx = torch.randint(0, valid_candidates.size(0), (1,)).item()
        chosen_center = valid_candidates[random_idx].tolist()  # [cx, cy]

    patch_top_left_x = max(chosen_center[0] - patch_size // 2, 0)
    patch_top_left_y = max(chosen_center[1] - patch_size // 2, 0)
    point = [patch_top_left_x, patch_top_left_y]

    return point


def position_generation (images,patch,masks,boxes,
                         alpha,model, patch_size,image_size,
                         population_size=10,generations=100,F=0.5,CR=0.9):
    """
    Use differential evolution algorithm to optimize patch position
    :param model: target neural network
    :param image: input image (Tensor, C x H x W)
    :param label: true category (Tensor)
    :param population_size: population size
    :param generations: number of iterations
    :param F: mutation factor
    :param CR: crossover probability
    :param patch_size: patch size
    :return: optimal patch position
    """
    model.eval()
    device = images.device
    patch = patch.to(device)
    batch_size, _, H, W = image_size
    masks = masks.to(device)

    populations = [[generate_random_point(image_size, patch_size, boxes[idx])
                         for _ in range(population_size)]
                   for idx in range(batch_size)]
    populations = torch.tensor(populations).to(device)  # size=(batch_size,population_size,2)
    uncertaintys = []
    for p_i in range(population_size):
        p_images = apply_patch(images, patch, masks, alpha, populations[:, p_i, :], patch_size, image_size)
        # Calculate the loss of the initial population
        uncertainty = compute_uncertainty(model, p_images)
        uncertaintys.append(uncertainty)
    uncertaintys = torch.stack(uncertaintys).t()

    for _ in range(generations):
        for i in range(population_size):
            # Select three different individuals for mutation
            a, b, c = random.sample(range(population_size), 3)
            Xa = populations[:, a, :].to(device)
            Xb = populations[:, b, :].to(device)
            Xc = populations[:, c, :].to(device)

            # Mutation: Calculation V_i = X_a + F * (X_b - X_c)
            Vi = Xa + F * (Xb - Xc)  # (batch_size, 2)
            Vi = Vi.to(device)

            global_min = torch.zeros_like(Vi)
            global_max = torch.tensor([W - patch_size, H - patch_size], dtype=Vi.dtype, device=device).expand_as(Vi)
            Vi = torch.clamp(Vi, min=global_min, max=global_max)
            Vi = Vi.to(torch.float32) 
            boxes_tensor = torch.tensor(boxes, dtype=Vi.dtype, device=device)

            half = patch_size // 2
            center = Vi.to(dtype=torch.float32) + half
            bx, by, bw, bh = boxes_tensor.unbind(dim=1)
            cx, cy = center.unbind(dim=1)
            inside = (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)

            dl = cx - bx
            dr = (bx + bw) - cx
            dt = cy - by
            db = (by + bh) - cy

            new_cx = torch.where(inside,torch.where(dl < dr, bx, bx + bw),cx)
            new_cy = torch.where(inside,torch.where(dt < db, by, by + bh),cy)
            new_center = torch.stack([new_cx, new_cy], dim=1)

            Vi_new = new_center - half
            Vi_new[:, 0] = Vi_new[:, 0].clamp(0, W - patch_size)
            Vi_new[:, 1] = Vi_new[:, 1].clamp(0, H - patch_size)
            Vi = Vi_new.to(torch.int64)

            # Take out the current individual (target individual) for crossover, shape: (batch_size, 2)
            population = populations[:, i, :].to(device)

            # Cross: Generate a random tensor with the same shape as population and compare with CR
            cond = torch.rand(population.shape, device=device) < CR
            # Use torch.where to select the crossover result: if the condition is met, take Vi, otherwise take the original population
            new_individual = torch.where(cond, Vi, population).to(device)

            perturbated_image = apply_patch(images, patch, masks, alpha, new_individual, patch_size, image_size)

            new_uncertainty = compute_uncertainty(model, perturbated_image)

            new_uncertainty_tensor = new_uncertainty.clone().to(device)    #######
            uncertainty_tensor = uncertaintys.clone()[:, i].to(device)
            mask = new_uncertainty_tensor > uncertainty_tensor
            population_mask = mask.unsqueeze(1)  # shape: (8, 1)
            new_population = torch.where(population_mask, new_individual, population)
            new_uncertainty_list = torch.where(mask, new_uncertainty_tensor, uncertainty_tensor)
            populations[:, i, :] = new_population
            uncertaintys[:, i] = new_uncertainty_list

    max_indices = torch.argmax(uncertaintys, dim=1)  # (batch_size,)
    batch_indices = torch.arange(populations.size(0), device=device)  # (batch_size,)
    new_populations = populations[batch_indices, max_indices, :]  # (batch_size, 2)
    return new_populations

def test(target_idx, patch, mask, test_loader, model, epoch, args):
    model.eval()
    log_path = args.log_path
    device = args.device
    class_num = args.class_num
    patch_size = args.patch_size

    test_total = 0
    test_actual_total = 0
    acc_origin = 0
    acc_attack = 0
    alpha = args.alpha
    res_acc = dict()
    res_attack = dict()
    test_acc = os.path.join(log_path, 'train_log/test_acc.txt')
    test_attack = os.path.join(log_path, 'train_log/test_attack.txt')
    os.makedirs(os.path.dirname(test_acc), exist_ok=True)
    print('\n ******** Test ********')
    for batch in tqdm(test_loader):
        images = batch['img'].to(device)
        labels = batch['label'].to(device)
        boxes = batch['bbox']
        batch_size, _,_,_ = images.size()

        test_total += labels.shape[0]
        with torch.no_grad():
            _, res_batch = model(images)
            _, pred_batch = torch.max(res_batch, 1)
        gt_batch = labels.to(device)
        for pred, gt in zip(pred_batch, gt_batch):
            if gt.item() in res_acc:
                res_acc[gt.item()].append(1 if torch.equal(pred, gt) else 0)
            else:
                res_acc[gt.item()] = [1 if torch.equal(pred, gt) else 0]


        best_positions = position_generation(images, patch, mask,boxes,
                                             alpha, model, patch_size, image_size=images.size())

        perturbated_image = apply_patch(images, patch, mask, alpha,
                                           best_positions, patch_size,image_size=images.size())
       
        with torch.no_grad():
            _, patchs_res = model(perturbated_image)
            _, patch_res_batch = torch.max(patchs_res, 1)

        for batch_idx in range(batch_size):
            pred = pred_batch[batch_idx]
            gt = gt_batch[batch_idx]
            patch_res = patch_res_batch[batch_idx]
            if pred.item() == gt.item():
                acc_origin += 1
            if pred.item() == gt.item() and pred.item() != target_idx:
                test_actual_total += 1
            if pred.item() == gt.item() and pred.item() != target_idx and patch_res.item() != gt.item():
                acc_attack += 1
                if gt.item() in res_attack:
                    res_attack[gt.item()].append(1)
                else:
                    res_attack[gt.item()] = [1]
            if pred.item() == gt.item() and pred.item() != target_idx and patch_res.item() == gt.item():
                if gt.item() in res_attack:
                    res_attack[gt.item()].append(0)
                else:
                    res_attack[gt.item()] = [0]

    acc_array = np.zeros(class_num)
    attack_array = np.zeros(class_num)

    for key, values in res_acc.items():
        acc_array[key] = np.mean(values)

    for key, values in res_attack.items():
        attack_array[key] = np.mean(values)

    print('\nTest Accuracy%: ', end='')
    acc_str = [f'Class {i}: {int(acc_array[i] * 100)}% ' for i in range(class_num) if acc_array[i] > 0]
    print(''.join(acc_str))
    with open(test_acc, 'a') as f:
        f.write(f'Epoch {epoch} Accuracy:\n')
        # acc_lines = [f'Class {i}: {int(acc_array[i] * 100)}%\n' for i in range(class_num) if acc_array[i] > 0]
        f.writelines(acc_str)

    print('\nTest Attack%: ', end='')
    attack_str = [f'Class {i}: {int(attack_array[i] * 100)}% ' for i in range(class_num) if attack_array[i] > 0]
    print(''.join(attack_str))
    with open(test_attack, 'a') as f:
        f.write(f'Epoch {epoch} Attack:\n')
        # attack_lines = [f'Class {i}: {int(attack_array[i] * 100)}%\n' for i in range(class_num) if
        #                 attack_array[i] > 0]
        f.writelines(attack_str)

    return acc_origin / test_total, acc_attack / test_actual_total
