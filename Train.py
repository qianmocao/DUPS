import time
import random
import os
from dataset import loadData
from model_resnet import Model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import gc
from patch_utils import *
from util_log import parse_args
import torch.nn.functional as F
import torch.nn as nn

def patch_attack(images, patch, mask, positions, patch_size, target, labels, model, lr, max_iteration=100, alpha=0.5):
    model.eval()

    target_probability, count = 0, 0
    tot_prob = []
    all_loss = []
    alpha = alpha

    images = images.detach()
    image_size = images.size()
    while count < max_iteration:
        count += 1
        # Optimize the patch
        patch = patch.detach().clone().requires_grad_(True)
        perturbated_image = apply_patch(images, patch, mask, alpha, positions, patch_size,image_size)

        _, output = model(perturbated_image)
        log_probs = F.log_softmax(output, dim=1)  # (batch_size, num_classes)
        loss_each = -log_probs[:, target]  # (batch_size,)
        pred = torch.argmax(output, dim=1)  # (batch_size,)

        success_mask = (pred != labels)  
        masked_loss = loss_each * success_mask.float()

        if success_mask.float().sum() > 0:
            loss_avg = masked_loss.sum() / (success_mask.float().sum() + 1e-8)
        else:
            loss_avg = torch.mean(loss_each)

        all_loss.append(loss_avg)
        loss_avg.backward()
        
        patch_grad = patch.grad.clone().cuda()
        patch.grad.data.zero_()
        patch = lr * patch_grad + patch.type(torch.FloatTensor).cuda()
        patch[:, 0, :, :] = torch.clamp(patch[:, 0, :, :], min=-3, max=3)

        # green (channel 1)
        min_g, max_g = -0.1, 0.1
        patch[:, 1, :, :] = torch.clamp(patch[:, 1, :, :], min=min_g, max=max_g)

        # blue (channel 2)
        min_b, max_b = -0.1, 0.1
        patch[:, 2, :, :] = torch.clamp(patch[:, 2, :, :], min=min_b, max=max_b)

        # Test the patch
        perturbated_image = apply_patch(images, patch, mask, alpha, positions, patch_size,image_size)
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.cuda()
        with torch.no_grad():
            _, output = model(perturbated_image)
            target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
            tot_prob.append(target_probability.item())
    return perturbated_image, patch, all_loss

def collate_batch(batch):
    new_batch = {k: [dic[k] for dic in batch] for k in batch[0]}
    new_batch['img'] = torch.tensor(np.array(new_batch['img'], dtype=np.float32))
    new_batch['label'] = torch.tensor(np.array(new_batch['label'], dtype=np.int64))
    return new_batch

def train(args):
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    class_num = args.class_num
    patch_size = args.patch_size
    weights = args.weights
    max_epochs = args.max_epochs
    eraly_stop = args.eraly_stop
    alpha = args.alpha
    target_idx = args.target_idx
    log_path = args.log_path
    p_threshold = args.p_threshold

    train_acc = os.path.join(log_path, 'train_log/train_acc.txt')
    train_attack = os.path.join(log_path, 'train_log/train_attack.txt')
    train_log = os.path.join(log_path, 'train_log/train_log.txt')
    test_log = os.path.join(log_path, 'train_log/test_log.txt')
    epoch_log = os.path.join(log_path, 'epoch_log/')
    train_patches_numpy = os.path.join(log_path, 'train_patches_numpy/')
    training_patches = os.path.join(log_path, 'training_patches/')
    os.makedirs(os.path.dirname(train_acc), exist_ok=True)
    os.makedirs(os.path.dirname(epoch_log), exist_ok=True)
    os.makedirs(os.path.dirname(train_patches_numpy), exist_ok=True)
    os.makedirs(os.path.dirname(training_patches), exist_ok=True)

    data = loadData(data_path=args.data_path, data_aug=args.data_aug)
    dataloader_train = DataLoader(data['train'], batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, collate_fn=collate_batch)
    dataloader_val = DataLoader(data['valid'], batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_batch)
    dataloader_test = DataLoader(data['test'], batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, collate_fn=collate_batch)

    model = Model(device=device).to(device)
    if weights:
        checkpoint = torch.load(weights, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        print(f'Saved weights loaded: {weights}')
    model.eval()

    patch, mask = patch_initialization(args)

    best_patch_epoch = 0
    best_patch_success_rate = 0
    stop_count = 0
    res_acc = dict()
    res_attack = dict()
    log_buffer = []
    print('\n  ******** Train ********')
    for epoch in range(1, max_epochs + 1):
        acc_origin = 0
        acc_attack = 0
        class_count = dict()
        train_total, train_actual_total, train_success = 0, 0, 0
        time_s = time.time()

        if epoch < 100:
            lr = args.lr
        else:
            lr = args.lr * (1 - (epoch - 100) / (max_epochs - 100))
            lr = max(lr, 0.0) 

        with open(f'{epoch_log}epoch-{epoch}.log', 'w') as out:
            for batch in tqdm(dataloader_train):
                images = batch['img'].to(device)
                labels = batch['label'].to(device)
                boxes = batch['bbox']
                # img_url = batch['img_url']
                train_total += images.shape[0]
                batch_size = images.shape[0]

                with torch.no_grad():
                    feat64, res = model(images)
                    _, pred_batch = torch.max(res, 1)

                gt_batch = labels
                for pred, gt in zip(pred_batch, gt_batch):
                    if gt.item() in res_acc:
                        res_acc[gt.item()].append(1 if torch.equal(pred, gt) else 0)
                    else:
                        res_acc[gt.item()] = [1 if torch.equal(pred, gt) else 0]

                best_positions = position_generation(images, patch, mask, boxes, alpha,
                                                     model, patch_size, image_size=images.size())

                perturbated_img, patch, loss = patch_attack(images, patch, mask, best_positions, patch_size,
                                                            target_idx, labels, model, lr,
                                                            max_iteration=100,
                                                            alpha=alpha)

                log_buffer.append(f'\n{len(loss)} iters: ' + ' '.join(f"{num:.2f}" for num in loss))
                out.write('\n'.join(log_buffer))
                log_buffer = []

                with torch.no_grad():
                    feat64, patchs_res = model(perturbated_img)
                    _, patch_res_batch = torch.max(patchs_res, 1)

                for batch_idx in range(batch_size):
                    pred = pred_batch[batch_idx]
                    gt = gt_batch[batch_idx]
                    patch_res = patch_res_batch[batch_idx]
                    if pred.item() == gt.item():
                        acc_origin += 1
                    if pred.item() == gt.item() and pred.item() != target_idx:
                        train_actual_total += 1
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
                patch = patch * mask

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        patch_numpy = patch.detach().squeeze(0).cpu().numpy()
        np.save(train_patches_numpy + f"Epoch-{epoch}_patch_numpy.npy", patch_numpy)  # save .npy
        plt.imshow(np.clip(np.transpose(patch_numpy, (1, 2, 0)) * std + mean, 0, 1))
        plt.axis('off')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.savefig(training_patches + f"Epoch-{epoch}_patch.png")

        print(
            f'\n [TRAIN] acc_origin (full): {acc_origin / train_total * 100:.2f}%, acc_attack (condition): {acc_attack / train_actual_total * 100:.2f}%, time: {(time.time() - time_s) / 60:.1f}min')

        res_acc_array = np.zeros(class_num)
        res_attack_array = np.zeros(class_num)

        for key, values in res_acc.items():
            res_acc_array[key] = np.mean(values)

        for key, values in res_attack.items():
            res_attack_array[key] = np.mean(values)

        print('\nTrain Accuracy%: ', end='')
        res_acc_str = [f'Class {i}: {int(acc * 100)}% ' for i, acc in enumerate(res_acc_array) if acc > 0]
        print(''.join(res_acc_str))
        with open(train_acc, 'a') as f:
            f.write(f'Epoch {epoch} Accuracy:\n')
            f.writelines(res_acc_str)

        print('\nTrain Attack%: ', end='')
        res_attack_str = [f'Class {i}: {int(attack * 100)}% ' for i, attack in enumerate(res_attack_array) if
                          attack > 0]
        print(''.join(res_attack_str))
        with open(train_attack, 'a') as f:
            f.write(f'Epoch {epoch} Attack:\n')
            f.writelines(res_attack_str)

        time_s = time.time()
        acc_ori, acc_att = test(target_idx, patch, mask, dataloader_val, model, epoch, args)
        print(
            f"\n Epoch:{epoch} Val acc_origin: {acc_ori * 100:.2f}%, acc_attack: {acc_att * 100:.2f}%, time: {(time.time() - time_s) / 60:.1f}min")
        with open(test_log, 'a') as f:
            f.write(
                f'\n Epoch {epoch} Val acc_origin: {acc_ori * 100:.2f}%, acc_attack: {acc_att * 100:.2f}%, time: {(time.time() - time_s) / 60:.1f}min')

        test_success_rate = acc_att
        if test_success_rate > best_patch_success_rate:
            stop_count = 0
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
            best_patch = patch
            np.save(train_patches_numpy + f"best_patch_numpy.npy", patch_numpy)
            plt.imshow(np.clip(np.transpose(patch_numpy, (1, 2, 0)) * std + mean, 0, 1))
            # plt.savefig("training_patches/best_patch.png")
            plt.axis('off')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.savefig("./logs/training_patches/best_patch.png", bbox_inches='tight', pad_inches=0)

            with open(test_log, 'a') as f:
                f.write(
                    f'\n Best Epoch {epoch} Test acc_origin: {acc_ori * 100:.2f}%, acc_attack: {acc_att * 100:.2f}%, time: {(time.time() - time_s) / 60:.1f}min')
        else:
            stop_count += 1

        if stop_count >= eraly_stop:
            print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch,
                                                                                                100 * best_patch_success_rate))
            # break

    time_s = time.time()
    acc_ori, acc_att = test(target_idx, best_patch, mask, dataloader_test, model, epoch, args)
    print(
        f"\n Epoch:{epoch} Test acc_origin: {acc_ori * 100:.2f}%, acc_attack: {acc_att * 100:.2f}%, time: {(time.time() - time_s) / 60:.1f}min")
    # print(f"Epoch:{epoch} Patch attack success rate on trainset: {train_success_rate*100:.2f}%, acc_origin: {acc_ori*100:.2f}%, acc_attack: {acc_att*100:.2f}%, time: {time.time()-time_s:.1f}s")
    with open(train_log, 'a') as f:
        f.write(
            f'\n Epoch {epoch} Test acc_origin: {acc_ori * 100:.2f}%, acc_attack: {acc_att * 100:.2f}%, time: {(time.time() - time_s) / 60:.1f}min')


if __name__ == '__main__':
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)

    if args.fix_seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    train(args)
