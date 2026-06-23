import time
import random
from torch.utils.data import DataLoader
from PIL import Image
from patch_utils import *
from util_log import parse_args
from dataset_all import loadData
from model_resnet import Model

{"num_classes": 16, "names": ["letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]}

def collate_batch(batch):
    new_batch = {k: [dic[k] for dic in batch] for k in batch[0]}
    new_batch['img'] = torch.tensor(np.array(new_batch['img'], dtype=np.float32))
    new_batch['label'] = torch.tensor(np.array(new_batch['label'], dtype=np.int64))
    return new_batch

def evl(args):
    batch_size = args.batch_size
    num_workers = args.num_workers
    weights = args.weights
    target_idx = args.target_idx
    log_path = args.log_path
    device = args.device

    data = loadData(data_path=args.data_path, data_aug=args.data_aug)

    dataloader_test = DataLoader(data['test'], batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, collate_fn=collate_batch)

    model = Model().to(device)
    if weights:
        checkpoint = torch.load(weights, map_location=device)
        model.load_state_dict(checkpoint)
        print(f'Saved weights loaded: {weights}')
    model.eval()

    # extract patch
    patch_numpy = np.load('./logs/train_patches_numpy/best_patch_numpy.npy')
    mask_numpy = np.load('./patch_init/mask_numpy.npy')
    print("Patch Image size:", patch_numpy.shape)
    patch_size = args.patch_size
    patch = torch.tensor(patch_numpy, dtype=torch.float32).unsqueeze(0)
    # patch_tensor = F.interpolate(patch_tensor, size=(patch_size, patch_size), mode='bilinear', align_corners=False)

    mask = torch.tensor(mask_numpy, dtype=torch.float32).expand(3, -1, -1).unsqueeze(0)
    mask = F.interpolate(mask, size=(patch_size, patch_size), mode='bilinear', align_corners=False)

    time_s = time.time()
    # acc_ori, acc_att = test_patch( target_idx, patch, mask, dataloader_test, model, epoch=1, args=args)
    acc_ori, acc_att = test( target_idx, patch, mask, dataloader_test, model, epoch=1, args=args)

    print(
        f"Epoch:acc_origin: {acc_ori * 100:.2f}%, acc_attack: {acc_att * 100:.2f}%, time: {(time.time() - time_s) / 60:.1f}min")
    # print(f"Epoch:{epoch} Patch attack success rate on trainset: {train_success_rate*100:.2f}%, acc_origin: {acc_ori*100:.2f}%, acc_attack: {acc_att*100:.2f}%, time: {time.time()-time_s:.1f}s")


if __name__ == '__main__':
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
    if args.fix_seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    evl(args)
