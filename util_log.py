import os
import torch
import argparse

{"num_classes": 16, "names": ["letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]}

def parse_args():
    parser = argparse.ArgumentParser(description='Document Classification Attack')
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--num_workers', type=int, default=5, help="num_workers")
    parser.add_argument('--noise_percentage', type=float, default=0.1,
                        help="percentage of the patch size compared with the image size")
    parser.add_argument('--p_threshold', type=float, default=0.9, help="minimum target probability")
    parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
    parser.add_argument('--max_iteration', type=int, default=100, help="max iteration")
    parser.add_argument('--target_idx', type=int, default=0, help="target label ")
    parser.add_argument('--max_epochs', type=int, default=200, help="total epoch")
    parser.add_argument('--eraly_stop', type=int, default=20, help="early stop")
    parser.add_argument('--weights', type=str, default='./rvl-16.model',
                        help="dir of the weights")
    parser.add_argument('--data_path', type=str, default='./RVL-CDIP/',
                        help="dir of the dataset")
    parser.add_argument('--class_num', type=int, default=16, help="class numbers")
    parser.add_argument('--patch_init', type=str, default='./patch_init/patch_init_red.png',
                        help="dir of the dataset")
    parser.add_argument('--patch_size', type=int, default=70, help="patch size")
    parser.add_argument('--alpha', type=float, default=1.0, help="watermark transparency")
    parser.add_argument('--data_aug', type=bool, default=False, help="data augmentation")
    parser.add_argument('--num_samples', type=int, default=20, help="num samples per class")
    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
    parser.add_argument('--gpuid', type=int, default=3, help="index pf used GPU")
    parser.add_argument('--seed', default=42)
    parser.add_argument('--fix_seed', type=bool, default=True)
    parser.add_argument('--log_path', type=str, default='./logs/', help='dir of the log')

    args = parser.parse_args()
    print(args)

    return args

class LOG:
    def __init__(self, file_name, epoch):
        if not os.path.exists('logs'):
            os.makedirs('logs')  
        self.file = open(f'logs/{file_name}_epoch-{epoch}.log', 'a')
        self.file.write('Groundtruth | Prediction | Image URL\n')

    def write(self, img_urls, gts, preds):
        for img_url, gt, pred in list(zip(img_urls, gts, preds)):
            self.file.write(f'{gt} | {pred} | {img_url}\n')


def save_model(model, epoch, prefix=None):
    if not os.path.exists('weights'):
        os.makedirs('weights')
    if prefix:
        name = f'weights/unlearn-{prefix}-rvl-{epoch}.model'
    else:
        name = f'weights/rvl-{epoch}.model'
    torch.save(model.state_dict(), name)
