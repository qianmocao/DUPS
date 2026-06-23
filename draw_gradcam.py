from model_resnet import Model
from PIL import Image
# from patch_utils import *
import torchvision.transforms as transforms
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import random
from collections import defaultdict
from util_log import parse_args
import numpy as np
import torch
from patch_utils import *

{"num_classes": 16,
 "names": ["letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication",
           "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire",
           "resume", "memo"]}

TIF = {
"0":'imagesf/f/c/l/fcl54f00/0060341198.tif',
"1":'imagesf/f/n/k/fnk23d00/513161442.tif',
"2":'imagesg/g/t/u/gtu29c00/2084573574a.tif',
"3":'imagesu/u/d/f/udf65d00/504622315.tif',
"4":'imagesn/n/f/d/nfd02d00/71390160.tif',
"5":'imagesg/g/z/l/gzl07e00/2057996570.tif',
"6":'imageso/o/i/q/oiq36d00/50589658-9664.tif',
"7":'imagesw/w/a/n/wan45c00/2069716084.tif',
"8":'imagesi/i/r/w/irw77d00/2065345646.tif',
"9":'imagesw/w/z/y/wzy04f00/0000240413.tif',
"10":'imagese/e/v/e/eve72d00/83616873.tif',
"11":'imagesi/i/v/c/ivc27e00/2028704076.tif',
"12":'imagesz/z/b/t/zbt87d00/2078606169_6171.tif',
"13":'imagesg/g/j/m/gjm01f00/0011567510.tif',
"14":'imagesv/v/l/c/vlc6aa00/11307757.tif',
"15":'imagest/t/x/d/txd57e00/2030719694.tif'
}

def imagelist_txt(label_path,save_select_path,split='test'):
    # read data
    data_dict = defaultdict(list)
    file_path = os.path.join(label_path, f'{split}.txt')

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    with open(file_path, 'r') as f:
        for line in f:
            image_path, class_num = line.strip().split()
            class_num = int(class_num)
            data_dict[class_num].append(line.strip())

    # Randomly select 5 images from each category
    selected_images = []

    for num, images in data_dict.items():
        if len(images) >= 20:
            selected_images.extend(random.sample(images, 20))  # Randomly select images
        else:
            selected_images.extend(images)  #If there are less than X images, select all

    # write the selected image to a new file
    with open(save_select_path, 'w') as f:
        for image in selected_images:
            f.write(f"{image}\n")

def evl(args):
    device = args.device
    weights = args.weights

    model = Model(device=device).to(device)
    if weights:
        checkpoint = torch.load(weights, map_location=device)
        model.load_state_dict(checkpoint)
        print(f'Saved weights loaded: {weights}')
    model.eval()

    save_path = os.path.join('./patch_image_red71_00/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # extract patch
    patch_numpy = np.load('./logs/train_patches_numpy/best_patch_numpy.npy')
    mask_numpy = np.load('./patch_init/mask_numpy.npy')
    print("Patch Image size:", patch_numpy.shape)
    patch_size = args.patch_size
    file_path = save_select_path
    result_txt = os.path.join(save_path, 'test_log.txt')

    category_dict = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            image_path, category = line.strip().rsplit(' ', 1) 
            category = int(category) 
            category_dict[category].append(image_path) 

    test_total = 20*16
    test_actual_total = 0
    acc_origin = 0
    acc_attack = 0

    for category, images in category_dict.items():
        indx = 0
        for tif_path in images:
            class_num = category
            # tif_path = TIF[str(class_num)]
            image_path = os.path.join(datasets_path, tif_path)
            image = cv2.imread(image_path)
            image = np.float32(image) / 255
            image = cv2.resize(image, (224, 224))
            input_tensor = preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            input_tensor = input_tensor.cuda()
            # print("Original Image size:", input_tensor.shape)
            # GradCAM model traget layer name
            target_layers = [model.resnet.encoder.stages[3].layers[2].layer[0].convolution]
            targets = [ClassifierOutputTarget(class_num)]
            # cam = GradCAM(model=model, target_layers=target_layers)

            with (GradCAM(model=model, target_layers=target_layers) as cam):
                # Original Image Grad-CAM
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]
                cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                image_size = (3, 224, 224)

                model_outputs = torch.argmax(cam.outputs[1].detach(), dim=-1).cpu().numpy()
                # _, res = model(input_tensor.cuda())
                # _, model_outputs = torch.max(res, 1)
                boxes = []
                if model_outputs==class_num:
                    acc_origin += 1
                    test_actual_total += 1

                    mask = torch.tensor(mask_numpy, dtype=torch.float32).expand(3, -1, -1).unsqueeze(0)
                    mask = F.interpolate(mask, size=(patch_size, patch_size), mode='bilinear', align_corners=False)

                    heatmap = np.uint8(255 * grayscale_cam)
                    threshold_value = np.max(heatmap) * 0.8
                    _, binary_heatmap = cv2.threshold(heatmap, threshold_value, 255, cv2.THRESH_BINARY)

                    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        bboxes = [cv2.boundingRect(cnt) for cnt in contours]  # 每个边界框格式为 (x, y, w, h)
                        bbox = max(bboxes, key=lambda r: r[2] * r[3])
                        # x, y, w, h = bbox
                        boxes.append(bbox)
                    else:
                        bbox = (0, 0, 0, 0)
                        boxes.append(bbox)

                    patch_tensor = torch.tensor(patch_numpy, dtype=torch.float32).unsqueeze(0)
                    patch = F.interpolate(patch_tensor, size=(patch_size, patch_size), mode='bilinear',
                                                 align_corners=False)

                    applied_patch, applied_mask, best_positions = position_generation(input_tensor, patch, mask, boxes,
                                                                                  0.5, model, patch_size,
                                                                                  image_size=input_tensor.size())
                    input_patch_tensor = applied_mask * 0.5 * applied_patch + (1 - applied_mask) * input_tensor + (
                        1 - 0.5) * applied_mask * input_tensor


                    feat64, res = model(input_patch_tensor.cuda())
                    _, pred = torch.max(res, 1)
                    pred = pred.item()
                    tragets_pred = [ClassifierOutputTarget(pred)]
                    patch_cam = cam(input_tensor=input_patch_tensor, targets=tragets_pred)
                    # In this example grayscale_cam has only one image in the batch:
                    patch_cam = patch_cam[0, :]

                    input_patch_np = input_patch_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
                    input_patch_np = (input_patch_np - input_patch_np.min()) / (
                            input_patch_np.max() - input_patch_np.min())  # 归一化

                    cam_patch = show_cam_on_image(input_patch_np, patch_cam, use_rgb=True)
                    cam_patch = cv2.cvtColor(cam_patch, cv2.COLOR_RGB2BGR)

                    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = (image * 255).astype(np.uint8)
                    cam_patch = cv2.addWeighted(image, 0.5, cam_patch, 0.5, 0)

                    input_patch_uint8 = (input_patch_np * 255).astype('uint8')
                    input_patch_bgr = cv2.cvtColor(input_patch_uint8, cv2.COLOR_RGB2BGR)

                    if pred != class_num:
                        acc_attack +=1 
                        output_image_path = os.path.join(save_path, f'patch_image1/patch_image_class{class_num}_idx{indx}_pre{pred}.png')
                        patch_save = os.path.join(save_path, f'patch_image_cam1/patch_image_class{class_num}_idx{indx}_pre{pred}_cam.png')
                        image_save = os.path.join(save_path, f'original_image1/original_image_class{class_num}_idx{indx}_pre{model_outputs}.png')
                        original_cam_save = os.path.join(save_path, f'original_image_cam1/original_image_class{class_num}_idx{indx}_pre{model_outputs}_cam.png')
                    else:
                        output_image_path = os.path.join(save_path,f'patch_image0/patch_image_class{class_num}_idx{indx}_pre{pred}.png')
                        patch_save = os.path.join(save_path,f'patch_image_cam0/patch_image_class{class_num}_idx{indx}_pre{pred}_cam.png')
                        image_save = os.path.join(save_path, f'original_image0/original_image_class{class_num}_idx{indx}_pre{model_outputs}.png')
                        original_cam_save = os.path.join(save_path, f'original_image_cam0/original_image_class{class_num}_idx{indx}_pre{model_outputs}_cam.png')

                    os.makedirs(os.path.dirname(image_save), exist_ok=True)
                    os.makedirs(os.path.dirname(original_cam_save), exist_ok=True)
                    os.makedirs(os.path.dirname(patch_save), exist_ok=True)
                    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                    cv2.imwrite(image_save, image)
                    # cv2.imwrite(original_cam_save, cam_image)
                    x, y, w, h = boxes[0]
                    cv2.rectangle(cam_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imwrite(original_cam_save, cam_image)

                    cv2.imwrite(patch_save, cam_patch)
                    cv2.imwrite(output_image_path, input_patch_bgr)
                    indx += 1
    acc_ori = acc_origin / test_total
    acc_att = acc_attack / test_actual_total
    print(f"\n Test acc_origin: {acc_ori*100:.2f}%, acc_attack: {acc_att*100:.2f}%")
    with open(result_txt, 'a') as f:
        f.write(
            f"\n Test acc_origin: {acc_ori*100:.2f}%, acc_attack: {acc_att*100:.2f}%")

        # cv2.imshow('Image-CAM', cam_image)
        # cv2.imshow('Image', image)
        # cv2.imshow('Patch-Image-CAM', cam_patch)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':

    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    datasets_path = os.path.join(args.data_path,'images')
    label_path = os.path.join(args.data_path,'labels')
    # tif_path = 'imagesw/w/a/n/wan45c00/2069716084.tif'
    save_select_path = ('./gradcam_img.txt')
    imagelist_txt(label_path,save_select_path,split='test')
    
    evl(args)