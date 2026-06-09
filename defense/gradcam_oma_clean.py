# defense/gradcam_oma_clean.py

import os
import cv2
import torch as t
import torch.nn as nn
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.dataset import TestDataset, inverse_normalize
from utils.config import opt


class GradCAM:
    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, target_score):
        target_score.backward(retain_graph=True)

        if self.activations is None:
            raise RuntimeError("没有捕获到 activations，请检查 target_layer 是否选对。")
        if self.gradients is None:
            raise RuntimeError("没有捕获到 gradients，请检查 target_score 是否可反向传播。")

        gradients = self.gradients       # [1, C, H, W]
        activations = self.activations   # [1, C, H, W]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)

        cam = t.relu(cam)
        cam = cam[0].detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def find_last_conv(module):
    last_name = None
    last_layer = None
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_name = name
            last_layer = layer
    return last_name, last_layer


def to_rgb_uint8_from_chw(img_chw):
    """
    img_chw 是 TestDataset 输出的预处理后图像，形状一般是 C,H,W。
    inverse_normalize 用于还原到可视化图像。
    """
    img_vis = inverse_normalize(img_chw)
    img_vis = np.clip(img_vis, 0, 255)

    if img_vis.shape[0] == 3:
        img_vis = img_vis.transpose(1, 2, 0)

    return img_vis.astype(np.uint8)


def overlay_cam_on_image(img_rgb, cam):
    h, w = img_rgb.shape[:2]
    cam = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.45 * heatmap + 0.55 * img_rgb
    overlay = np.uint8(overlay)

    return overlay


def main():
    # ====== 你可以改这里 ======
    model_path = "checkpoints/OMA_s1_fasterrcnn_04152223_13_0.652332266852541_0.3109069755677748_0.9705882352941176"
    target_class = 14
    save_dir = "defense_results/gradcam_oma"
    num_samples = 10
    # =========================

    os.makedirs(save_dir, exist_ok=True)

    opt._parse({
        "dataset": "voc2007",
        "data_dir": "/root/autodl-tmp/invisible-backdoor-object-detection/VOCdevkit/VOC2007",
        "load_path": model_path,
        "pretrained_model": "vgg16",
        "caffe_pretrain": False,
        "use_drop": False,
        "env": "gradcam",
        "env2": "gradcam2",
        "env3": "gradcam3",
        "port": 6006,
    })

    print("加载 Faster R-CNN...")
    faster_rcnn = FasterRCNNVGG16(n_fg_class=20)
    trainer = FasterRCNNTrainer(faster_rcnn, opt).cuda()
    trainer.load(model_path, load_optimizer=False)

    faster_rcnn = trainer.faster_rcnn
    faster_rcnn.eval()

    last_conv_name, target_layer = find_last_conv(faster_rcnn.extractor)
    print("Grad-CAM target layer:", last_conv_name, target_layer)

    gradcam = GradCAM(target_layer)

    print("加载测试集...")
    testset = TestDataset(opt)

    for sample_index in range(num_samples):
        print("=" * 60)
        print("processing sample:", sample_index)

        img, size, gt_bbox, gt_label, difficult = testset[sample_index]

        img_tensor = t.from_numpy(img[None]).float().cuda()
        scale = img_tensor.shape[3] / size[1]

        faster_rcnn.zero_grad()

        roi_cls_locs, roi_scores, rois, roi_indices = faster_rcnn(img_tensor, scale=scale)

        target_scores = roi_scores[:, target_class]
        target_score = target_scores.max()

        print("sample:", sample_index, "target_score:", float(target_score.detach().cpu()))

        cam = gradcam.generate(target_score)

        img_rgb = to_rgb_uint8_from_chw(img)
        overlay = overlay_cam_on_image(img_rgb, cam)

        clean_path = os.path.join(save_dir, f"sample_{sample_index:03d}_clean.jpg")
        cam_path = os.path.join(save_dir, f"sample_{sample_index:03d}_clean_gradcam_class{target_class}.jpg")

        cv2.imwrite(clean_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print("保存完成：")
        print(clean_path)
        print(cam_path)

        t.cuda.empty_cache()

    gradcam.close()


if __name__ == "__main__":
    main()