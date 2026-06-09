import os
import sys
import random
import numpy as np
import torch as t
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import opt
from data.dataset import TestDataset
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model import AutoEncoder
from trainer import FasterRCNNTrainer
from utils.backdoor_tool import clip_image, resize_image


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


def load_detector(model_path):
    faster_rcnn = FasterRCNNVGG16(n_fg_class=20)
    trainer = FasterRCNNTrainer(faster_rcnn, opt=opt).cuda()
    trainer.load(model_path, load_optimizer=False)
    faster_rcnn = trainer.faster_rcnn
    faster_rcnn.eval()
    return faster_rcnn


def load_atk_model(atk_model_path):
    atk_model = AutoEncoder().cuda()
    atk_model.load(atk_model_path)
    atk_model.eval()
    return atk_model


def build_poisoned_image(img_tensor, atk_model, epsilon):
    trigger = atk_model(img_tensor)
    resized_trigger = resize_image(trigger, img_tensor[0][0].shape)
    atk_imgs = clip_image(img_tensor + resized_trigger * epsilon)
    return atk_imgs


@t.no_grad()
def mix_images(x, x_rand, alpha=0.5):
    if x_rand.shape[2:] != x.shape[2:]:
        x_rand = resize_image(x_rand, x[0][0].shape)
    mixed = alpha * x + (1 - alpha) * x_rand
    mixed = clip_image(mixed)
    return mixed


def compute_entropy_for_image(detector, img_tensor, orig_size, top_k=10):
    scale = img_tensor.shape[3] / orig_size[1]
    roi_cls_locs, roi_scores, rois, roi_indices = detector(img_tensor, scale=scale)

    probs = t.softmax(roi_scores, dim=1)
    if probs.shape[0] == 0:
        return 0.0

    fg_conf, _ = probs[:, 1:].max(dim=1)
    k = min(top_k, probs.shape[0])
    top_idx = t.topk(fg_conf, k=k, dim=0).indices
    probs_top = probs[top_idx]

    entropy = -(probs_top * t.log(probs_top + 1e-8)).sum(dim=1).mean()
    return float(entropy.detach().cpu())


def strip_entropy(detector, x, dataset, sample_idx, num_mix=30, alpha=0.5, top_k=10):
    _, orig_size, _, _, _ = dataset[sample_idx]
    entropies = []

    for _ in range(num_mix):
        rand_idx = random.randint(0, len(dataset) - 1)
        while rand_idx == sample_idx:
            rand_idx = random.randint(0, len(dataset) - 1)

        x_rand, _, _, _, _ = dataset[rand_idx]
        x_rand_tensor = t.from_numpy(x_rand[None]).float().cuda()

        mixed = mix_images(x, x_rand_tensor, alpha=alpha)
        ent = compute_entropy_for_image(detector, mixed, orig_size, top_k=top_k)
        entropies.append(ent)

    return float(np.mean(entropies))


def main():
    detector_path = "checkpoints/ODA_fasterrcnn_04141613_0_0.504398510956926_0.2084460517785102_0.9662162162162162"
    atk_model_path = "checkpoints/ODA_autoencoder_04141613_0_0.504398510956926_0.2084460517785102_0.9662162162162162"

    epsilon = 0.06
    num_samples = 1000
    num_mix = 30
    alpha = 0.5
    top_k = 10
    save_dir = "defense_results/strip_oda"

    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)

    opt._parse({
        "dataset": "voc2007",
        "data_dir": "/root/autodl-tmp/invisible-backdoor-object-detection/VOCdevkit/VOC2007",
        "load_path": detector_path,
        "load_path_atk": atk_model_path,
        "atk_model": "autoencoder",
        "attack_type": "d",
        "target_class": 14,
        "epsilon": epsilon,
        "pretrained_model": "vgg16",
        "caffe_pretrain": False,
        "use_drop": False,
        "env": "strip_oda",
        "env2": "strip_oda2",
        "env3": "strip_oda3",
        "port": 6006,
    })

    print("加载测试集...")
    dataset = TestDataset(opt)

    print("加载 detector...")
    detector = load_detector(detector_path)

    print("加载攻击模型...")
    atk_model = load_atk_model(atk_model_path)

    clean_entropy = []
    poison_entropy = []

    print("开始 STRIP...")
    total = min(num_samples, len(dataset))
    for sample_idx in range(total):
        print(f"[{sample_idx+1}/{total}] processing...")

        img, size, gt_bbox, gt_label, difficult = dataset[sample_idx]
        img_tensor = t.from_numpy(img[None]).float().cuda()

        clean_ent = strip_entropy(
            detector=detector,
            x=img_tensor,
            dataset=dataset,
            sample_idx=sample_idx,
            num_mix=num_mix,
            alpha=alpha,
            top_k=top_k
        )
        clean_entropy.append(clean_ent)

        atk_imgs = build_poisoned_image(img_tensor, atk_model, epsilon)
        poison_ent = strip_entropy(
            detector=detector,
            x=atk_imgs,
            dataset=dataset,
            sample_idx=sample_idx,
            num_mix=num_mix,
            alpha=alpha,
            top_k=top_k
        )
        poison_entropy.append(poison_ent)

        print(f"  clean_entropy  = {clean_ent:.4f}")
        print(f"  poison_entropy = {poison_ent:.4f}")

    clean_entropy = np.array(clean_entropy)
    poison_entropy = np.array(poison_entropy)

    np.save(os.path.join(save_dir, "clean_entropy.npy"), clean_entropy)
    np.save(os.path.join(save_dir, "poison_entropy.npy"), poison_entropy)

    print("clean mean:", clean_entropy.mean(), "std:", clean_entropy.std())
    print("poison mean:", poison_entropy.mean(), "std:", poison_entropy.std())

    plt.figure(figsize=(7, 4.8))

    all_vals = np.concatenate([clean_entropy, poison_entropy])
    xmin = float(all_vals.min()) - 0.02
    xmax = float(all_vals.max()) + 0.02
    bins = np.linspace(xmin, xmax, 26)

    plt.hist(clean_entropy, bins=bins, density=True, alpha=0.55, label="clean")
    plt.hist(poison_entropy, bins=bins, density=True, alpha=0.55, label="backdoor")

    plt.xlabel("entropy", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title("ODA", fontsize=18, fontweight="bold")
    plt.xlim(xmin, xmax)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "strip_oda_hist.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("结果已保存到：", save_dir)


if __name__ == "__main__":
    main()