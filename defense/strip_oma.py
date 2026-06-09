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
    """
    OMA 在你项目里的 test.py 逻辑是：
    trigger -> resize -> full-image add -> clip
    """
    trigger = atk_model(img_tensor)
    resized_trigger = resize_image(trigger, img_tensor[0][0].shape)
    atk_imgs = clip_image(img_tensor + resized_trigger * epsilon)
    return atk_imgs


@t.no_grad()
def mix_images(x, x_rand, alpha=0.5):
    """
    x, x_rand: [1, C, H, W]
    若尺寸不同，则将 x_rand resize 到 x 的尺寸
    """
    if x_rand.shape[2:] != x.shape[2:]:
        x_rand = resize_image(x_rand, x[0][0].shape)
    mixed = alpha * x + (1 - alpha) * x_rand
    mixed = clip_image(mixed)
    return mixed


def compute_entropy_for_image(detector, img_tensor, orig_size, top_k=20):
    """
    对单张图像计算 detector 输出 entropy。
    做法：
    1. forward 拿到 roi_scores
    2. softmax 得到类别分布
    3. 按 foreground confidence 选 top-k proposals
    4. 计算这些 proposals 的平均 entropy
    """
    scale = img_tensor.shape[3] / orig_size[1]

    roi_cls_locs, roi_scores, rois, roi_indices = detector(img_tensor, scale=scale)

    probs = t.softmax(roi_scores, dim=1)   # [R, 21]
    if probs.shape[0] == 0:
        return 0.0

    # background 是第 0 类，STRIP 这里更关注前景 proposals
    fg_conf, _ = probs[:, 1:].max(dim=1)

    k = min(top_k, probs.shape[0])
    top_idx = t.topk(fg_conf, k=k, dim=0).indices
    probs_top = probs[top_idx]

    entropy = -(probs_top * t.log(probs_top + 1e-8)).sum(dim=1).mean()
    return float(entropy.detach().cpu())


def strip_entropy(detector, x, dataset, sample_idx, num_mix=10, alpha=0.5, top_k=20):
    """
    对一张输入 x 做 STRIP：
    随机取 num_mix 张别的图混合，计算平均 entropy
    """
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
    # ========= 可改参数 =========
    detector_path = "checkpoints/OMA_s1_fasterrcnn_04152223_13_0.652332266852541_0.3109069755677748_0.9705882352941176"
    atk_model_path = "checkpoints/OMA_autoencoder_04152223_13_0.652332266852541_0.3109069755677748_0.9705882352941176"

    epsilon = 0.08
    num_samples = 1000
    num_mix = 30
    alpha = 0.5
    top_k = 10
    save_dir = "defense_results/strip_oma"
    # ===========================

    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)

    opt._parse({
        "dataset": "voc2007",
        "data_dir": "/root/autodl-tmp/invisible-backdoor-object-detection/VOCdevkit/VOC2007",
        "load_path": detector_path,
        "load_path_atk": atk_model_path,
        "atk_model": "autoencoder",
        "attack_type": "m",
        "target_class": 14,
        "epsilon": epsilon,
        "pretrained_model": "vgg16",
        "caffe_pretrain": False,
        "use_drop": False,
        "env": "strip_oma",
        "env2": "strip_oma2",
        "env3": "strip_oma3",
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
    for sample_idx in range(min(num_samples, len(dataset))):
        print(f"[{sample_idx+1}/{min(num_samples, len(dataset))}] processing...")

        img, size, gt_bbox, gt_label, difficult = dataset[sample_idx]
        img_tensor = t.from_numpy(img[None]).float().cuda()

        # clean
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

        # poisoned
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
    plt.title("OMA", fontsize=18, fontweight="bold")
    plt.xlim(xmin, xmax)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "strip_oma_hist.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("结果已保存到：", save_dir)


if __name__ == "__main__":
    main()