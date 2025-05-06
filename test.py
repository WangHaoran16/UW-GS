import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import math
import matplotlib.pyplot as plt
import glob


def load_image(image_path):
    """加载图片并转换为张量."""
    image = Image.open(image_path).convert('RGB')
    return to_tensor(image)


def psnr(pred, gt):
    """计算PSNR值."""
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


generated_images_paths = glob.glob('D:/3DGS/output/underwaterwatch/train/ours_30000/renders/*.PNG')
gt_images_paths = glob.glob('D:/3DGS/output/underwaterwatch/train/ours_30000/gt/*.PNG')

generated_images_paths_freeze = glob.glob('D:/3DGS/output/freeze-underwaterwatch/train/ours_30000/renders/*.PNG')
gt_images_paths_freeze = glob.glob('D:/3DGS/output/freeze-underwaterwatch/train/ours_30000/gt/*.PNG')

generated_images_paths.sort()
gt_images_paths.sort()

generated_images_paths_freeze.sort()
gt_images_paths_freeze.sort()

assert len(generated_images_paths) == len(gt_images_paths), "图片数量不匹配"

psnr_values = []
psnr_values2 = []
for gen_path, gt_path, gen_path_f, gt_path_f in zip(generated_images_paths, gt_images_paths,
                                                    generated_images_paths_freeze, gt_images_paths_freeze):
    gen_image = load_image(gen_path)
    gt_image = load_image(gt_path)

    gen_image_f = load_image(gen_path_f)
    gt_image_f = load_image(gt_path_f)
    # 确保两张图片的尺寸相同
    assert gen_image.shape == gt_image.shape, "图片尺寸不匹配"

    current_psnr = psnr(gen_image, gt_image)
    current_psnr2 = psnr(gen_image_f, gt_image_f)
    psnr_values.append(current_psnr)
    psnr_values2.append(current_psnr2)

# average_psnr1 = sum(psnr_values) / len(psnr_values)
# average_psnr2 = sum(psnr_values2) / len(psnr_values2)
# print("original average PSNR:{}, freeze average PSNR:{}".format(average_psnr1,average_psnr2))

plt.figure(figsize=(10, 6))
plt.plot(psnr_values, marker='o', linestyle='-', color='b', label='without depth')
plt.plot(psnr_values2, marker='s', linestyle='-', color='r', label='with depth')
plt.title('PSNR Values for Image Pairs')
plt.xlabel('Image Pair Index')
plt.ylabel('PSNR Value')
plt.grid(True)
plt.show()
