import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import os
from PIL import Image
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform
class ImageProcessor:
    def __init__(self, root, max_num, total_max_num, input_size, use_thumbnail):
        # 数据根目录
        self.root = root
        # 每张图像生成的最大patch数量
        self.max_num = max_num
        # 所有图像生成patch的总数量上限
        self.total_max_num = total_max_num
        # 输入图像尺寸
        self.input_size = input_size
        # 是否使用缩略图
        self.use_thumbnail = use_thumbnail
        # 图像变换方法（如torchvision.transforms）
        self.transform = build_transform(input_size)
        # 动态调整图像尺寸的标志
        self.dynamic_image_size = True  # 假设启用动态尺寸（根据实际需求修改）
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio

def dynamic_preprocess_fix(image, min_num=1, max_num=20, image_size=448, use_thumbnail=False, output_dir='./out'):
    # 定义需要支持的分割模式
    split_modes = [
        {'rows': 2, 'cols': 2},
        {'rows': 4, 'cols': 4}
    ]

    all_processed_images = []

    for mode in split_modes:
        rows, cols = mode['rows'], mode['cols']

        # 计算目标宽高比
        target_aspect_ratio = (cols, rows)
        print(f"Processing {rows}x{cols} mode, target_aspect_ratio: {target_aspect_ratio}")

        # 调整图像尺寸
        target_width = image_size * cols
        target_height = image_size * rows
        resized_img = image.resize((target_width, target_height))

        # 分割图像
        blocks = rows * cols
        processed_images = []
        for i in range(blocks):
            x = (i % cols) * image_size
            y = (i // cols) * image_size
            box = (x, y, x + image_size, y + image_size)
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        # 添加缩略图（若需要）
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        # 保存结果
        mode_output_dir = os.path.join(output_dir, f"{rows}x{cols}")
        os.makedirs(mode_output_dir, exist_ok=True)
        for idx, img in enumerate(processed_images):
            file_name = f"{mode_output_dir}/split_{idx:04d}.jpg"
            img.save(file_name)

        all_processed_images.extend(processed_images)

    return all_processed_images
def dynamic_preprocess(image, min_num=1, max_num=20, image_size=448, use_thumbnail=False,output_dir = './out'):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    # target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_aspect_ratio = (2,2)
    print(type(target_aspect_ratio))
    print('target_aspect_ratio:', target_aspect_ratio)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)


    # 创建输出目录（若不存在）
    os.makedirs(output_dir, exist_ok=True)

    for idx, split_img in enumerate(processed_images):
        file_name = f"{output_dir}/split_{idx:04d}.jpg"
        split_img.save(file_name)  # 使用Pillow的save方法保存图像[2,7](@ref)

    return processed_images

def main(self):
    page_ids = ['1']
    image_list = []
    for page_id in page_ids:
        image_path = os.path.join(self.root, page_id + '.jpg')
        image = Image.open(image_path).convert('RGB')
        image_list.append(image)

    max_num = max(1, min(self.max_num, self.total_max_num // len(image_list)))
    num_patches_list = []
    if self.dynamic_image_size:
        images = []
        for image in image_list:
            tiles = dynamic_preprocess_fix(image, image_size=self.input_size,
                                       use_thumbnail=self.use_thumbnail,
                                       max_num=max_num)
            images += tiles
            num_patches_list.append(len(tiles))
    else:
        images = image_list
        num_patches_list.append(1)
    pixel_values = [self.transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

if __name__ == '__main__':
    root = 'images'
    input_size = 448
    max_num=20
    dynamic_image_size = True
    use_thumbnail = False
    total_max_num =20
    outoutput_dir = 'out_images'
    label_image = ImageProcessor(root, max_num, total_max_num, input_size, use_thumbnail)
    main(label_image)
