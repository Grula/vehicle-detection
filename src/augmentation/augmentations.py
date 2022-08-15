import csv
import random

import cv2
import numpy as np



def flip(image: np.array, **kwargs) -> np.array:
    """Flip image.
    Args:
        image (np.array): Image to flip.
        flipcode (int): Flip code. default is 1 (horizontally).
    Returns:
        np.array: Flipped image.
    """
    if kwargs['flipcode'] == 1:
        return cv2.flip(image, 1)
    return cv2.flip(image, 0)

def rotation(image: np.array, angle: int = 90) -> np.array:
    """Rotate image.
    Args:
        image (np.array): Image to rotate.
        angle (int): Angle to rotate image by.
    Returns:
        np.array: Rotated image.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    flags = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    flag = random.choice(flags)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=flag)
    return result
    
def zoom(image: np.array, scale: float = 0.5) -> np.array:
    """Scale image but keep original shape.
    Args:
        image (np.array): Image to scale.
        scale (float): Scale factor.
    Returns:
        np.array: Scaled image
    """
    original_shape = image.shape
    # if scale is less than 1, then image is scaled down
    scale_mat = cv2.getRotationMatrix2D((original_shape[1] / 2, original_shape[0] / 2), 0, scale)
    flags = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    flag = random.choice(flags)
    result = cv2.warpAffine(image, scale_mat, image.shape[1::-1], flags=flag, borderMode=cv2.BORDER_REPLICATE)
    return result

def crop(image: np.array, crop_size: int = 256) -> np.array:
    """Crop image.
    Args:
        image (np.array): Image to crop.
        crop_size (int): Size of crop.
    Returns:
        np.array: Cropped image.
    """
    row, col, ch = image.shape
    if crop_size > row or crop_size > col:
        crop_size = min(row, col)
    start_row = random.randint(0, row - crop_size)
    start_col = random.randint(0, col - crop_size)
    image = image[start_row:start_row + crop_size, start_col:start_col + crop_size]
    image = cv2.resize(image, (col, row))
    return image

def translation(image: np.array, translation_size: int = 10,) -> np.array:
    """Translate image.
    Args:
        image (np.array): Image to translate.
        translation_size (int): Size of translation.
    Returns
        np.array: Translated image.
    """
    row, col, ch = image.shape
    if translation_size > row or translation_size > col:
        translation_size = min(row, col)
    start_row = random.randint(0, row - translation_size)
    start_col = random.randint(0, col - translation_size)
    translation_mat = np.float32([[1, 0, start_col], [0, 1, start_row]])
    flags = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    flag = random.choice(flags)
    result = cv2.warpAffine(image, translation_mat, (col, row), flags=flag, borderMode=cv2.BORDER_REPLICATE)
    return result

def guassian_noise(image: np.array,) -> np.array:
    """Add guassian noise to image.
    Args:
        image (np.array): Image to add noise to.
    Returns:
        np.array: Image with noise.
    """
    row, col, ch = image.shape
    mean = 0.5
    gaussian = np.random.normal(mean, 0.1, (row, col, ch)).astype(np.float32)
    gaussian_img = cv2.addWeighted(image.astype(np.float32), 0.75, gaussian, 100, 0)
    gaussian_img = gaussian_img.astype(np.uint8)
    return gaussian_img

def guassian_blur(image: np.array, kernel_size = (7,7),) -> np.array:
    """Blur image with guassian blur.
    Args:
        image (np.array): Image to blur.
    Returns:
        np.array: Blurred image.
    """
    row, col, ch = image.shape
    gaussian_blur = cv2.GaussianBlur(image, kernel_size, 0)
    return gaussian_blur

def brightness(image: np.array, brightness_factor = 1.0,) -> np.array:
    """Change brightness of image.
    Args:
        image (np.array): Image to change brightness of.
    Returns:
        np.array: Image with changed brightness.
    """
    brightness_img = cv2.multiply(image, np.array([brightness_factor]))
    return brightness_img

def contrast(image: np.array,) -> np.array:
    """Change contrast of image.
    Args:
        image (np.array): Image to change contrast of.
    Returns:
        np.array: Image with changed contrast.
    """
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clipLimit = np.random.uniform(0.0, 10.0)
    tileGridSize = np.random.randint(1, 11,)
    tileGridSize = (tileGridSize, tileGridSize)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limage = cv2.merge((cl,a,b))
    # Converting image from LAB Color model to BGR color spcae
    enhanced_image = cv2.cvtColor(limage, cv2.COLOR_LAB2BGR)
    # Stacking the original image with the enhanced image
    return enhanced_image

def saturation(image: np.array, satmul = 1.0,) -> np.array:
    """Change saturation of image.
    Args:
        image (np.array): Image to change saturation of.
    Returns:
        np.array: Image with changed saturation.
    """
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s * satmul
    s = np.clip(s,0,255)
    imghsv = cv2.merge([h,s,v])
    image = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return image

def hue(image: np.array, huemul = 1.0,) -> np.array:
    """Change hue of image.
    Args:
        image (np.array): Image to change hue of.
    Returns:
        np.array: Image with changed hue.
    """
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    h = h * huemul
    h = np.clip(h,0,255)
    imghsv = cv2.merge([h,s,v])
    image = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return image


def augment_image(image: np.array, bbox : np.array) -> np.array:

    color_augmentations = ['brightness', 'contrast', 'saturation', 'hue']
    affine_augmentations = ['flip', 'translation', 'zoom']
    noise_augmentations = ['guassian_noise', 'guassian_blur']


    augmentation = color_augmentations + affine_augmentations + noise_augmentations
    augmentation = random.sample(augmentation, 5)

    for aug in augmentation:
        if aug == 'flip':
            bbox[:, 0] = image.shape[1] - bbox[:, 0]
            bbox[:, 2] = image.shape[1] - bbox[:, 2]
            arg = 1
        elif aug == 'translation':
            bbox[:, 0] = bbox[:, 0] + random.randint(-10, 10)
            bbox[:, 1] = bbox[:, 1] + random.randint(-10, 10)
            bbox[:, 2] = bbox[:, 2] + random.randint(-10, 10)
            bbox[:, 3] = bbox[:, 3] + random.randint(-10, 10)
            arg = 10
        elif aug == 'zoom':
            scale = random.uniform(0.8, 1.2)
            bbox[:, 0] = bbox[:, 0] * scale
            bbox[:, 1] = bbox[:, 1] * scale
            bbox[:, 2] = bbox[:, 2] * scale
            bbox[:, 3] = bbox[:, 3] * scale
            bbox[:, 0] = np.clip(bbox[:, 0], 0, image.shape[1])
            bbox[:, 1] = np.clip(bbox[:, 1], 0, image.shape[0])
            bbox[:, 2] = np.clip(bbox[:, 2], 0, image.shape[1])
            bbox[:, 3] = np.clip(bbox[:, 3], 0, image.shape[0])
            arg = scale
        image = eval(aug)(image, arg)
        

    
    return image, bbox


if __name__ == '__main__':
    
    #load test image
    image = cv2.imread('test.jpg')
    

    scale_percent = 20
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height))

    #add noise to image
    # augmented = add_noise(image, 'gaussian')
    
    #flip image horizontally
    aug_flip_1 = flip(image, 1)
    aug_flip_0 = flip(image, 0)
    img = np.hstack((image, aug_flip_0, aug_flip_1))
    cv2.imwrite('flip.jpg', img)


    #rotate image
    # augmented = rotate(image, np.random.randint(0, 360))
    aug_rot45 = rotation(image, 45)
    aug_rot120 = rotation(image, 120)
    img = cv2.hconcat([image, aug_rot45, aug_rot120])
    cv2.imwrite('rotate.jpg', img)


    #scale image
    aug_scale2 = zoom(image, 2)
    aug_scale0_5 = zoom(image, 0.5)
    img = cv2.hconcat([image, aug_scale2, aug_scale0_5])
    cv2.imwrite('scale.jpg', img)


    #crop image
    aug_crop256 = crop(image, 256)
    aug_crop128 = crop(image, 128)
    img = cv2.hconcat([image, aug_crop256, aug_crop128])
    cv2.imwrite('crop.jpg', img)

    #translate image
    aug_trans256 = translation(image, 256)
    aug_trans64 = translation(image, 64)
    img = cv2.hconcat([image, aug_trans256, aug_trans64])
    cv2.imwrite('translate.jpg', img)


    #add guassian noise to image
    aug_noise_1 = guassian_noise(image)
    aug_noise_2 = guassian_noise(image)
    img = cv2.hconcat([image, aug_noise_1, aug_noise_2])
    cv2.imwrite('noise.jpg', img)

    #blur image with guassian blur
    aug_blur7 = guassian_blur(image, (3,3))
    aug_blur11 = guassian_blur(image, (5,5))
    img = cv2.hconcat([image, aug_blur7, aug_blur11])
    cv2.imwrite('blur.jpg', img)

    #change brightness of image
    aug_brighness0_5 = brightness(image, 0.5)
    aug_brighness2 = brightness(image, 2.0)
    img = cv2.hconcat([image, aug_brighness0_5, aug_brighness2])
    cv2.imwrite('brightness.jpg', img)



    #change contrast of image
    aug_contrast_1 = contrast(image)
    aug_contrast_2 = contrast(image)
    img = cv2.hconcat([image, aug_contrast_1, aug_contrast_2])
    cv2.imwrite('contrast.jpg', img)


    #change saturation of image
    aug_sat1 = saturation(image, 1.5)
    aug_sat2 = saturation(image, 4.0)
    img = cv2.hconcat([image, aug_sat1, aug_sat2])
    cv2.imwrite('saturation.jpg', img)

    #change hue of image
    aug_hue_1 = hue(image, 0.5)
    aug_hue_2 = hue(image, 2.0)
    img = cv2.hconcat([image, aug_hue_1, aug_hue_2])
    cv2.imwrite('hue.jpg', img)