import csv
import random

import cv2
import numpy as np



def _flip(image: np.array, **kwargs) -> np.array:
    """Flip image.
    Args:
        image (np.array): Image to flip.
    Returns:
        np.array: Flipped image.
    """
    if kwargs['flipcode'] == 1:
        return cv2.flip(image, 1)
    return cv2.flip(image, 0)

def _rotation(image: np.array, **kwargs) -> np.array:
    """Rotate image.
    Args:
        image (np.array): Image to rotate.
        angle (int): Angle to rotate image by.
    Returns:
        np.array: Rotated image.
    """
    angle = 90 if 'angle' not in kwargs else kwargs['angle']

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    flags = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    flag = random.choice(flags)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=flag)
    return result
    
def _zoom(image: np.array, **kwargs) -> np.array:
    """Scale image but keep original shape.
    Args:
        image (np.array): Image to scale.
        scale (float): Scale factor.
    Returns:
        np.array: Scaled image
    """
    scale = 0.5 if 'scale' not in kwargs else kwargs['scale']

    original_shape = image.shape
    # if scale is less than 1, then image is scaled down
    scale_mat = cv2.getRotationMatrix2D((original_shape[1] / 2, original_shape[0] / 2), 0, scale)
    flags = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    flag = random.choice(flags)
    result = cv2.warpAffine(image, scale_mat, image.shape[1::-1], flags=flag, borderMode=cv2.BORDER_REPLICATE)
    return result

def _crop(image: np.array, **kwargs) -> np.array:
    """Crop image.
    Args:
        image (np.array): Image to crop.
        crop_size (int): Size of crop.
    Returns:
        np.array: Cropped image.
    """
    crop_size = 16 if 'crop_size' not in kwargs else kwargs['crop_size']

    row, col, ch = image.shape
    if crop_size > row or crop_size > col:
        crop_size = min(row, col)
    start_row = random.randint(0, row - crop_size)
    start_col = random.randint(0, col - crop_size)
    image = image[start_row:start_row + crop_size, start_col:start_col + crop_size]
    image = cv2.resize(image, (col, row))
    return image

def _translation(image: np.array, **kwargs) -> np.array:
    """Translate image.
    Args:
        image (np.array): Image to translate.
        translation_size (int): Size of translation.
    Returns
        np.array: Translated image.
    """
    translation_size = 10 if 'translation_size' not in kwargs else kwargs['translation_size']

    row, col, ch = image.shape
    if translation_size > row or translation_size > col:
        translation_size = min(row, col)
    start_row = random.randint(0, translation_size)
    start_col = random.randint(0, translation_size)
    translation_mat = np.float32([[1, 0, start_row], [0, 1, start_col]])
    flags = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    flag = random.choice(flags)
    result = cv2.warpAffine(image, translation_mat, (col, row), flags=flag, borderMode=cv2.BORDER_REPLICATE)
    return result

def _guassian_noise(image: np.array, **kwargs) -> np.array:
    """Add guassian noise to image.
    Args:
        image (np.array): Image to add noise to.
    Returns:
        np.array: Image with noise.
    """
    mean = 0 if 'mean' not in kwargs else kwargs['mean']
    stddev = 0.1 if 'stddev' not in kwargs else kwargs['stddev']

    row, col, ch = image.shape
    gaussian = np.random.normal(mean, stddev, (row, col, ch)).astype(np.float32)
    image = image / 127.5 - 1
    noisy_image = image + gaussian
    noisy_image = np.clip(noisy_image, -1, 1)
    noisy_image = (noisy_image + 1) * 127.5
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def _guassian_blur(image: np.array, **kwargs) -> np.array:
    """Blur image with guassian blur.
    Args:
        image (np.array): Image to blur.
    Returns:
        np.array: Blurred image.
    """
    ksize = 7 if 'ksize' not in kwargs else kwargs['ksize']

    row, col, ch = image.shape
    gaussian_blur = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return gaussian_blur

def _brightness(image: np.array, **kwargs) -> np.array:
    """Change brightness of image.
    Args:
        image (np.array): Image to change brightness of.
    Returns:
        np.array: Image with changed brightness.
    """
    brightness = 1.0 if 'brightness' not in kwargs else kwargs['brightness']

    brightness_img = cv2.multiply(image, np.array([brightness]))
    return brightness_img

def _contrast(image: np.array, **kwargs) -> np.array:
    """Change contrast of image.
    Args:
        image (np.array): Image to change contrast of.
    Returns:
        np.array: Image with changed contrast.
    """
    clipLimit = np.random.uniform(0.0, 10.0) if 'clipLimit' not in kwargs else kwargs['clipLimit']
    tileGridSize = np.random.randint(1, 11,) if 'tileGridSize' not in kwargs else kwargs['tileGridSize']


    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:


    tileGridSize = (tileGridSize, tileGridSize)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limage = cv2.merge((cl,a,b))
    # Converting image from LAB Color model to BGR color spcae
    enhanced_image = cv2.cvtColor(limage, cv2.COLOR_LAB2BGR)
    # Stacking the original image with the enhanced image
    return enhanced_image

def _saturation(image: np.array, **kwargs) -> np.array:
    """Change saturation of image.
    Args:
        image (np.array): Image to change saturation of.
    Returns:
        np.array: Image with changed saturation.
    """
    sat_delta = 1.0 if 'sat_delta' not in kwargs else kwargs['sat_delta']

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s * sat_delta
    s = np.clip(s,0,255)
    imghsv = cv2.merge([h,s,v])
    image = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return image

def _hue(image: np.array, **kwargs) -> np.array:
    """Change hue of image.
    Args:
        image (np.array): Image to change hue of.
    Returns:
        np.array: Image with changed hue.
    """
    hue_delta = 1.0 if 'hue_delta' not in kwargs else kwargs['hue_delta']

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    h = h * hue_delta
    h = np.clip(h,0,255)
    imghsv = cv2.merge([h,s,v])
    image = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return image


def augment_image(image: np.array, bbox : np.array) -> np.array:
    """
    Augment image with random transformations.
    Args:
        image (np.array): Image to augment.
        bbox (np.array): Bounding box of image. [x, y, w, h]
    Returns:
        Augmented image and bounding box.
    """
    color_augmentations = [_brightness, _contrast, _saturation, _hue]
    affine_augmentations = [_flip, _translation, _zoom]
    noise_augmentations = [_guassian_noise, _guassian_blur]


    augmentation =  random.sample(color_augmentations,2) +\
                [random.choice(affine_augmentations)] +\
                [random.choice(noise_augmentations)]

    kwargs = {'flipcode': 1,
              'translation_size': 10,
              'scale' : random.uniform(0.95, 1),
              'rotation': random.randint(0, 3),
              'brightness': random.uniform(0.5, 1.5),
              'sat_delta': random.uniform(0.0, 10.0),
              'hue_delta': random.uniform(0.0, 10.0),
            }

    for aug in augmentation:
        # bbox is type np.array, [x, y, w, h]
        if 'flip' in aug.__name__ : # flip the coordiantes horizontally
            bbox[0] = image.shape[1] - bbox[0] - bbox[2]
            kwargs['flipcode'] = 1
        elif 'translation' in aug.__name__: # translate the coordinates by a random amount
            bbox[0] += random.randint(-kwargs['translation_size'], kwargs['translation_size'])
            bbox[0] = max(0, min(image.shape[1] ,bbox[0]))
            bbox[1] += random.randint(-kwargs['translation_size'], kwargs['translation_size'])
            bbox[1] = max(0, min(image.shape[0] ,bbox[1]))
        elif 'zoom' in aug.__name__:  # scale the coordiantes by a random amount
            bbox[0] = int(bbox[0] * kwargs['scale'])
            bbox[0] = max(0, min(image.shape[1] ,bbox[0]))
            
            bbox[1] = int(bbox[1] * kwargs['scale'])
            bbox[1] = max(0, min(image.shape[0] ,bbox[1]))

            bbox[2] = int(bbox[2] * kwargs['scale'])
            bbox[3] = int(bbox[3] * kwargs['scale'])

        image = aug(image, **kwargs)
        

    
    return image, bbox


if __name__ == '__main__':
    
    #load test image
    image = cv2.imread('test.jpg')
    image = cv2.resize(image, (512, 512))

    # test guassian noise
    image = _guassian_noise(image)
    cv2.imshow('guassian noise', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # for _ in range(5):
    #     tmp = augment_image(image, np.array([[0, 0, image.shape[1], image.shape[0]]]))
    #     cv2.imshow('image', tmp[0])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
