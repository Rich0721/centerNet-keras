from __future__ import division
import numpy as np
import cv2
from PIL import Image

from utils.transform import change_transform_origin


def read_image_bgr(path):

    image = np.asarray(Image.open(path).covert("RGB"))
    return image[:, :, ::-1].copy()


def preprocess_image(x, mode='tf'):

    '''
    Preprocess an image by substrcating the ImageNet mean.

    Args:
        x(array): (None, None, 3) or (3, None, None)
        model(string): 'tf' or 'caffe'
    Return:
        The input with the ImageNet mean subtractced
    '''

    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
    
    return x


def adjust_transform_for_image(transfom, image, relative_translation):
    """
    Adjust a transformation for specific image.
    """

    height, width, channels = image.shape
    result = transfom

    if relative_translation:
        result[0:2, 2] *= [width, height]
    
    result = change_transform_origin(transfom, (0.5*width, 0.5*height))
    
    return result


class TransformParameters:
    """
    Struct holding parameters determining how to apply a transformation to an image.

    Args:
        fill_mode(string): the fill method of image. 'constant', 'nearest', 'reflect', 'wrap'
        interpolation(string): the method of image to bigger or smaller. 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval(int): Fill value to use with fill_mode
        relative_translation(bool):  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """

    def __init__(self, fill_mode='nearest',
            interpolation='linear',
            cval=0,
            relative_translation=True):
        
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.cval = cval
        self.relative_translation = relative_translation
    
    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4
    

def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.
    The origin of transformation is at the top left corner of the image.
    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.
    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image, matrix[:2, :],
        dsize=(image.shape[0], image.shape[1]),
        flags=params.cvInterpolation(),
        borderMode=params.cvBorderMode(),
        borderValue=params.cval
    )
    return output


def compute_resize_scale(image_shape, min_side=800, max_side=1333):

    rows, cols, _ = image_shape

    smaller_side = min(rows, cols)

    scale = min_side / smaller_side

    lager_side = max(rows, cols)
    if lager_side * scale > max_side:
        scale = max_side / lager_side
    return scale


def resize_image(img, min_side=800, max_side=1333):

    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    return img, scale


def _uniform(val_range):
    return np.random.uniform(val_range[0], val_range[1])


def _check_range(val_range, min_val=None, max_val=None):
    if val_range[0] > val_range[1]:
        raise ValueError('interval lower bound > upper bound')
    if min_val is not None and val_range[0] < min_val:
        raise ValueError('invalid interval lower bound')
    if max_val is not None and val_range[1] > max_val:
        raise ValueError('invalid interval upper bound')


def _clip(image):
    return np.clip(image, 0, 255).astype(np.uint8)


class VisualEffect:
    def __init__(
            self,
            contrast_factor,
            brightness_delta,
            hue_delta,
            saturation_factor,
    ):
        self.contrast_factor = contrast_factor
        self.brightness_delta = brightness_delta
        self.hue_delta = hue_delta
        self.saturation_factor = saturation_factor

    def __call__(self, image):
        """
        Apply a visual effect on the image.
        Args
            image: Image to adjust
        """

        if self.contrast_factor:
            image = adjust_contrast(image, self.contrast_factor)
        if self.brightness_delta:
            image = adjust_brightness(image, self.brightness_delta)

        if self.hue_delta or self.saturation_factor:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if self.hue_delta:
                image = adjust_hue(image, self.hue_delta)
            if self.saturation_factor:
                image = adjust_saturation(image, self.saturation_factor)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image


def random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-.1, .1),
        hue_range=(-0.05, 0.05),
        saturation_range=(0.95, 1.05)
):
    _check_range(contrast_range, 0)
    _check_range(brightness_range, -1, 1)
    _check_range(hue_range, -1, 1)
    _check_range(saturation_range, 0)

    def _generate():
        while True:
            yield VisualEffect(
                contrast_factor=_uniform(contrast_range),
                brightness_delta=_uniform(brightness_range),
                hue_delta=_uniform(hue_range),
                saturation_factor=_uniform(saturation_range),
            )

    return _generate()


def adjust_contrast(image, factor):
    mean = image.mean(axis=0).mean(axis=0)
    return _clip((image - mean) * factor + mean)


def adjust_brightness(image, delta):
    return _clip(image + delta * 255)


def adjust_hue(image, delta):
    image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
    return image


def adjust_saturation(image, factor):
    image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
    return image