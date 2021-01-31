import torch
import torchvision
import torchvision.transforms.functional as F

from PIL import Image

import math
import numbers
from collections.abc import Sequence


class Mask_Aug():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
    

class ToTensor:
    """
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    Only applied to image not mask.
    """
    def __call__(self, image, mask):
        return F.to_tensor(image), mask


class PILToTensor:
    """
    Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    Only applied to mask not image.
    """
    def __call__(self, image, mask):
        return image, F.pil_to_tensor(mask)


class ToPILImage:
    """
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image while preserving the value range.
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, image, mask):
        return F.to_pil_image(image, self.mode), F.to_pil_image(mask, self.mode)


class Normalize(torch.nn.Module):
    """
    Normalize a tensor image with mean and standard deviation.
    Only applied to image not mask.
    """
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, image, mask):
        return F.normalize(image, self.mean, self.std, self.inplace), mask


class Resize(torch.nn.Module):
    """
    Resize the input image to the given size.
    """
    def __init__(self, size, image_interpolation=Image.BILINEAR, mask_interpolation=Image.NEAREST):
        super().__init__()
        if not isinstance(size, (int, Sequence)): raise TypeError('Size should be int or sequence. Got {}'.format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2): raise ValueError('If size is a sequence, it should have 1 or 2 values')
        self.size = size
        self.image_interpolation = image_interpolation
        self.mask_interpolation = mask_interpolation

    def forward(self, image, mask):
        return F.resize(image, self.size, self.image_interpolation), F.resize(mask, self.size, self.mask_interpolation)


class CenterCrop(torch.nn.Module):
    """
    Crops the given image at the center.
    """
    def __init__(self, size):
        super().__init__()
        self.size = _setup_size(size, error_msg='Please provide only two dimensions (h, w) for size.')

    def forward(self, image, mask):
        return F.center_crop(image, self.size), F.center_crop(mask, self.size)



class Pad(torch.nn.Module):
    """
    Pad the given image on all sides with the given pad value.
    Only applied to image not mask.
    """
    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__()
        if not isinstance(padding, (numbers.Number, tuple, list)): raise TypeError('Got inappropriate padding arg')
        if not isinstance(fill, (numbers.Number, str, tuple)): raise TypeError('Got inappropriate fill arg')
        if padding_mode not in ['constant', 'edge', 'reflect', 'symmetric']: raise ValueError('Padding mode should be either constant, edge, reflect or symmetric')
        if isinstance(padding, Sequence) and len(padding) not in [1, 2, 4]: raise ValueError('Padding must be an int or a 1, 2, or 4 element tuple, not a {} element tuple'.format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, image, mask):
        return F.pad(image, self.padding, self.fill, self.padding_mode), mask


class RandomCrop(torch.nn.Module):
    """
    Crop the given image at a random location.
    """
    @staticmethod
    def get_params(image, output_size):
        w, h = F._get_image_size(image)
        th, tw = output_size

        if (h+1 < th) or (w+1 < tw): raise ValueError('Required crop size {} is larger then input image size {}'.format((th, tw), (h, w)))
            
        if w == tw and h == th: return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw


    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        super().__init__()
        self.size = tuple(_setup_size(size, error_msg='Please provide only two dimensions (h, w) for size.'))
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, image, mask):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            mask = F.pad(mask, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(image)
        
        # pad if needed
        if self.pad_if_needed and (width < self.size[1]):
            padding = [self.size[1] - width, 0]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.fill, self.padding_mode)

        if self.pad_if_needed and (height < self.size[0]):
            padding = [0, self.size[0] - height]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)


class RandomHorizontalFlip(torch.nn.Module):
    """
    Horizontally flip the given image randomly with a given probability.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image, mask):
        if torch.rand(1) < self.p:
            return F.hflip(image), F.hflip(mask)
        return image, mask


class RandomVerticalFlip(torch.nn.Module):
    """
    Vertically flip the given image randomly with a given probability.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image, mask):
        if torch.rand(1) < self.p:
            return F.vflip(image), F.vflip(mask)
        return image, mask


class RandomPerspective(torch.nn.Module):
    """
    Performs a random perspective transformation of the given image with a given probability.
    """
    @staticmethod
    def get_params(width, height, distortion_scale):
        half_height = height // 2
        half_width = width // 2
        topleft = [int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()), 
                   int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())]
        topright = [int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()), 
                    int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())]
        botright = [int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
                    int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())]
        botleft = [int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
                   int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints
    
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BILINEAR, fill=0):
        super().__init__()
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale
        self.fill = fill

    def forward(self, image, mask):
        if torch.rand(1) < self.p:
            width, height = F._get_image_size(image)
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return F.perspective(image, startpoints, endpoints, self.interpolation, self.fill), F.perspective(mask, startpoints, endpoints, self.interpolation, self.fill)
        return image, mask


class RandomResizedCrop(torch.nn.Module):
    """
    Crop the given image to random size and aspect ratio.
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F._get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), 
                 image_interpolation=Image.BILINEAR, mask_interpolation=Image.NEAREST):
        super().__init__()
        self.size = _setup_size(size, error_msg='Please provide only two dimensions (h, w) for size.')

        if not isinstance(scale, Sequence): raise TypeError('Scale should be a sequence')
        if not isinstance(ratio, Sequence): raise TypeError('Ratio should be a sequence')
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]): warnings.warn('Scale and ratio should be of kind (min, max)')

        self.image_interpolation = image_interpolation
        self.mask_interpolation = mask_interpolation
        self.scale = scale
        self.ratio = ratio

    def forward(self, image, mask):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        return F.resized_crop(image, i, j, h, w, self.size, self.image_interpolation), F.resized_crop(mask, i, j, h, w, self.size, self.mask_interpolation)


class ColorJitter(torch.nn.Module):
    """
    Randomly change the brightness, contrast and saturation of an image.
    Only applied to image not mask.
    """
    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0: raise ValueError('If {} is a single number, it must be non negative.'.format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and (len(value) == 2):
            if not bound[0] <= value[0] <= value[1] <= bound[1]: raise ValueError('{} values should be between {}'.format(name, bound))
        else: raise TypeError('{} should be a single number or a list/tuple with lenght 2.'.format(name))

        if value[0] == value[1] == center:
            value = None
        return value
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def forward(self, image, mask):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if (fn_id == 0) and (self.brightness is not None):
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                image = F.adjust_brightness(image, brightness_factor)

            if (fn_id == 1) and (self.contrast is not None):
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                image = F.adjust_contrast(image, contrast_factor)

            if (fn_id == 2) and (self.saturation is not None):
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                image = F.adjust_saturation(image, saturation_factor)

            if (fn_id == 3) and (self.hue is not None):
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                image = F.adjust_hue(image, hue_factor)

        return image, mask


class RandomRotation(torch.nn.Module):
    """
    Rotate the image by angle.
    """
    @staticmethod
    def get_params(degrees):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle
    
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        super().__init__()
        self.degrees = _setup_angle(degrees, name='degrees', req_sizes=(2, ))

        if center is not None:
            _check_sequence_input(center, 'center', req_sizes=(2, ))

        self.center = center
        self.resample = resample
        self.expand = expand
        self.fill = fill

    def forward(self, image, mask):
        angle = self.get_params(self.degrees)
        return F.rotate(image, angle, self.resample, self.expand, self.center, self.fill), F.rotate(mask, angle, self.resample, self.expand, self.center, self.fill)


class RandomAffine(torch.nn.Module):
    """
    Random affine transformation of the image keeping center invariant.
    """
    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear
    
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0):
        super().__init__()
        self.degrees = _setup_angle(degrees, name='degrees', req_sizes=(2, ))

        if translate is not None:
            _check_sequence_input(translate, 'translate', req_sizes=(2, ))
            for t in translate:
                if not (0.0 <= t <= 1.0): raise ValueError('translation values should be between 0 and 1')
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, 'scale', req_sizes=(2, ))
            for s in scale:
                if s <= 0: raise ValueError('scale values should be positive')
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name='shear', req_sizes=(2, 4))
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    def forward(self, image, mask):
        img_size = F._get_image_size(image)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        
        return F.affine(image, *ret, resample=self.resample, fillcolor=self.fillcolor), F.affine(mask, *ret, resample=self.resample, fillcolor=self.fillcolor)


class RandomGrayscale(torch.nn.Module):
    """
    Randomly convert image to grayscale with a probability of p (default 0.1).
    Only applied to image not mask.
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, image, mask):
        num_output_channels = F._get_image_num_channels(image)
        if torch.rand(1) < self.p:
            return F.rgb_to_grayscale(image, num_output_channels=num_output_channels), mask
        return image, mask


class GaussianBlur(torch.nn.Module):
    """
    Blurs image with randomly chosen Gaussian blur.
    Only applied to image not mask.
    """
    @staticmethod
    def get_params(sigma_min, sigma_max):
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()
    
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, 'Kernel size should be a tuple/list of two integers')
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0: raise ValueError('Kernel size value should be an odd and positive number.')

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError('If sigma is a single number, it must be positive.')
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]: raise ValueError('sigma values should be positive and of the form (min, max).')
        else: raise ValueError('sigma should be a single number or a list/tuple with length 2.')

        self.sigma = sigma

    def forward(self, image, mask):
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(image, self.kernel_size, [sigma, sigma]), mask


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence): raise TypeError('{} should be a sequence of length {}.'.format(name, msg))
    if len(x) not in req_sizes: raise ValueError('{} should be sequence of length {}.'.format(name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0: raise ValueError('If {} is a single number, it must be positive.'.format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]