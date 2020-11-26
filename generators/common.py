import cv2
from tensorflow.keras.utils import Sequence
import math
import numpy as np
import random
import warnings
from imgaug import augmenters as iaa
import imgaug as ia

from generators.utils import get_affine_transform, affine_transform
from generators.utils import gaussian_radius, draw_gaussian, gaussian_radius_2

ia.seed(1)
class Generator(Sequence):

    def __init__(self, multi_scale=False, multi_image_sizes=(320, 352, 384, 416, 448, 480, 512, 544, 576, 608),
                    batch_size=1, group_method='ratio',
                    shuffle_groups=True, input_size=512, max_objects=100, train_data=True):
        
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.input_size = input_size
        self.output_size = self.input_size // 4
        self.max_objects = max_objects
        self.groups = None
        self.multi_scale = multi_scale
        self.multi_image_sizes = multi_image_sizes
        self.current_index = 0
        self.train_data = train_data
        if self.train_data:
            self.sequential = iaa.Sequential([
                iaa.Flipud(0.5), # vertically flip 20 %
                iaa.Fliplr(0.5), # horizontal flip 20%
                iaa.Multiply((0.5, 1.5), per_channel=0.5), # brightness
                iaa.GaussianBlur(sigma=(0, 3.0)),
                iaa.Affine(translate_px={"x": 15, "y": 15}, scale=(0.8, 0.95), rotate=(-30, 30)),
                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                #iaa.Invert(0.05, per_channel=True), # invert color channels
                #iaa.Add((-10, 10), per_channel=0.5), # Add a value of -10 to 10 to each pixel.
                #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),# Same as sharpen, but for an embossing effect
                ],random_order=True)
            

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            random.shuffle(self.groups)
    
    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)
        self.current_index = 0
    
    def size(self):
        """
        Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """
        Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """
        Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """
        Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """
        Load annotations for all images in group.
        """
        # load_annotations {'labels': np.array, 'annotations': np.array}
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        
        for annotations in annotations_group:
            assert (isinstance(annotations,
                               dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(
                type(annotations))
            assert (
                    'labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert (
                    'bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            #print(annotations)
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] <= 0) |
                (annotations['bboxes'][:, 3] <= 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]
            
            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
            if annotations['bboxes'].shape[0] == 0:
                warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(
                    group[index],
                    image.shape,
                ))
        return image_group, annotations_group

    def load_image_group(self, group):
        """
        Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    
    def preprocess_group_entry(self, image, annotations):
        """
        Preprocess image and its annotations.
        """

        # preprocess the image
        image, scale, offset_h, offset_w = self.preprocess_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= scale
        annotations['bboxes'][:, [0, 2]] += offset_w
        annotations['bboxes'][:, [1, 3]] += offset_h
        # print(annotations['bboxes'][:, [2, 3]] - annotations['bboxes'][:, [0, 1]])
        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """
        Preprocess each image and its annotations in its group.
        """
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index],
                                                                                       annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        """
        Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images

        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]
    
    def compute_inputs(self, image_group, annotations_group):
        """
        Compute inputs for the network using an image_group.
        """
        # construct an image batch object
        batch_images = np.zeros((len(image_group), self.input_size, self.input_size, 3), dtype=np.float32)

        batch_hms = np.zeros((len(image_group), self.output_size, self.output_size, self.num_classes()),
                             dtype=np.float32)
        batch_hms_2 = np.zeros((len(image_group), self.output_size, self.output_size, self.num_classes()),
                               dtype=np.float32)
        batch_whs = np.zeros((len(image_group), self.max_objects, 2), dtype=np.float32)
        batch_regs = np.zeros((len(image_group), self.max_objects, 2), dtype=np.float32)
        batch_reg_masks = np.zeros((len(image_group), self.max_objects), dtype=np.float32)
        batch_indices = np.zeros((len(image_group), self.max_objects), dtype=np.float32)
        
        # copy all images to the upper left part of the image batch object
        for b, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
            s = max(image.shape[0], image.shape[1]) * 1.0
            trans_input = get_affine_transform(c, s, self.input_size)

            # inputs
            image = self.preprocess_image(image, c, s, tgt_w=self.input_size, tgt_h=self.input_size)
            batch_images[b] = image

            # outputs
            bboxes = annotations['bboxes']
            
            assert bboxes.shape[0] != 0
            class_ids = annotations['labels']
            assert class_ids.shape[0] != 0

            trans_output = get_affine_transform(c, s, self.output_size)
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i].copy()
                cls_id = class_ids[i]
                # (x1, y1)
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                # (x2, y2)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_size - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    
                    radius_h, radius_w = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius_h = max(0, int(radius_h))
                    radius_w = max(0, int(radius_w))

                    radius = gaussian_radius_2((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, (radius_h, radius_w))
                    draw_gaussian(batch_hms_2[b, :, :, cls_id], ct_int, radius)
                    batch_whs[b, i] = 1. * w, 1. * h
                    batch_indices[b, i] = ct_int[1] * self.output_size + ct_int[0]
                    batch_regs[b, i] = ct - ct_int
                    batch_reg_masks[b, i] = 1

        return [batch_images, batch_hms_2, batch_whs, batch_regs, batch_reg_masks, batch_indices]

    def compute_targets(self, image_group, annotations_group):
        """
        Compute target outputs for the network using images and their annotations.
        """
        return np.zeros((len(image_group),))

    def data_arguments(self, image_group, annotations_group):
        images = image_group.copy()
        self.sequential = self.sequential.to_deterministic()
        for index, (image, annotation) in enumerate(zip(images, annotations_group)):
            if np.random.uniform() > 0.5:
                continue

            bbs = ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=annotation['bboxes'][0, 0], y1=annotation['bboxes'][0, 1], x2=annotation['bboxes'][0, 2], y2=annotation['bboxes'][0, 3]) 
            ], shape=image.shape)
            image_aug = self.sequential.augment_images([image])[0]
            bbs_augs = self.sequential.augment_bounding_boxes([bbs])[0]

            for i, bbs_aug in enumerate(bbs_augs.bounding_boxes):
                annotation['bboxes'][i, 0] = bbs_aug.x1
                annotation['bboxes'][i, 1] = bbs_aug.y1
                annotation['bboxes'][i, 2] = bbs_aug.x2
                annotation['bboxes'][i, 3] = bbs_aug.y2
            
            images[index] = image_aug
            
        return images, annotations_group

    def compute_inputs_targets(self, group):
        """
        Compute inputs and target outputs for the network.
        """

        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        
        annotations_group = self.load_annotations_group(group)
        
        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        #print(annotations_group)
        if self.train_data:
            image_group, annotations_group = self.data_arguments(image_group, annotations_group)
       
        if len(image_group) == 0:
            return None, None

        # compute network inputs
        inputs = self.compute_inputs(image_group, annotations_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[self.current_index]
        if self.multi_scale:
            if self.current_index % 10 == 0:
                random_size_index = np.random.randint(0, len(self.multi_image_sizes))
                self.image_size = self.multi_image_sizes[random_size_index]
        inputs, targets = self.compute_inputs_targets(group)
        while inputs is None:
            current_index = self.current_index + 1
            if current_index >= len(self.groups):
                current_index = current_index % (len(self.groups))
            self.current_index = current_index
            group = self.groups[self.current_index]
            inputs, targets = self.compute_inputs_targets(group)
        current_index = self.current_index + 1
        if current_index >= len(self.groups):
            current_index = current_index % (len(self.groups))
        self.current_index = current_index
        return inputs, targets

    def preprocess_image(self, image, c, s, tgt_w, tgt_h):
        trans_input = get_affine_transform(c, s, (tgt_w, tgt_h))
        image = cv2.warpAffine(image, trans_input, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
        image = image.astype(np.float32)

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image

    def get_transformed_group(self, group):
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)
        return image_group, annotations_group

    def get_cropped_and_rotated_group(self, group):
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_crop_group(image_group, annotations_group)
        image_group, annotations_group = self.random_rotate_group(image_group, annotations_group)
        return image_group, annotations_group

