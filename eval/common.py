import cv2
import numpy as np
import progressbar

from utils.compute_overlap import compute_overlap
from utils.visualization import draw_detections, draw_annotations
from generators.utils import get_affine_transform, affine_transform

assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

def _compute_ap(recall, precision):

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size-1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i+1] - mrec[i])) * mpre[i+1]
    return ap


def _get_detections(generators, model, score_threshold=0.05, max_detections=100, visualize=False,
                    flip_test=False, keep_resolution=False):
    """
    Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]
    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.
    Returns:
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generators.num_classes()) if generators.has_label(i)] for j in range(generators.size())]

    for i in progressbar.progressbar(range(generators.size()), prefix='Running network:'):
        image = generators.load_image(i)
        src_image = image.copy()

        c = np.array([image.shape[1]/2., image.shape[0]/2.], dtype=np.float32)
        s = max(image.shape[0], image.shape[1]) * 1.0

        if not keep_resolution:
            tgt_w = generators.input_size
            tgt_h = generators.input_size
        else:
            tgt_w = image.shape[1] | 31 + 1
            tgt_h = image.shape[0] | 31 + 1

        image = generators.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
        
        if flip_test:
            flip_image = image[:, :, ::-1]
            inputs = np.stack([image, flip_image], axis=0)
        else:
            inputs = np.expand_dims(image, axis=0)
        
        detections = model.predict_on_batch(inputs)[0]
        scores = detections[:, 4]
        indices = np.where(scores > score_threshold)[0]

        detections = detections[indices]
        detections_copy = detections.copy()
        detections = detections.astype(np.float64)
        trans = get_affine_transform(c, s, (tgt_w//4, tgt_h//4), inv=1)

        for j in range(detections.shape[0]):
            detections[j, 0:2] = affine_transform(detections[j, 0:2], trans)
            detections[j, 2:4] = affine_transform(detections[j, 2:4], trans)
        
        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, src_image.shape[1])
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, src_image.shape[0])

        if visualize:
            # draw_annotations(src_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(src_image, detections[:5, :4], detections[:5, 4], detections[:5, 5].astype(np.int32),
                            label_to_name=generators.label_to_name,
                            score_threshold=score_threshold)

            # cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
            cv2.namedWindow('{}'.format(i), cv2.WINDOW_NORMAL)
            cv2.imshow('{}'.format(i), src_image)
            cv2.waitKey(0)
        
        # copy detections to all_detections
        for class_id in range(generators.num_classes()):
            all_detections[i][class_id] = detections[detections[:, -1] == class_id, :-1]
    
    return all_detections


def _get_annotations(generator):

    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.01,
        max_detections=100,
        visualize=False,
        flip_test=False,
        keep_resolution=False
):

    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     visualize=visualize, flip_test=flip_test, keep_resolution=keep_resolution)
    all_annotations = _get_annotations(generator)
    average_precisions = {}


     # process detections and annotations
    for label in range(generator.num_classes()):
        
        if not generator.has_label(label):
            continue
            
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []
            
            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
        
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions