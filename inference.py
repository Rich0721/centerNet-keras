from generators.pascal import PascalVocGenerator
from models.centernet import centernet
import cv2
import numpy as np
from generators.utils import affine_transform, get_affine_transform
import os.path as osp
from config import Config


############################################################
cfg = Config()
model_path = "./centernet/centernerNet_imgaug-83.h5"
############################################################


def main():

    generator = PascalVocGenerator(cfg.IMAGE_DIR, cfg.ANNOTATION_DIR, cfg.TEST_TEXT, classes=cfg.CLASSES, skip_difficult=True, train_data=False)
    
    num_classes = generator.num_classes()
    classes = list(generator.classes.keys())
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
    model = centernet(num_classes, score_threshold=cfg.SCORE_THRESHOLD, nms=cfg.NMS, flip_test=cfg.FLIP_TEST, training=False)
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    for i in range(10):
        image = generator.load_image(i)
        #cv2.imwrite("./results/{}_o.jpg".format(i), image)
        src_image = image.copy()

        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        s = max(image.shape[0], image.shape[1]) * 1.0

        tgt_w = generator.input_size
        tgt_h = generator.input_size
        image = generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)

        if cfg.FLIP_TEST:
            flipped_image = image[:, ::-1]
            inputs = np.stack([image, flipped_image], axis=0)
        else:
            inputs = np.expand_dims(image, axis=0)

        detections = model.predict_on_batch(inputs)[0]

        scores = detections[:, 4]

        indices = np.where(scores > cfg.SCORE_THRESHOLD)[0]

        detections = detections[indices]
        detections_copy = detections.copy()
        detections = detections.astype(np.float64)
        trans = get_affine_transform(c, s, (tgt_w // 4, tgt_h // 4), inv=1)

        for j in range(detections.shape[0]):
            detections[j, 0:2] = affine_transform(detections[j, 0:2], trans)
            detections[j, 2:4] = affine_transform(detections[j, 2:4], trans)
        
        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, src_image.shape[1])
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, src_image.shape[0])
        
        for detection in detections:
            xmin = int(round(detection[0]))
            ymin = int(round(detection[1]))
            xmax = int(round(detection[2]))
            ymax = int(round(detection[3]))
            score = '{:.4f}'.format(detection[4])
            class_id = int(detection[5])
            color = colors[class_id]
            class_name = classes[class_id]
            label = '-'.join([class_name, score])
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        #cv2.imwrite("./results/{}_r.jpg".format(i), src_image)
        cv2.imshow('image', src_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
