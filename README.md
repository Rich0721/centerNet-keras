# centerNet-keras

This is an implementation of [CenterNet](https://arxiv.org/abs/1904.07850) for object detection on keras(2.4.3) and Tensorflow(2.2.0). The project reference [keras-CenterNet](https://github.com/xuannianz/keras-CenterNet) modified the code to tf2.x version.

## Todo
- [x]  Be able to train own datasets.
- [x] Inference.
- [ ] Evaluate model.
- [ ] COCO datasets.
- [ ] More backbone.
- [ ] So on.


## Install library
```
pip install requirments.txt
```

## Train
If you want to download compute_overlap.pyx on other python version you can go to [keras-retinanet](https://github.com/fizyr/keras-retinanet).
If you want to training self datasets, you can modify config.py CLASSES, IMAGE_DIR, ANNOTATION_DIR, TRAIN_TEXT, VAL_TEXT and TEST_TEXT.
```
python train.py
```

## Test
```
python inference.py
```
