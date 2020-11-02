from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Conv2DTranspose, BatchNormalization, Conv2D, Lambda, MaxPooling2D, ZeroPadding2D
from keras.activations import relu
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

from losses import loss
from models.common import decode


def centernet(num_classes, input_size=512, max_objects=100, score_threshold=0.1,
              nms=True, flip_test=False, training=True, l2_norm=5e-4):
    
    output_size = input_size // 4
    image_input = Input(shape=(input_size, input_size, 3))
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    resnet = ResNet50(include_top=False, input_tensor=image_input)
    x = resnet.outputs[-1]

    num_filters = 256
    for i in range(3):
        num_filters = num_filters // pow(2, i)
        x = Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(l2_norm))(x)
        x = BatchNormalization()(x)
        x = relu(x)

    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(x)
    y1 = BatchNormalization()(y1)
    y1 = relu(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(x)
    y2 = BatchNormalization()(y2)
    y2 = relu(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(x)
    y3 = BatchNormalization()(y3)
    y3 = relu(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(y3)

    loss_ = Lambda(loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    
    if training:
        model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
    else:
        # detections = decode(y1, y2, y3)
        detections = Lambda(lambda x: decode(*x,
                                            max_objects=max_objects,
                                            score_threshold=score_threshold,
                                            nms=nms,
                                            flip_test=flip_test,
                                            num_classes=num_classes))([y1, y2, y3])
        model = Model(inputs=image_input, outputs=detections)
    return model