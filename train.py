from datetime import date, timedelta
import keras
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN, EarlyStopping, LearningRateScheduler, CSVLogger
from keras.optimizers import Adam, SGD
import os
import sys
import tensorflow as tf

from augmentor.color import VisualEffect
from augmentor.msic import MiscEffect
from models.centernet import centernet
from eval.pascal import EvaluateVoc
from config import Config

from generators.pascal import PascalVocGenerator
######################################################

log_dir = "./centernet"
h5file_name = "centernerNet"
epochs = 100
step_per_epochs = 1000
batch_size = 8
img_size = 512
image_dir = "../datasets/voc/JPEGImages"
annotation_dir = "../datasets/voc/Annotations"
train_text = "../datasets/voc/train.txt"
val_text = "../datasets/voc/val.txt"
multi_scale = False

common_args = {
        'batch_size': batch_size,
        'input_size': img_size,
    }

classes = {
         'airwaves-mint':0, 'eclipse-lemon':1, 'eclipse-mint':2, 'eclipse-mint-fudge':3,
        'extra-lemon':4, 'hallsxs-buleberry':5, 'hallsxs-lemon':6, 'meiji-blackchocolate':7,
        'meiji-milkchocolate':8, 'rocher':9}

######################################################

cfg = Config()

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    config = tf.ConfigProto()
    tf.config.gpu.set_per_process_memory_growth(True)

    sess = tf.Session(config=config)
    return sess


def lr_schedule(epoch):

    if epoch < 50:
        return 0.001
    elif epoch < 80:
        return 0.0001
    else:
    
        return 0.00001


def create_callbacks():

    tensorbord_callback = []
    tensorbord_callback = TensorBoard(
        log_dir=cfg.STORE_FOLDER, histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )

    checkpoint = ModelCheckpoint(os.path.join(cfg.STORE_FOLDER, cfg.STORE_FILE + "-{epoch:02d}.h5"),
                                verbose=1, save_best_only=True, mode=1)  

    terminateOnNaN = TerminateOnNaN()

    csv_logger = CSVLogger(filename=os.path.join(cfg.STORE_FOLDER, cfg.STORE_FILE + '_training_log.csv'),
                       separator=',',
                       append=True)

    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule)
    
    callbacks = [tensorbord_callback, checkpoint, terminateOnNaN, csv_logger, learning_rate_scheduler]
    
    return callbacks


def create_generators(train=False):

    if train:
        misc_effect = MiscEffect(border_value=0)
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None
    train_generator = PascalVocGenerator(
        image_dir=cfg.IMAGE_DIR, annotation_dir=cfg.ANNOTATION_DIR,
        text_file=cfg.TRAIN_TEXT, skip_difficult=True, classes=cfg.CLASSES,
        multi_scale=cfg.MULTI_SCALE, misc_effect=misc_effect,
        visual_effect=visual_effect, **cfg.COMMON_ARGS
            
    )

    validation_generator = PascalVocGenerator(
        image_dir=cfg.IMAGE_DIR, annotation_dir=cfg.ANNOTATION_DIR,
        text_file=cfg.VAL_TEXT, skip_difficult=True, classes=cfg.CLASSES,shuffle_groups=False
        ,**cfg.COMMON_ARGS
    )


    return train_generator, validation_generator

def main():
    #K.set_session(get_session())
    train_generator, validation_generator = create_generators()

    num_classes = train_generator.num_classes()
    model = centernet(num_classes=num_classes, input_size=img_size)

    model.compile(optimizer=Adam(lr=1e-3), loss={'centernet_loss': lambda y_true, y_pred: y_pred})
    #model.compile(optimizer=SGD(lr=1e-3, momentum=0.9, nesterov=True, decay=1e-5),
    #               loss={'centernet_loss': lambda y_true, y_pred: y_pred})
    
    callbacks = create_callbacks()


    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=cfg.STEP_PER_EPOCHS,
        initial_epoch=0,
        epochs=cfg.EPOCHS,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator
    )

if __name__ == "__main__":
    main()