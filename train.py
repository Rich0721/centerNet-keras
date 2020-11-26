from datetime import date, timedelta
import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN, EarlyStopping, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam, SGD
import os
import sys
import tensorflow as tf

from models.centernet import centernet
from eval.pascal import EvaluateVoc
from config import Config

from generators.pascal import PascalVocGenerator


cfg = Config()

def makedirs(path):
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
    '''
    tensorbord_callback = TensorBoard(
        log_dir=cfg.STORE_FOLDER, histogram_freq=0,
        batch_size=cfg.BATCH_SIZE,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )
    '''
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


    train_generator = PascalVocGenerator(
        image_dir=cfg.IMAGE_DIR, annotation_dir=cfg.ANNOTATION_DIR,
        text_file=cfg.TRAIN_TEXT, skip_difficult=True, classes=cfg.CLASSES,
        multi_scale=cfg.MULTI_SCALE, **cfg.COMMON_ARGS      
    )

    validation_generator = PascalVocGenerator(
        image_dir=cfg.IMAGE_DIR, annotation_dir=cfg.ANNOTATION_DIR,
        text_file=cfg.VAL_TEXT, skip_difficult=True, classes=cfg.CLASSES,shuffle_groups=False
        ,**cfg.COMMON_ARGS, train_data=False
    )


    return train_generator, validation_generator

def main():
    #K.set_session(get_session())
    train_generator, validation_generator = create_generators()

    num_classes = train_generator.num_classes()
    model = centernet(num_classes=num_classes, input_size=cfg.IMAGE_SIZE)

    model.compile(optimizer=Adam(lr=1e-3), loss={'centernet_loss': lambda y_true, y_pred: y_pred})
    #model.compile(optimizer=SGD(lr=1e-3, momentum=0.9, nesterov=True, decay=1e-5),
    #               loss={'centernet_loss': lambda y_true, y_pred: y_pred})
    
    callbacks = create_callbacks()


    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        initial_epoch=0,
        epochs=cfg.EPOCHS,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

if __name__ == "__main__":
    main()