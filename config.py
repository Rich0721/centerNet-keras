from collections import OrderedDict


class Config(object):

    #########Train related############
    EPOCHS = 100
    STEP_PER_EPOCHS = 1000
    BATCH_SIZE = 8
    IMAGE_SIZE = 512
    MULTI_SCALE = False
    COMMON_ARGS = {
        'batch_size': BATCH_SIZE,
        'input_size': IMAGE_SIZE,
    }

    CLASSES = {
         'airwaves-mint':0, 'eclipse-lemon':1, 'eclipse-mint':2, 'eclipse-mint-fudge':3,
        'extra-lemon':4, 'hallsxs-buleberry':5, 'hallsxs-lemon':6, 'meiji-blackchocolate':7,
        'meiji-milkchocolate':8, 'rocher':9}

    ########Test related##############
    FLIP_TEST = True
    NMS = True
    KEEP_RESOLUTION = False
    SCORE_THRESHOLD = 0.1

    #########Storage related##########
    STORE_FOLDER = "./centernet"
    STORE_FILE = "centernerNet"

    #########Image folder#############
    IMAGE_DIR = "../datasets/voc/JPEGImages"
    ANNOTATION_DIR = "../datasets/voc/Annotations"
    TRAIN_TEXT = "../datasets/voc/train.txt"
    VAL_TEXT = "../datasets/voc/val.txt"
    TEST_TEXT = "../datasets/voc"

