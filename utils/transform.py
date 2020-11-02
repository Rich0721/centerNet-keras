import numpy as np

DEFAULT_PRNG = np.random

def colvec(*args):
    return np.array([args]).T


def transfrom_coordinate(transform, coordinate):

    x1, y1, x2, y2 = coordinate

    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y1, y1],
        [1, 1, 1, 1]
    ])

    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


def _random_vector(mins, maxs, prng=DEFAULT_PRNG):

    mins = np.array(mins)
    maxs = np.array(maxs)
    assert mins.shape == maxs.shape
    assert len(mins.shape) == 1
    return prng.uniform(mins, maxs)


def roration(angle):

    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_rotation(mins, maxs, prng=DEFAULT_PRNG):

    return roration(prng.uniform(mins, maxs))


def translation(trans):

    return np.array([
        [1, 0, trans[0]],
        [0, 1, trans[1]],
        [0, 0, 1]
    ])

def random_translation(mins, maxs, prng=DEFAULT_PRNG):

    return translation(prng.uniform(mins, maxs))


def shear(angle):
    return np.array([
        [1, -np.sin(angle), 0],
        [0, np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_shear(mins, maxs, prng=DEFAULT_PRNG):
    return shear(prng.uniform(mins, maxs))


def scaling(factor):
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def random_scaling(mins, maxs, prng=DEFAULT_PRNG):
    return scaling(prng.uniform(mins, maxs))


def random_flip(flip_x_chance, flip_y_chance, prng=DEFAULT_PRNG):
    flip_x = prng.uniform(0, 1) < flip_x_chance
    flip_y = prng.uniform(0, 1) < flip_y_chance
    return scaling((1-2*flip_x, 1-2*flip_y))


def change_transform_origin(transform, center):
    center = np.array(center)
    return np.linalg.multi_dot([translation(center), transform, translation(-center)])


def random_transform(
    min_rotation=0,
    max_rotation=0,
    min_translation=(0, 0),
    max_translation=(0, 0),
    min_shear=0,
    max_shear=0,
    min_scaling=(1, 1),
    max_scaling=(1, 1),
    flip_x_chance=0,
    flip_y_chance=0,
    prng=DEFAULT_PRNG
):

    return np.linalg.multi_dot([
        random_rotation(min_rotation, max_rotation, prng),
        random_translation(min_translation, max_translation, prng),
        random_shear(min_shear, max_shear, prng),
        random_scaling(min_scaling, max_scaling, prng),
        random_flip(flip_x_chance, flip_y_chance, prng)
    ])


def random_transform_generator(prng=None, **kwargs):

    if prng is None:
        # RandomState automatically seeds using the best available method.
        prng = np.random.RandomState()

    while True:
        yield random_transform(prng=prng, **kwargs)