from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


def random_crop(img, crop):
    return fn.crop(img, crop=crop,
                   crop_pos_x=fn.random.uniform(range=(0.2, 0.8)),
                   crop_pos_y=fn.random.uniform(range=(0.2, 0.8)),
                   out_of_bounds_policy='pad',
                   fill_values=0)


def center_crop(img, crop):
    return fn.crop(img, crop=crop, crop_pos_x=0.5, crop_pos_y=0.5, out_of_bounds_policy='pad', fill_values=0)


def adjust_size(img, size, policy='resize'):
    if policy == 'random_crop':
        img = random_crop(img, size)
    elif policy == 'center_crop':
        img = center_crop(img, size)
    elif policy == 'resize':
        img = fn.resize(img, size=size, interp_type=types.DALIInterpType.INTERP_CUBIC)
    else:
        assert False, f"Adjust size policy [{policy}] is not supported."

    return img


def augment(img):
    img = fn.flip(img, horizontal=fn.random.coin_flip(), vertical=0)
    img = fn.flip(img, horizontal=0, vertical=fn.random.coin_flip())

    return img


@pipeline_def
def cover_secret_pipline(cover_source, secret_source, cover_size, secret_size, augmentation=False, to_tensor=True,
                         adjust_size_policy_cover='center_crop', adjust_size_policy_secret='random_crop'):
    cover = fn.external_source(source=cover_source, num_outputs=1)
    cover = fn.decoders.image(cover, device='mixed')

    secret = fn.external_source(source=secret_source, num_outputs=1)
    secret = fn.decoders.image(secret, device='mixed')

    if augmentation:
        cover = augment(cover)
        secret = augment(secret)

    cover = adjust_size(cover, cover_size, policy=adjust_size_policy_cover)
    secret = adjust_size(secret, secret_size, policy=adjust_size_policy_secret)

    if to_tensor:
        cover = fn.transpose(cover, perm=[2, 0, 1]) / 255.0 - 0.5
        secret = fn.transpose(secret, perm=[2, 0, 1]) / 255.0 - 0.5

    return cover, secret
