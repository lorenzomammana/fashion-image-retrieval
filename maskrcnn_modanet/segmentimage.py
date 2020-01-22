import keras

# import keras_retinanet
from keras_maskrcnn import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

from skimage.measure import regionprops

path = '/home/ubuntu/fashion-dataset/'


def loadModel():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'bag', 1: 'belt', 2: 'boots', 3: 'footwear', 4: 'outer', 5: 'dress', 6: 'sunglasses',
                       7: 'pants', 8: 'top', 9: 'shorts', 10: 'skirt', 11: 'headwear', 12: 'scarf/tie'}

    model_path = path + 'resnet50_modanet.h5'

    model = models.load_model(model_path, backbone_name='resnet50')

    return model, labels_to_names


def get_session():
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def draw_mask_only(image, box, mask, label=None, color=None, binarize_threshold=0.5):
    """ Draws a mask in a given box and makes everything else black.
    Args
        image              : Three dimensional image to draw on.
        box                : Vector of at least 4 values (x1, y1, x2, y2) representing a box in the image.
        mask               : A 2D float mask which will be reshaped to the size of the box, binarized and drawn over the image.
        color              : Color to draw the mask with. If the box has 5 values, the last value is assumed to be the label and used to construct a default color.
        binarize_threshold : Threshold used for binarizing the mask.
    """

    from keras_retinanet.utils.colors import label_color

    # import miscellaneous modules
    import cv2
    import numpy as np

    if label is not None:
        color = label_color(label)
    if color is None:
        color = (255, 255, 255)

    # resize to fit the box
    mask = mask.astype(np.float32)
    mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))

    # binarize the mask
    mask = (mask > binarize_threshold).astype(np.uint8)

    # draw the mask in the image
    mask_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    mask_image[box[1]:box[3], box[0]:box[2]] = mask
    mask = mask_image

    # compute a nice border around the mask
    border = mask - cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

    # apply color to the mask and border
    mask = (np.stack([mask] * 3, axis=2) * color).astype(np.uint8)
    border = (np.stack([border] * 3, axis=2) * (255, 255, 255)).astype(np.uint8)
    # this is how you look into the mask
    # for i in mask:
    # 	for j in i:
    # 		b = False
    # 		for k in i:
    # 			for l in k:
    # 				if l != 0:
    # 					b = True
    # 				if b:
    # 					break
    # 			if b:
    # 				break
    # 		if b:
    # 			print (j)

    # draw the mask
    indices = np.where(mask != color)
    image[indices[0], indices[1], :] = 0 * image[indices[0], indices[1], :]

    return mask


def main(filename):
    model, labels_to_names = loadModel()

    image = read_image_bgr(filename)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    outputs = model.predict_on_batch(np.expand_dims(image, axis=0))

    boxes = outputs[-4][0]
    scores = outputs[-3][0]
    labels = outputs[-2][0]
    masks = outputs[-1][0]

    # correct for image scale
    boxes /= scale

    segment_id = 0

    threshold_score = 0.3

    for box, score, label, mask in zip(boxes, scores, labels, masks):
        if score < threshold_score:
            break

        drawclone = np.copy(draw)

        b = box.astype(int)
        color = label_color(label)
        # draw_box(drawclone, b, color=color)

        mask = mask[:, :, label]
        # draw_mask_only(drawclone, b, mask, color=label_color(label))
        mask_binary = draw_mask_only(drawclone, b, mask, color=None)

        bbox = regionprops(mask_binary[:, :, 0])
        bbox = bbox[0].bbox

        # draw_caption(drawclone, b, caption)
        plt.figure()
        plt.axis('off')
        plt.imshow(drawclone[bbox[0]:bbox[2], bbox[1]:bbox[3]])

        segment_path = '/tmp/segment_' + str(segment_id) + '.jpg'
        save_path_segment = segment_path
        plt.savefig(save_path_segment)
        plt.close()

        segment_id += 1

    return segment_id

