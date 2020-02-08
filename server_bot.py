from bot.Updater import Updater
import os
from deepranking import compute_single_image, FashionSimilarity, compute_single_image_lsh, FashionSimilarityLSH
from maskrcnn_modanet import load_mask_rcnn, segment_image

print('Loading models...')
mask_rcnn_model, labels_to_names = load_mask_rcnn()
similarity_model = FashionSimilarity()
# similarity_model = FashionSimilarityLSH()
print('Models loaded.')


def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def imageHandler(bot, message, chat_id, local_filename):
    print(local_filename)
    # send message to user
    bot.sendMessage(chat_id, "Hi, please wait until the image is ready")

    num_images = segment_image(local_filename, mask_rcnn_model, labels_to_names)

    bot.sendMessage(chat_id, "I've found " + str(num_images) + " potential clothes")

    for i in range(num_images):
        compute_single_image(i, similarity_model)
        bot.sendImage(chat_id, '/tmp/out_' + str(i) + '.jpg', "")
        # compute_single_image_lsh(i, similarity_model)
        # bot.sendImage(chat_id, '/tmp/out_lsh_' + str(i) + '.jpg', "")


if __name__ == "__main__":
    bot_id = '995419223:AAFdJPfz318tuB7FGZkAvNTtM0QP0Y4zHQs'
    updater = Updater(bot_id)
    updater.setPhotoHandler(imageHandler)
    updater.start()
