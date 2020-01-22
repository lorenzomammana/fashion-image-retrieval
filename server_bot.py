from bot.Updater import Updater
import os, sys, platform, subprocess
from maskrcnn_modanet.segmentimage import main


def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def imageHandler(bot, message, chat_id, local_filename):
    print(local_filename)
    # send message to user
    bot.sendMessage(chat_id, "Hi, please wait until the image is ready")

    num_images = main(local_filename)

    bot.sendMessage(chat_id, "I've found " + str(num_images) + " potential clothes")

    for i in range(num_images + 1):
        bot.sendImage(chat_id, '/tmp/segment_' + str(i) + '.jpg', "")


if __name__ == "__main__":
    bot_id = '995419223:AAFdJPfz318tuB7FGZkAvNTtM0QP0Y4zHQs'
    updater = Updater(bot_id)
    updater.setPhotoHandler(imageHandler)
    updater.start()
