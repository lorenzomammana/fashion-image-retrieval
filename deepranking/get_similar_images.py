import fashion_utils
import files
from pathlib import Path
from fashion_similarity import FashionSimilarity

if __name__ == '__main__':
    
    # TODO get from cmd or something else
    query_img = Path('/home/ubuntu/fashion-dataset/small_classes/Belts/20423.jpg')
    n=5

    similarity = FashionSimilarity()
    img_class, similar_images = similarity.get_similar_images(query_img, n)

    print(similar_images)
