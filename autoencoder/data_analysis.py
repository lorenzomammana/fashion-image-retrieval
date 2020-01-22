import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import files

if __name__ == '__main__':
    
    data = pd.read_csv(files.ROOT / 'styles.csv', usecols=range(10))

    classes = data['articleType'].value_counts(normalize=True)
    classes = classes[classes >= 0.01]

    for i in range(classes.shape[0]):
        print('{}\t{}'.format(classes.index[i], classes.values[i]))

    print('{} classes'.format(classes.shape[0]))

    # plt.pie(classes.values)
    # plt.savefig(files.classes_distribution_pdf_path)


    # Modanet label correspondance (Modanet class -> our dataset class)
    # 1: bag -> Handbags
    # 2: belt -> Belts
    # 3: boots -> /
    # 4: footwear -> Casual Shoes, Sports Shoes, Heels, Sandals, Formal Shoes, Flats
    # 5: outer -> /
    # 6: dress -> Dresses
    # 7: sunglasses -> Sunglasses
    # 8: pants -> Jeans, Trousers
    # 9: top -> Tshirts, Shirts, Tops
    # 10: shorts -> Shorts
    # 11: skirt -> /
    # 12: headwear -> /
    # 13: scarf & tie -> /
