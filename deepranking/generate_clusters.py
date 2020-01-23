import fashion_utils
import files
from fashion_ranking_model import FashionRankingModel

if __name__ == '__main__':
    
    model = FashionRankingModel().compile(weights=files.deepranking_weights_path)
