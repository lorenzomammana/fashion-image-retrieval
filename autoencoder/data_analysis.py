import pandas as pd
from pathlib import Path
from files import ROOT

if __name__ == '__main__':
    
    data = pd.read_csv(ROOT / 'styles.csv')

    for c in data.columns:
        print('---- {} ---'.format(c))
        print(data[c].value_counts())


