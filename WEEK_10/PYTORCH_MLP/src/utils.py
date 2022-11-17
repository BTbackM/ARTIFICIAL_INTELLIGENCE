from os import path

import pandas as pd

# NOTE: Global variables

ABS_PATH = path.dirname(path.realpath(__file__))
DATA_PATH = path.join(ABS_PATH, '../data')
IMG_PATH = path.join(ABS_PATH, '../img')

# NOTE: Utility functions

def parse_csv(file):
    try:
        df = pd.read_csv(path.join(DATA_PATH, file), header=None)
        col, tmp_df = None, None
        for column in df:
            if df[column].dtypes == object:
                col = df[column]
                df[column] = df[column].astype('category').cat.codes
                tmp_df = pd.DataFrame({'Column': col, 'Code': df[column]})
                tmp_df = tmp_df.groupby(['Column', 'Code']).count()
                tmp_df = pd.DataFrame(list(tmp_df.index.values), columns=['Column', 'Code'])
                print(tmp_df)
        df.to_csv(path.join(DATA_PATH, 'parsed.csv'), header=None, index=False)
    except:
        print('Could not read the file')