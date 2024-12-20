import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import os
import yaml


def prepare_data(split: float, seed: int):
    iris = load_iris()

    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    train_df, test_df = train_test_split(df, test_size=split, random_state=seed)
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    train_df.to_csv('data/prepared/train.csv', index=False)
    test_df.to_csv('data/prepared/test.csv', index=False)
    
def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    split = params["split"]
    seed = params["seed"]
    prepare_data(split, seed)

if __name__ == '__main__':
    main()