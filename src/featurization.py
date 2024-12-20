from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
import pickle
import os
import yaml

def transform_data(input_dir: str, output_dir: str, train: bool) -> None:
    df = pd.read_csv(input_dir)
    X = df.drop(columns=['target'])
    y = df['target']
    if train:
        encoder = LabelEncoder()
        scaler = StandardScaler()
        
        y = encoder.fit_transform(y)
        X = scaler.fit_transform(X)
        
        with open('data/features/encoders_scaler.pkl', 'wb') as f:
            pickle.dump({'label_encoder': encoder, 'scaler': scaler}, f)
        
    else:
        coders = None
        with open('data/features/encoders_scaler.pkl', 'rb') as f:
            coders = pickle.load(f)

        encoder: LabelEncoder = coders['label_encoder']
        scaler: StandardScaler = coders['scaler']
        
        y = encoder.transform(y)
        X = scaler.transform(X)
        
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    with open(output_dir, 'wb') as f:
        pickle.dump(dataloader, f)
        
def main():
    os.makedirs(os.path.join("data", "features"), exist_ok=True)
    
    transform_data("data/prepared/train.csv", "data/features/train.pkl", True)
    transform_data("data/prepared/test.csv", "data/features/test.pkl", False)
    
if __name__ == "__main__":
    main()
        
    
        
    