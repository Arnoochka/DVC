from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
list(data.target_names)
data.frame.to_csv('data/iris.csv', index=False)
