import joblib

def reduce_embeddings(embeddings):
    pca = joblib.load("models/artifacts/pca.joblib")
    return pca.transform(embeddings)
