from song_recommender import SongRecommender
import joblib

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.cluster import KMeans

if __name__ == "__main__":
    df = pd.read_csv("../data/top10s_final.csv", index_col=0)
    sorted_years = df.year.sort_values().unique()
    preprocessor = ColumnTransformer(
        [
            ("scaler", StandardScaler(), list(range(4, 14))),
            ("ohe", OneHotEncoder(sparse_output=False), [2]),
            ("ordinal", OrdinalEncoder(categories=[sorted_years]), [3]),
        ],
        remainder="drop",
    )

    # Best model from grid search
    model = KMeans(init="random", max_iter=900, n_clusters=10, n_init="auto")

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print(df.head())
    recommender = SongRecommender(pipeline)
    recommender.fit(df)

    # save model
    joblib.dump(recommender, "model.obj")
    print("Model saved at: model.obj")
