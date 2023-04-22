from sklearn.pipeline import Pipeline
from pandas import DataFrame, Series
from numpy import ndarray

from typing import Optional, List


class SongRecommender:
    def __init__(self, model: Pipeline):
        self.model: Pipeline = model
        self.data: Optional[DataFrame] = None

    def fit(self, data: DataFrame) -> None:
        self.data = data
        self.model.fit(data.values)
        self.data["clusters"] = self.model.predict(data.values)

    def get_song_id(self, title: str, artist: str) -> int | None:
        if self.data is None:
            return -1

        songs = self.data[
            (self.data.title == title) & (self.data.artist == artist)
        ].index

        if len(songs) < 1:
            return None

        return songs[0]

    def get_cluster(self, user_data: ndarray) -> int:
        return self.model.predict([user_data])[0]

    def get_k_recommendations(self, user_data: Series, k: int) -> List[int | str]:
        if self.data is None:
            return []

        cluster = self.get_cluster(user_data.values)
        similar_songs = self.data[self.data.clusters == cluster]

        return similar_songs.sample(k, replace=True).index.tolist()
