import streamlit as st
import joblib
import pandas as pd

# Number of Recommendations to show
k = 5

model = joblib.load("model.obj")
st.title("Song Recommendation")

fields = {
    "title": st.selectbox("Song Title", model.data.title.unique()),
    "artist": st.selectbox("Song Artist", model.data.artist.unique()),
    # "genre": st.selectbox("Song Genre", model.data.genre.unique()),
}


if st.button("Predict"):
    data = pd.Series(fields)
    song_id = model.get_song_id(*data)

    if song_id is None:
        st.write("Song not found. Make sure the song exists")
    else:
        data = model.data.loc[song_id][:-1]

        recommendations = model.get_k_recommendations(data, k)
        recommendations = model.data.loc[recommendations, ["title", "artist"]]
        recommendations.index = range(1, recommendations.shape[0] + 1)
        st.table(recommendations)
