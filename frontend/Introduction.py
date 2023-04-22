import streamlit as st
import pandas as pd
import joblib
import os

st.title("Song Recommendation")

st.write(os.path.abspath("."))

model = joblib.load("model.obj")
data: pd.DataFrame = model.data.drop("clusters", axis=1)

# Dataframe preview
st.write("## Data")
data_container = st.container()
with data_container:
    st.dataframe(model.data.drop("clusters", axis=1))


# info about the columns
data_info = st.container()
with data_info:
    st.write(
        """
        # Data Description
        The top songs BY YEAR in the world by spotify. 
        This dataset has several variables about the songs and is based on Billboard

        ## Columns
        1. **title**: Song's title
        2. **artist**: Song's artist
        3. **genre**: the genre of the song
        4. **year**: song's year in the Billboard
        5. **tempo**: beats-per-minute.
        6. **energy**: The energy of the song. 
                       The higher the value, the more energetic song.
        7. **danceability**: The higher the value, the easier it is to dance to this song.
        8. **loudness**: The higher the value, the louder the song
        9. **liveness**: The higher the value, the more likely the song is a live recording
        10. **valence**: The higher the value, the more positive mood for the song.
        11. **length**: Song duration
        12. **acousticness**: The higher the value the more acoustic the song is. 
        13. **speechiness**: The higher the value the more spoken words the song contains.
        14. **popularity**: Popularity of the song
        """
    )
