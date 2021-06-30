
from fastapi import FastAPI
from joblib import load
import pandas as pd

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

@app.get("/predict")
def predict(acousticness, danceability, duration_ms, energy, explicit, id, instrumentalness, key, liveness, loudness, mode, name, release_date, speechiness, tempo, valence, artist):
    model = load('model.joblib')
    X = pd.DataFrame({"acousticness":[acousticness], "danceability":[danceability], "duration_ms":[duration_ms], "energy":[energy], "explicit":[explicit], "id":[id], "instrumentalness":[instrumentalness], "key":[key], "liveness":[liveness], "loudness":[loudness], "mode":[mode], "name":[name], "release_date":[release_date], "speechiness":[speechiness], "tempo":[tempo], "valence":[valence], "artist":[artist]})
    pop_predicted= model.predict(X)[0]
    return {
        'artist': artist,
        'name': name,
        'popularity': pop_predicted,
        }