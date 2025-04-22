import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v3.json'


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

# Przygotowanie danych dla wykresu radarowego
def prepare_radar_data(df):
    # Przygotowanie danych dla kategorii wiek
    age_counts = df['age'].value_counts(normalize=True).sort_index()
    
    # Przygotowanie danych dla kategorii wykształcenie
    edu_counts = df['edu_level'].value_counts(normalize=True)
    
    # Przygotowanie danych dla kategorii ulubione zwierzęta
    animals_counts = df['fav_animals'].value_counts(normalize=True)
    
    # Przygotowanie danych dla kategorii ulubione miejsce
    place_counts = df['fav_place'].value_counts(normalize=True)
    
    # Przygotowanie danych dla kategorii płeć
    gender_counts = df['gender'].value_counts(normalize=True)
    
    # Łączenie wszystkich danych
    categories = []
    values = []
    
    for category, value in age_counts.items():
        categories.append(f"Wiek: {category}")
        values.append(value)
    
    for category, value in edu_counts.items():
        categories.append(f"Edu: {category}")
        values.append(value)
    
    for category, value in animals_counts.items():
        categories.append(f"Zwierzęta: {category}")
        values.append(value)
    
    for category, value in place_counts.items():
        categories.append(f"Miejsce: {category}")
        values.append(value)
    
    for category, value in gender_counts.items():
        categories.append(f"Płeć: {category}")
        values.append(value)
    
    return categories, values

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

# Tworzenie wykresu radarowego pokazującego profil grupy
st.header("Profil grupy ")

# Generowanie danych dla wykresu radarowego
categories, values = prepare_radar_data(same_cluster_df)

# Dodanie ostatniego punktu aby zamknąć wykres (połączyć koniec z początkiem)
categories.append(categories[0])
values.append(values[0])

# Tworzenie wykresu radarowego za pomocą plotly
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name=f'Grupa {predicted_cluster_data["name"]}'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    title=f"Charakterystyka grupy {predicted_cluster_data['name']}",
    showlegend=True
)

st.plotly_chart(fig)
st.header("Osoby z grupy")
# Wyświetlenie przykładowych osób z tej samej grupy
# st.dataframe(same_cluster_df.sample(5), hide_index=True)

fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)





