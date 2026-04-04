import os
import pandas as pd
import folium

# -----------------------------
# LOAD DATA (ROBUST PATH)
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

data_path = os.path.abspath(
    os.path.join(BASE_DIR, "../data/synthetic/risk_dataset.csv")
)

print(f"Loading dataset from: {data_path}")

df = pd.read_csv(data_path)

# -----------------------------
# CREATE MAP
# -----------------------------
# Center around Vijayawada
map_center = [16.5062, 80.6480]

m = folium.Map(location=map_center, zoom_start=13)

# -----------------------------
# COLOR FUNCTION
# -----------------------------
def get_color(risk):
    if risk == "High":
        return "red"
    elif risk == "Moderate":
        return "orange"
    else:
        return "green"

# -----------------------------
# ADD MARKERS
# -----------------------------
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=5,
        color=get_color(row['risk']),
        fill=True,
        fill_color=get_color(row['risk']),
        fill_opacity=0.7,
        popup=(
            f"Risk: {row['risk']}<br>"
            f"Hour: {row['hour']}<br>"
            f"Crime Score: {row['crime_score']}<br>"
            f"Crowd: {row['crowd_density']}"
        )
    ).add_to(m)

# -----------------------------
# SAVE MAP
# -----------------------------
output_path = os.path.join(BASE_DIR, "risk_map.html")
m.save(output_path)

print(f"\n✅ Map generated successfully at: {output_path}")