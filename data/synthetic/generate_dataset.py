import pandas as pd
import random
rows = []
for _ in range(1000):
    lat = round(random.uniform(16.50, 16.55), 4)
    long = round(random.uniform(80.64, 80.70), 4)
    hour = random.randint(0, 23)
    crime_score = round(random.uniform(0, 1), 2)
    crowd_density = round(random.uniform(0, 1), 2)

    if (hour >= 20 or hour <= 5) and crime_score > 0.6 and crowd_density < 0.4:
        risk = "High"
    elif crime_score > 0.4:
        risk = "Moderate"
    else:
        risk = "Safe"
    rows.append([lat, long, hour, crime_score, crowd_density, risk])
df = pd.DataFrame(rows, columns=[
    "lat", "long", "hour", "crime_score", "crowd_density", "risk"
])
df.to_csv("data/synthetic/risk_dataset.csv", index=False)

print("Dataset generated successfully!")