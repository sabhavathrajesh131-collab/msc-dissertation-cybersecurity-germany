"""
Data extraction from the GDELT Document API
for cybersecurity incidents in the German financial sector.
"""

import requests
import pandas as pd

def fetch_gdelt_data():
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "Germany financial cyber attack",
        "mode": "ArtList",
        "format": "JSON",
        "maxrecords": 250,
        "sourcelang": "English"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data.get("articles", []))

if __name__ == "__main__":
    df = fetch_gdelt_data()
    print(df.head())
