import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
st.write("Testing World Bank API connectivity...")
try:
    r = requests.get("http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv")
    st.write("Status code:", r.status_code)
    st.write("Content length:", len(r.content))
except Exception as e:
    st.error(f"HTTP request failed: {e}")
import zipfile, io

if r.status_code == 200:
    z = zipfile.ZipFile(io.BytesIO(r.content))
    st.write("Files in ZIP:", z.namelist())
csv_name = [name for name in z.namelist() if name.endswith(".csv") and "Data" in name]
if csv_name:
    df = pd.read_csv(z.open(csv_name[0]), skiprows=4)
    st.write("Columns in World Bank CSV:", df.columns.tolist())
    st.write(df.head(10))
else:
    st.error("No population data CSV found in ZIP!")
st.title("Global Population Forecast (2000–2035)")

# Try live data with wbgapi
data = []
try:
    import wbgapi as wb
    for year in range(2000, 2024):
        val = list(wb.data.get('SP.POP.TOTL', 'WLD', time=year))
        if len(val) > 0 and isinstance(val[0], dict) and ('value' in val[0]) and (val[0]['value'] is not None):
            data.append({'year': year, 'population': val[0]['value']})
    pop_df = pd.DataFrame(data)
    source = "wbgapi"
except Exception as e:
    st.warning(f"wbgapi failed: {e}. Trying HTTP fallback...")
    # Fallback to World Bank HTTP API
    import requests
    import zipfile, io
    url = "http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv"
    r = requests.get(url)
    if r.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = [name for name in z.namelist() if name.endswith(".csv") and "Data" in name]
        if csv_name:
            df = pd.read_csv(z.open(csv_name[0]), skiprows=4)
            world = df[df['Country Name'] == 'World']
            years = [str(y) for y in range(2000, 2024)]
            pop_df = world.melt(id_vars=['Country Name'], value_vars=years, var_name='year', value_name='population')
            pop_df = pop_df[['year', 'population']]
            pop_df['year'] = pop_df['year'].astype(int)
            pop_df['population'] = pd.to_numeric(pop_df['population'], errors='coerce')
            pop_df = pop_df.dropna()
            source = "HTTP direct download"
        else:
            st.error("Could not find the data CSV in the World Bank ZIP file.")
            st.stop()
    else:
        st.error("Failed to download data from World Bank API.")
        st.stop()

# Show columns and first rows for diagnostics
st.write("Data source:", source)
st.write("pop_df columns:", pop_df.columns.tolist())
st.dataframe(pop_df.head())

# DataFrame column check
if pop_df.empty or 'year' not in pop_df.columns or 'population' not in pop_df.columns:
    st.error("No valid population data retrieved or columns missing. Columns present: " + str(pop_df.columns.tolist()))
    st.stop()

# Linear Regression with scikit-learn
X = pop_df[['year']]
y = pop_df['population']
model = LinearRegression().fit(X, y)

future_years = np.arange(2024, 2036).reshape(-1, 1)
future_pred = model.predict(future_years)

predicted_df = pd.DataFrame({'year': future_years.flatten(), 'population': future_pred})
final_df = pd.concat([pop_df, predicted_df], ignore_index=True)

st.subheader("Future Predictions (2024–2035)")
st.dataframe(predicted_df)

# Plot
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(final_df['year'], final_df['population'], marker='o', label='Population (Real + Predicted)')
ax.axvline(x=2023, color='red', linestyle='--', label='Prediction Starts')
ax.set_xlabel('Year')
ax.set_ylabel('Population')
ax.set_title('Global Population Trends (2000–2035) — Real + Forecast')
ax.legend()
ax.grid(True)
st.pyplot(fig)
