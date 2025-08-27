import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
import zipfileimport streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
import zipfile
import io

st.title("Global Population Forecast (2000–2035)")

# Download the latest population data CSV from the World Bank API
url = "http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv"
r = requests.get(url)
if r.status_code != 200:
    st.error("Failed to download data from the World Bank API.")
    st.stop()

z = zipfile.ZipFile(io.BytesIO(r.content))
file_list = z.namelist()
# Use the main data file (not metadata)
csv_candidates = [name for name in file_list if name.startswith("API_SP.POP.TOTL") and name.endswith(".csv")]
if not csv_candidates:
    st.error("No population data CSV found in ZIP! Check file names above.")
    st.stop()

# Read the first matching CSV (should only be one)
df = pd.read_csv(z.open(csv_candidates[0]), skiprows=4)

# Show columns for diagnostics
st.write("Columns in CSV:", df.columns.tolist())
st.write(df.head(10))

# Filter for the 'World' row
world = df[df['Country Name'] == 'World']
if world.empty:
    st.error("No 'World' row found in the CSV!")
    st.stop()

# Melt years 2000-2023 to long format
years = [str(y) for y in range(2000, 2024)]
pop_df = world.melt(id_vars=['Country Name'], value_vars=years, var_name='year', value_name='population')
pop_df = pop_df[['year', 'population']]
pop_df['year'] = pop_df['year'].astype(int)
pop_df['population'] = pd.to_numeric(pop_df['population'], errors='coerce')
pop_df = pop_df.dropna()

if pop_df.empty or 'year' not in pop_df.columns or 'population' not in pop_df.columns:
    st.error("No valid population data retrieved or columns missing. Columns present: " + str(pop_df.columns.tolist()))
    st.stop()

st.subheader("Sample (Real) Population Data")
st.dataframe(pop_df.head())

# Linear Regression for prediction
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
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(final_df['year'], final_df['population'], marker='o', label='Population (Real + Predicted)')
ax.axvline(x=2023, color='red', linestyle='--', label='Prediction Starts')
ax.set_xlabel('Year')
ax.set_ylabel('Population')
ax.set_title('Global Population Trends (2000–2035) — Real + Forecast')
ax.legend()
ax.grid(True)
st.pyplot(fig)
import io

st.title("Global Population Forecast (2000–2035)")

# Download the latest population data CSV from the World Bank API
url = "http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv"
r = requests.get(url)
if r.status_code != 200:
    st.error("Failed to download data from the World Bank API.")
    st.stop()

z = zipfile.ZipFile(io.BytesIO(r.content))
file_list = z.namelist()
st.write("Files in World Bank ZIP:", file_list)

# Find the file with 'Data' and '.csv' in the name
csv_candidates = [name for name in file_list if name.endswith(".csv") and "Data" in name]
if not csv_candidates:
    st.error("No population data CSV found in ZIP! Check file names above.")
    st.stop()

# Read the first matching CSV
df = pd.read_csv(z.open(csv_candidates[0]), skiprows=4)

# Show columns for diagnostics
st.write("Columns in CSV:", df.columns.tolist())
st.write(df.head(10))

# Filter for the 'World' row
world = df[df['Country Name'] == 'World']
if world.empty:
    st.error("No 'World' row found in the CSV!")
    st.stop()

# Melt years 2000-2023 to long format
years = [str(y) for y in range(2000, 2024)]
pop_df = world.melt(id_vars=['Country Name'], value_vars=years, var_name='year', value_name='population')
pop_df = pop_df[['year', 'population']]
pop_df['year'] = pop_df['year'].astype(int)
pop_df['population'] = pd.to_numeric(pop_df['population'], errors='coerce')
pop_df = pop_df.dropna()

if pop_df.empty or 'year' not in pop_df.columns or 'population' not in pop_df.columns:
    st.error("No valid population data retrieved or columns missing. Columns present: " + str(pop_df.columns.tolist()))
    st.stop()

st.subheader("Sample (Real) Population Data")
st.dataframe(pop_df.head())

# Linear Regression for prediction
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
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(final_df['year'], final_df['population'], marker='o', label='Population (Real + Predicted)')
ax.axvline(x=2023, color='red', linestyle='--', label='Prediction Starts')
ax.set_xlabel('Year')
ax.set_ylabel('Population')
ax.set_title('Global Population Trends (2000–2035) — Real + Forecast')
ax.legend()
ax.grid(True)
st.pyplot(fig)
