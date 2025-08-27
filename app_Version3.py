import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import wbgapi as wb
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Global Population Forecast (2000–2035)")

# Fetch data as a dictionary, then build a DataFrame
data = []
for year in range(2000, 2024):
    val = wb.data.get('SP.POP.TOTL', 'WLD', time=year)
    val = list(val)
    if len(val) > 0:
        value = val[0]['value']
        data.append({'year': year, 'population': value})

pop_df = pd.DataFrame(data)

st.subheader("Sample (Real) Population Data")
st.dataframe(pop_df.head())

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
