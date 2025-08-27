import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import wbgapi as wb
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Global Population Forecast (2000–2035)")

# Fetch population data and robustly get the year column
pop_df = wb.data.DataFrame('SP.POP.TOTL', economy='WLD', time=range(2000, 2024)).reset_index()
# Check which time column exists
time_col = 'time' if 'time' in pop_df.columns else 'Time'
pop_df['year'] = pop_df[time_col].astype(int)
pop_df = pop_df.rename(columns={'value': 'population'})
pop_df = pop_df[['year', 'population']]

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
