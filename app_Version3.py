import os
import findspark
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import wbgapi as wb

# ---- Environment setup ----
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.1-bin-hadoop3"
findspark.init()

# ---- Streamlit app ----
st.title("Global Population Forecast (2000–2035)")

spark = SparkSession.builder.appName("PopulationTrends").getOrCreate()

# ---- Fetch World Bank data ----
pop_df = wb.data.DataFrame('SP.POP.TOTL', economy='WLD', time=range(2000, 2024))
pop_df = pop_df.rename(columns={'value': 'population'}).reset_index()
pop_df['year'] = pop_df['time'].astype(int)
pop_df = pop_df[['year', 'population']]

st.subheader("Sample (Real) Population Data")
st.dataframe(pop_df.head())

# ---- PySpark pipeline ----
df = spark.createDataFrame(pop_df)
assembler = VectorAssembler(inputCols=["year"], outputCol="features")
df_features = assembler.transform(df)

lr = LinearRegression(featuresCol="features", labelCol="population")
lr_model = lr.fit(df_features)

# ---- Predict future ----
future_years = [(y,) for y in range(2024, 2036)]
future_df = spark.createDataFrame(future_years, ["year"])
future_features = assembler.transform(future_df)
future_predictions = lr_model.transform(future_features)
predicted = future_predictions.select("year", col("prediction").alias("population"))
final_df = df.union(predicted)
pandas_df = final_df.orderBy("year").toPandas()

st.subheader("Future Predictions (2024–2035)")
st.dataframe(pandas_df.tail(15))

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(pandas_df["year"], pandas_df["population"], marker="o", label="Population (Real + Predicted)")
ax.axvline(x=2023, color="red", linestyle="--", label="Prediction Starts")
ax.set_xlabel("Year")
ax.set_ylabel("Population")
ax.set_title("Global Population Trends (2000–2035) — Real + Forecast")
ax.legend()
ax.grid(True)
st.pyplot(fig)