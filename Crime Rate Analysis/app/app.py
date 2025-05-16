import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

st.set_page_config(layout="centered")
st.title("📊 Crime Rate Analysis and Prediction (India)")

# Load and clean data
df = pd.read_csv("../data/Volume 1 Crime .csv")
df = df.iloc[1:43, 0:4].copy()
df.columns = ["S.No", "Year", "Crime Incidence", "Crime Rate"]
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Crime Incidence"] = pd.to_numeric(df["Crime Incidence"], errors="coerce")
df["Crime Rate"] = pd.to_numeric(df["Crime Rate"], errors="coerce")

# Drop NaNs (if any)
df.dropna(inplace=True)

# Show dataframe
st.subheader("📁 Crime Data (1981–2022)")
st.dataframe(df)

# 1️⃣ Crime Incidence Trend
st.subheader("📈 Total Crime Incidence Over the Years")
fig1, ax1 = plt.subplots()
sns.lineplot(x=df["Year"], y=df["Crime Incidence"], ax=ax1, marker="o", color="crimson")
ax1.set_xlabel("Year")
ax1.set_ylabel("Total Crime Incidence")
ax1.set_title("Crime Incidence in India (1981–2022)")
st.pyplot(fig1)

# 2️⃣ Crime Rate Trend
st.subheader("📉 Crime Rate per Lakh Population Over the Years")
fig2, ax2 = plt.subplots()
sns.lineplot(x=df["Year"], y=df["Crime Rate"], ax=ax2, marker="o", color="teal")
ax2.set_xlabel("Year")
ax2.set_ylabel("Crime Rate")
ax2.set_title("Crime Rate (per 1 lakh population)")
st.pyplot(fig2)

# 3️⃣ Top 5 Years by Crime Incidence
st.subheader("🔺 Top 5 Years with Highest Crime Incidence")
top5 = df.sort_values(by="Crime Incidence", ascending=False).head(5)
fig3, ax3 = plt.subplots()
sns.barplot(x=top5["Year"], y=top5["Crime Incidence"], ax=ax3, palette="Reds_r")
ax3.set_title("Top 5 Years with Highest Crimes")
st.pyplot(fig3)

# 4️⃣ Bottom 5 Years by Crime Incidence
st.subheader("🔻 Bottom 5 Years with Lowest Crime Incidence")
bottom5 = df.sort_values(by="Crime Incidence", ascending=True).head(5)
fig4, ax4 = plt.subplots()
sns.barplot(x=bottom5["Year"], y=bottom5["Crime Incidence"], ax=ax4, palette="Blues")
ax4.set_title("Bottom 5 Years with Lowest Crimes")
st.pyplot(fig4)

# 5️⃣ ML Model: Linear Regression for Prediction
st.subheader("🔮 Predict Future Crime Incidence with ML")

# Train model
model = LinearRegression()
X = df[["Year"]]
y = df["Crime Incidence"]
model.fit(X, y)

# User input
year_input = st.number_input("Enter a Year (2023–2030)", min_value=2023, max_value=2030, step=1)
prediction = model.predict([[year_input]])
st.success(f"Predicted Crime Incidence in {year_input}: {int(prediction[0]):,}")

# Actual vs Predicted Plot
st.subheader("📊 Actual vs Predicted (Regression Line)")
fig5, ax5 = plt.subplots()
X_np = X["Year"].values.reshape(-1, 1)
y_np = y.values
y_pred = model.predict(X_np)

ax5.scatter(X_np, y_np, label='Actual', color='blue')
ax5.plot(X_np, y_pred, label='Regression Line', color='red')
ax5.set_xlabel("Year")
ax5.set_ylabel("Crime Incidence")
ax5.set_title("Linear Regression: Actual vs Predicted")
ax5.legend()
st.pyplot(fig5)
