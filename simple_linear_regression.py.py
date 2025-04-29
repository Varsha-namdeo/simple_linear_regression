import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Title
st.set_page_config(page_title="Manual Linear Regression", layout="centered")
st.title("📈 Simple Linear Regression - Manual Calculation")
st.write("Predict exam scores based on hours studied using manually calculated regression.")

# Input data
hours = [1, 2, 3, 4, 5]
scores = [1.5, 3.7, 4.1, 5.9, 7.8]

# Step 1: Calculate the necessary sums
n = len(hours)
sum_x = sum(hours)
sum_y = sum(scores)
sum_xy = sum([x * y for x, y in zip(hours, scores)])
sum_x2 = sum([x ** 2 for x in hours])

# Step 2: Calculate the slope (m) and intercept (c)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
c = (sum_y - m * sum_x) / n

# Prediction function
def predict(x):
    return m * x + c

# Predict for existing data
predicted_scores = [predict(x) for x in hours]

# Step 3: Evaluation metrics
actual = np.array(scores)
predicted = np.array(predicted_scores)

mae = np.mean(np.abs(actual - predicted))
mse = np.mean((actual - predicted) ** 2)
rmse = np.sqrt(mse)
ss_total = np.sum((actual - np.mean(actual)) ** 2)
ss_res = np.sum((actual - predicted) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Display regression parameters
st.subheader("🔢 Model Parameters")
st.metric("Slope (m)", round(m, 3))
st.metric("Intercept (c)", round(c, 3))

# Display predictions
st.subheader("📊 Predictions on Sample Data")
df = pd.DataFrame({
    "Hours Studied": hours,
    "Actual Scores": scores,
    "Predicted Scores": [round(p, 2) for p in predicted_scores]
})
st.dataframe(df, use_container_width=True)

# Show regression line plot
st.subheader("📉 Regression Line Visualization")
fig, ax = plt.subplots()
ax.scatter(hours, scores, color='blue', label='Actual Scores')
ax.plot(hours, predicted_scores, color='red', label='Regression Line')
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Exam Score")
ax.set_title("Regression Line")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Show evaluation metrics
st.subheader("📐 Model Evaluation Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Absolute Error (MAE)", round(mae, 3))
    st.metric("Mean Squared Error (MSE)", round(mse, 3))
with col2:
    st.metric("Root Mean Squared Error (RMSE)", round(rmse, 3))
    st.metric("R-squared (R²)", round(r_squared, 3))

# Optional: Predict new input
st.subheader("🔍 Predict New Score")
new_hour = st.number_input("Enter hours studied:", min_value=0.0, step=0.5)
if new_hour:
    prediction = predict(new_hour)
    st.success(f"Predicted Score for {new_hour} hours: {round(prediction, 2)}")
