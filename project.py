# =========================
# Airbnb Dynamic Pricing Recommendation Engine
# =========================

# === 1. Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# === 2. Load Data ===
df = pd.read_csv("Airbnb_Data.csv")
print("Initial Shape:", df.shape)

# === 3. Data Cleaning ===
df['price'] = np.exp(df['log_price'])  # Convert log_price to actual price
df['last_month'] = pd.to_datetime(df['last_review'], errors='coerce').dt.month.fillna(0)
df['cleaning_fee'] = df['cleaning_fee'].astype(int)

# Fill missing values
for col in ['bathrooms', 'bedrooms', 'beds', 'review_scores_rating']:
    df[col] = df[col].fillna(df[col].median())

df.dropna(subset=['city', 'room_type', 'property_type'], inplace=True)

print("Remaining Nulls:\n", df.isnull().sum())

# === 4. Exploratory Data Analysis (EDA) ===
sns.set(style="whitegrid")

# Plot 1: Price Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['price'], bins=60, kde=True, color="cornflowerblue")
plt.title('Price Distribution (with KDE)')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.close()

# Plot 2: Avg Price by Room Type (using groupby)
plt.figure(figsize=(8, 5))
room_avg = df.groupby('room_type')['price'].mean().sort_values()
room_avg.plot(kind='bar', color='teal')
plt.title('Average Price by Room Type')
plt.ylabel('Average Price')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("avg_price_by_room_type.png")
plt.close()

# Plot 3: Boxplot (Log Scale) to reduce congestion
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='room_type',
    y=np.log1p(df['price']),
    data=df,
    palette='Set2',
    width=0.6,
    fliersize=2,
    boxprops=dict(alpha=0.6),
    medianprops=dict(color='red')
)
plt.title('Log(Price) Distribution by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Log(Price)')
plt.tight_layout()
plt.savefig("boxplot_log_price_room_type.png")
plt.close()

# Plot 4: Price vs Number of Reviews
plt.figure(figsize=(8, 5))
sns.scatterplot(x='number_of_reviews', y='price', data=df, alpha=0.3)
plt.title('Price vs Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Price')
plt.tight_layout()
plt.savefig("price_vs_reviews.png")
plt.close()

# Plot 5: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Plot 6: Average Price by City (Top 10)
plt.figure(figsize=(10, 6))
city_avg = df.groupby("city")["price"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=city_avg.values, y=city_avg.index, palette="viridis")
plt.title("Top 10 Cities by Avg Price")
plt.xlabel("Average Price")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("avg_price_by_city_top10.png")
plt.close()

print("✅ All EDA plots saved.\n")

# === 5. Feature Engineering & Model Prep ===
features = ['city', 'property_type', 'room_type', 'accommodates', 'bathrooms',
            'bedrooms', 'beds', 'number_of_reviews', 'review_scores_rating',
            'cleaning_fee', 'last_month']
target = 'price'

data = df[features + [target]].dropna()
y = data[target].values
cat_cols = ['city', 'property_type', 'room_type']

# One-hot encoding for categorical features
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = enc.fit_transform(data[cat_cols])
X_num = data.drop(columns=cat_cols + [target]).values
X = np.hstack([X_cat, X_num])

# === 6. Model Training ===
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 7. Evaluation ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance:\nMAE = {mae:.2f}\nR2 Score = {r2:.2f}")

# === 8. Export Clean Data for Excel / Tableau ===
data.reset_index(drop=True).to_csv("airbnb_cleaned.csv", index=False)
print("📁 Cleaned data saved as 'airbnb_cleaned.csv'")

# === 9. Show Predicted vs Actual Prices ===
pred_df = pd.DataFrame({
    'Actual Price': y_test[:20],
    'Predicted Price': y_pred[:20]
})
print("\n🔮 Sample Predictions (first 20 rows):\n")
print(pred_df.round(2))
