import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "data/PS02_Retail_Customer_Purchase_data.csv"

df = pd.read_csv(file_path)


# Fill missing values for graphs
df["Device_Type"] = df["Device_Type"].fillna(
    df["Device_Type"].mode()[0]
)

df["Traffic_Source"] = df["Traffic_Source"].fillna(
    df["Traffic_Source"].mode()[0]
)


# 1. Purchase Distribution
sns.countplot(
    data=df,
    x="Purchase"
)

plt.title("Purchase Distribution")
plt.xlabel("Purchase")
plt.ylabel("Number of Customers")

plt.show()


# 2. Device Type vs Purchase
sns.countplot(
    data=df,
    x="Device_Type",
    hue="Purchase"
)

plt.title("Device Type vs Purchase")
plt.xlabel("Device Type")
plt.ylabel("Number of Customers")

plt.show()


# 3. Traffic Source vs Purchase
sns.countplot(
    data=df,
    x="Traffic_Source",
    hue="Purchase"
)

plt.title("Traffic Source vs Purchase")
plt.xlabel("Traffic Source")
plt.ylabel("Number of Customers")

plt.xticks(rotation=30)

plt.show()


# 4. Pages Visited vs Purchase
sns.boxplot(
    data=df,
    x="Purchase",
    y="Pages_Visited"
)

plt.title("Pages Visited vs Purchase")
plt.xlabel("Purchase")
plt.ylabel("Pages Visited")

plt.show()