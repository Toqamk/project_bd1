import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("ford.csv")

# Show abnormal prices using a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=df["price"])
plt.title('Abnormal Prices Boxplot')
plt.show()

# Save the visualization as an image
plt.savefig("/home/bd-a1/boxplot.png")

# Pie chart for fuel types
types = df["fuelType"].value_counts()
threshold = 0.05 * types.sum()
small_slices = types[types < threshold]
types["Others"] = small_slices.sum()
types.drop(small_slices.index, inplace=True)
plt.figure(figsize=(8, 8))
plt.pie(types, labels=types.index, autopct='%1.2f%%')
plt.title('Fuel Type Distribution')
plt.show()

# Save the visualization as an image
plt.savefig("/home/bd-a1/piechart.png")


# Barplot for year vs. price
plt.figure(figsize=(15, 7))
sns.barplot(x="year", y="price", data=df, order=df["year"].sort_values().unique())
plt.title('Car Prices Over Years')
plt.show()

# Save the visualization as an image
plt.savefig("/home/bd-a1/barplot.png")


# Histogram of the 'price' column
plt.figure(figsize=(8, 6))
plt.hist(df['price'])
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Car Prices')
plt.show()

# Save the visualization as an image
plt.savefig("/home/bd-a1/hist.png")


# Correlation matrix heatmap
print(df.dtypes)
numeric_columns = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Save the visualization as an image
plt.savefig("/home/bd-a1/heatmap.png")






