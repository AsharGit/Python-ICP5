import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat

# Read file and assign variable
train = pd.read_csv('train.csv')
x = train["GarageArea"]
y = train["SalePrice"]

# Scatter plot with all data points
plt.scatter(x, y)
plt.title("Garage Area Vs. Sale Price")
plt.xlabel("Garage Area")
plt.ylabel("Sale Price")
plt.show()

# Mean and standard deviation
mean = x.mean()
stdDev = stat.stdev(x)
# Outlier set at 2 standard deviation from the mean which accounts for 95% of the set
outlier = stdDev * 2
# Set upper and lower limit from the mean for outliers
lower = mean - outlier
upper = mean + outlier

# Remove outlier values
data = train[(train["GarageArea"] > lower) & (train["GarageArea"] < upper)]
x = data["GarageArea"]
y = data["SalePrice"]

# Scatter plot without outliers
plt.scatter(x, y)
plt.title("Garage Area Vs. Sale Price (w/o outliers)")
plt.xlabel("Garage Area")
plt.ylabel("Sale Price")
plt.show()


