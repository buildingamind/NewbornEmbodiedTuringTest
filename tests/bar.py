import matplotlib.pyplot as plt

# Sample dictionary
data = dict(
Old=10805,
New=1129
)

# Extract keys and values from the dictionary
labels = list(data.keys())
values = list(data.values())

# Create a bar chart
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
bars = plt.bar(labels, values, color='blue')  # You can customize the color

# Add title and labels
plt.title('Time to Test 20E×500 steps')
plt.ylabel('seconds')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height}',
      xy=(bar.get_x() + bar.get_width() / 2, height),
      xytext=(0, 3),  # 3 points vertical offset
      textcoords="offset points",
      ha='center', va='bottom')

# Display the bar chart
plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.savefig("bar_chart.png", dpi=300)
plt.show()
