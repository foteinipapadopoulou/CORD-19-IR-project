import matplotlib.pyplot as plt
import pandas as pd

# Change the variant topic depending on what you want to plot('title','narrative','description')
variant_topic = 'title'
csv_file = f'./avg_query_time_csv_img/avg_query_time_{variant_topic}.csv'
df = pd.read_csv(csv_file)

# Set font Family to serif to match the font in latex
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.figure(figsize=(10, 6))
for i, (index, row) in enumerate(df.iterrows()):
    # Plot the bar for each method of average query time and add the preferred color in the last column of csv
    plt.bar(row['Method'], row['Time (ms)'], color=row['Color'], label=row['Method'])

plt.ylabel('Time (ms)')
plt.xlabel('Retrieval Models')
plt.legend(title='Retrieval Models', loc='upper left', bbox_to_anchor=(0, 1))

# Display the time(ms) on top of each bar
for i, value in enumerate(df['Time (ms)']):
    plt.text(i, value + 1, f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()

# Remove the x labels on x-axis
plt.xticks([])

plt.savefig(f'./avg_query_time_csv_img/avg_query_time_{variant_topic}.png')
plt.show()

