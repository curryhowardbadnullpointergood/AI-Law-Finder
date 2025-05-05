import random
import numpy as np
import ast
import pandas as pd
import re
import subprocess
import ast
import pandas as pd
import re
import numpy as np
import plotly.graph_objects as go

# this cleans up the mess. 
subprocess.run([
    'sed',
    's/PySRRegressor\\.equations_ =//g',
    'y_results_noise_Simple_Harmonic_Motion_i.txt'
], stdout=open('output_file_conservation.txt', 'w'))


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


file_path = './output_file_conservation.txt'

txt_data = read_txt_file(file_path)
lines = txt_data.splitlines()

lines_with_pattern = [line for line in lines if ">>>>" in line]

# Print the lines with the pattern
for line in lines_with_pattern:
    print(line)


with open("y_results_noise_conservation_2.txt", "w") as file:
        file.write(",".join(map(str, lines_with_pattern)))
        
        
with open('output_2_conservation.txt', 'w') as outfile:
    subprocess.run(
        ["sed", r"s/x[^ ]*//g", "y_results_noise_conservation_2.txt"],
        stdout=outfile
    )

with open('output_3_conservation.txt', 'w') as outfile:
    subprocess.run(
        ["sed", r"s/\*//g", "output_2_conservation.txt"],
        stdout=outfile
    )
    


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data





file_path = './output_3_conservation.txt'

txt_data = read_txt_file(file_path)



sections = txt_data.split(',')
results = []

for section in sections:
    numbers = re.findall(r'[\d.eE+-]+', section)
    if len(numbers) >= 2:
        results.append(float(numbers[-2]))

print(results)

averages = []


averages = []

for i in range(0, len(results), 2):
    if i + 1 < len(results):
        pair_sum = results[i] + results[i + 1]
        average = pair_sum / 2
        averages.append(average)
    else:
        # Handle the last unpaired element if needed
        averages.append(results[i])  # or just append it as-is


    
print(averages)

y_results = averages


x_values = np.arange(len(y_results))







fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_values,
    y=y_results,
    mode='lines+markers',
    name='Noise vs Value',
    line=dict(color='blue')
))

fig.update_layout(
    title='Noise vs Value Plot',
    xaxis_title='Noise',
    yaxis_title='Value of Noise',
    legend_title='Legend',
    template='plotly_white',
    width=800,
    height=500,
)

# Save as a static image (you need kaleido installed)
fig.write_image("noise.pdf")

fig.show()





