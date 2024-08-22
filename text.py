import pandas as pd

df=pd.read_excel("POP.xlsx")



exec('''
import plotly.graph_objects as go

# Selecting the relevant columns for the graph
df_selected = df[['Country', 'Population']].head(7)

# Sorting the data in descending order based on population
df_selected_sorted = df_selected.sort_values(by='Population', ascending=False)

# Creating the bar graph
fig = go.Figure(data=go.Bar(x=df_selected_sorted['Country'],
                           y=df_selected_sorted['Population'],
                           marker_color='blue'))

# Adding labels and title to the graph
fig.update_layout(title='Population in Top 7 Countries',
                  xaxis_title='Country',
                  yaxis_title='Population')

# Saving the graph in html format
fig.write_html('temp/graph_ca8f6973581e11ef8c84fff0fd93fc55.html')
''')