from svgutils import compose as svg

from figures.local_config import figure_output_path

fig1_path = figure_output_path / 'fig1' / 'automatic'

fig_combined = svg.Figure(fig1_path / 'fig1B.svg')

fig_combined.save(fig1_path / 'fig1_combined.svg')
