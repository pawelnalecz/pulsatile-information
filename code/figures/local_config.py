from pathlib import Path

# Path to save the figures to
# By default it is the Figures folder in the downloaded package
# Can be any path on your computer, e.g., 'C:\\Users\\<username>\\Documents\\Figures'
# str or pathlib.Path
figure_output_path = Path(__file__).parent.parent.parent / 'figures'
