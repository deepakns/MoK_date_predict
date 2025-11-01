import os
from data_process import make_save_monthly_average

# call the function to make and save monthly average data
root_dir = "/gdata2/ERA5/"
make_save_monthly_average(root_dir, start_year=2008, end_year=2025)
