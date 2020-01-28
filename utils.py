import numpy as np
import pandas as pd



def trim_index_csv():
    index_csv = pd.read_csv('/Volumes/Conlon Backup 2TB/GCP Sentinel Hosting/index.csv')
    index_csv_usa = index_csv[(index_csv['NORTH_LAT'] <= 15) &
                              (index_csv['SOUTH_LAT'] >= 3) &
                              (index_csv['WEST_LON'] >= 32) &
                              (index_csv['EAST_LON'] <= 48) &
                              (index_csv['GEOMETRIC_QUALITY_FLAG'] != 'FAILED') &
                              (index_csv['CLOUD_COVER'] <= 30)]
    index_csv_usa.to_csv('/Volumes/Conlon Backup 2TB/GCP Sentinel Hosting/index_eth_only.csv')

