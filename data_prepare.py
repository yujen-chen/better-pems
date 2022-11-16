import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from deta import Deta



flow_col_range = list(range(0, 12))
flow_col_names = ['timestamp', 'station_id', 'district', 'route', 'direction',
             'lane_type', 'station_length', 'samples', 'pct_observed', 'total_flow',
             'avg_occu', 'avg_spd']

# a function to



df_raw = pd.read_csv(raw_data_path, header=None, usecols=col_range, names=col_names)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp']).dt.tz_localize(None)




# upload the file to deta.sh


test_drive = deta.Drive("test_drive")
test_txt = test_drive.get('test.txt')
content = test_txt.read()
print(content)
test_txt.close()
