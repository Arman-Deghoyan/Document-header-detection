import numpy as np
import pandas as pd
from tqdm import tqdm

def transform_raw_data(file_path):
    
    with open(file_path, encoding="utf8") as f:
        lines = [line.rstrip() for line in f]

    df_list = []
    for document in tqdm(lines[14000:]):

        labels = []
        values = []
        index = 0

        for line_dict in eval(document):
            
            if 'type' not in line_dict:
                labels.append("UNKNOWN")
            else:   
                labels.append(line_dict['type'])
            values.append([])

            for value in line_dict['values']:
                values[index].append(value['value'])

            index += 1

        current_df = pd.DataFrame({'row': values, 'label': labels})
        current_df['row'] = current_df['row'].apply(lambda x: " ".join((" ".join(x)).split()))
        current_df['label'] = np.where(current_df['label'] == "HEADERS", 1, 0)
        df_list.append(current_df)

    return df_list
