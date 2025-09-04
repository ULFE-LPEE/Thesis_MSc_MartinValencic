import pandas as pd
import numpy as np

def get_simulation_data(GEN_OUT: str, path: str):
    file_path = f"{path}/results_izpad_{GEN_OUT}.csv"
    data = pd.read_csv(file_path, delimiter=';', decimal=',', skiprows=0) # Read the CSV file
    # Remove the second 1st row from data (parameter description)
    data = data.drop(index=0)

    # Convert Time column to numeric 
    data['All calculations'] = data['All calculations'].str.replace(',', '.').astype(float)
    # Convert all other columns to numeric
    for col in data.columns[1:]:
        data[col] = data[col].str.replace(',', '.').astype(float)

    # Find the index of the first row where GEN_OUT") hits its minimum value
    min_index = int(data[GEN_OUT].idxmin())
    min_index = 2103 # We know the index from the simulation (as the min index is not always the first one)

    # Get values from all columns except the first column at the minimum index
    values_at_min_index = data.iloc[min_index - 1, 1:].values

    # Get values from all columns except the first column at the index before the minimum index
    values_before_min_index = data.iloc[min_index - 2, 1:].values
    substraction = np.array(values_at_min_index) - np.array(values_before_min_index)

    # Get the index of generator that was disturbed
    gen_out_index = data.columns.get_loc(GEN_OUT)
    substraction[gen_out_index - 1] = 0 # type: ignore
    rdP = substraction / sum(substraction) * 100
    print(sum(substraction))

    # Create generator name order list
    generator_name_order = data.columns[1:].tolist()

    return rdP, generator_name_order, data