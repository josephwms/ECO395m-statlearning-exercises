import csv
import os
import pandas as pd

folder_path = os.path.join(os.getcwd(), 'data', '09-21csv')
out_path = os.path.join(os.getcwd(), 'artifacts')

def read_csv_files(folder_path):
    data_dict = {}

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                
                # Extract column headers (variable names)
                headers = next(reader)
                
                # Count data rows
                line_count = sum(1 for row in reader)
                
                # Number of variables
                num_vars = len(headers)
                
                # Create dictionary for the file
                file_dict = {
                    'num_rows': line_count,
                    'num_vars': num_vars,
                    'vars_names': headers
                }
                
                # Add file dictionary to the data dictionary
                data_dict[filename] = file_dict
    
    return data_dict

# Read CSV files and get the dictionary
result_dict = read_csv_files(folder_path)

def output_var_names(result_dict):
    data_list = []

    for filename, file_data in result_dict.items():
        # Extract first two characters of filename to form 'year'
        year = f"20{filename[:2]}"
        
        # Create a dictionary for each file
        file_dict = {
            'year': year,
            **file_data  # Unpacks the file_data dictionary
        }
        
        data_list.append(file_dict)

    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    return df

# Get the DataFrame
df = output_var_names(result_dict)



# Write DataFrame to CSV file
output_file_path = os.path.join(out_path, 'output.csv')
df.to_csv(output_file_path, index=False, mode='w')
print(f"DataFrame has been written to {output_file_path}")

