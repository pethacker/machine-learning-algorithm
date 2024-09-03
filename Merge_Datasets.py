import pandas as pd
import numpy as np

def get_columns(file):
    # Read the first line of the file to get the column names
    df = pd.read_csv(file, header=None, nrows=1, encoding='latin1')
    columns = df.iloc[0].tolist()
    return columns

def writeData(file, chunksize=100000):
    print(f"Loading raw data from {file} in chunks...")
    return pd.read_csv(file, header=0, low_memory=False, encoding='latin1', chunksize=chunksize, skiprows=1)

# Remove dirty data from the dataset, rows containing data such as NaN, Infinity, etc.
def clearDirtyData(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# Merge multiple Dataframe data by rows
def mergeData():
    files = [
        "D:/毕设/GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
        "D:/毕设/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "D:/毕设/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "D:/毕设/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "D:/毕设/GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "D:/毕设/GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "D:/毕设/GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
        "D:/毕设/GeneratedLabelledFlows/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv"
    ]

    # Getting Column Name
    columns = get_columns(files[0])

    frames = []

    for file in files:
        chunk_list = []
        for chunk in writeData(file):
            chunk.columns = columns  # Setting Column Name
            chunk = clearDirtyData(chunk)
            chunk_list.append(chunk)
        df = pd.concat(chunk_list, ignore_index=True)
        frames.append(df)

    for i, df in enumerate(frames):
        print(f"File {files[i]} - shape before merge: {df.shape}")

    # Merge data
    result = pd.concat(frames, ignore_index=True)

    # check again
    result = clearDirtyData(result)
    print(f"Shape after merge: {result.shape}")

    # Check the distribution of values in column 85, the labelled columns
    print("Value counts for the last column after cleaning:")
    print(result.iloc[:, 84].value_counts())

    return result

if __name__ == '__main__':
    raw_data = mergeData()
    file = 'data/total.csv'
    raw_data.to_csv(file, index=False, header=False)
