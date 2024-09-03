import pandas as pd


def writeData(file):
    print(f"Loading raw data from {file}...")
    return pd.read_csv(file, header=None, low_memory=False, encoding='latin1')


def lookData(raw_data):
    # Get the index of the last column
    last_column_index = raw_data.shape[1] - 1

    # Print the number of occurrences of each unique value in the last column
    # print(raw_data.iloc[:, last_column_index].value_counts())

    # Take out the last column (labelled column)
    labels = raw_data.iloc[:, last_column_index:]
    print("Labels DataFrame shape:", labels.shape)

    # Convert the multi-dimensional array to the one-dimensional array
    labels = labels.values.ravel()

    # Remove null values and generate a collection of labels
    label_set = set(labels)

    return label_set


# Sort the large dataset based on label features and store it in the lists collection
def separateData(raw_data):
    # dataframe数据转换为多维数组
    lists = raw_data.values.tolist()    # Convert raw data to list format
    temp_lists = []

    # temp_lists is used to temporarily store the generated list of 15 features
    for i in range(0, 15):
        temp_lists.append([])
    # Get the collection of data labels for raw_data
    label_set = lookData(raw_data)

    # Convert the unordered data labels collection into an ordered list
    label_list = list(label_set)
    print(label_list, len(lists), len(lists[0]))
    print(lists[0])
    for i in range(0, len(lists)):
        data_index = label_list.index(lists[i][len(lists[0]) - 1])
        temp_lists[data_index].append(lists[i])

    return temp_lists


# Expand the dataset from a small number to at least 5000 entries
def expendData(lists, output_folder):
    totall_list = []
    for i in range(0, len(lists)):
        while len(lists[i]) < 5000:
            lists[i].extend(lists[i][:5000 - len(lists[i])])
        print(f"Class {i} expanded to {len(lists[i])} samples.")
        totall_list.extend(lists[i])

    # save data
    saveData(lists, output_folder)
    save = pd.DataFrame(totall_list)
    file = f'{output_folder}/total_extend.csv'
    save.to_csv(file, index=False, header=False)

def saveData(lists, output_folder):
    label_set = lookData(raw_data)
    label_list = list(label_set)
    for i in range(0, len(lists)):
        save = pd.DataFrame(lists[i])
        file1 = f'{output_folder}/{label_list[i]}.csv'
        save.to_csv(file1, index=False, header=False)


if __name__ == '__main__':
    file = 'data/total.csv'
    raw_data = writeData(file)
    lists = separateData(raw_data)

    output_folder = 'data/expendData'
    expendData(lists, output_folder)
