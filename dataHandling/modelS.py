#! usr/bin/env python3
import numpy as np
from dataHandling.dataStructure import SingleTimeDatapoint, subsampleArray

def loadModelS(length: int | None = None) -> SingleTimeDatapoint:
    """
    loads model S into a single time datapoint of length length. If left at None, load the whole array
    """
    pathToModelS = "dataHandling/model_S_new.dat"
    unitSymbolBracket = "["
    header = np.loadtxt(pathToModelS, dtype=str, max_rows=1)
    variableNames = []
    for word in header:
        variableNames.append(word.split(unitSymbolBracket)[0])

    allLoadedData = np.loadtxt(pathToModelS, skiprows=1)
    if length is None:
        length = len(allLoadedData[:, 0]) + 1
    variablesDictionary = {}
    for i, variableName in enumerate(variableNames):
        variablesDictionary[variableName] = subsampleArray(allLoadedData[:, i], length)
        if variableName == "r":
            variablesDictionary["zs"] = (
                variablesDictionary[variableName][0] - variablesDictionary[variableName]
            )
            variablesDictionary.pop("r")

    datapoint = SingleTimeDatapoint(**variablesDictionary)

    return datapoint

if __name__ == "__main__":
    """ 
    here im just getting the entropy of model S using MESA EOS
    """

    import numpy as np
    
    def add_column(input_file, output_file, column_data):
        # Load the data from the input file using NumPy
        data = np.genfromtxt(input_file, dtype=float, skip_header=2)

        # Extract the header row
        with open(input_file, 'r') as f:
            header = f.readline()

        # Create a new column with the specified data
        new_column = np.full(data.shape[0], column_data, dtype=float)
        new_column = np.reshape(new_column, (data.shape[0], 1))

        # Concatenate the new column with the existing data
        updated_data = np.concatenate((data, new_column), axis=1)

        # Save the updated data to the output file, including the header row
        np.savetxt(output_file, updated_data, delimiter='   ', fmt='%.9e', header=header, comments='')


    nablas = np.loadtxt('dataHandling/tmp', usecols=1)

    add_column("dataHandling/model_S_new.dat", "dataHandling/model_S_new.dat", nablas)
