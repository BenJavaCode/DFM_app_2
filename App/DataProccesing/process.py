import csv
import numpy as np
from BaselineRemoval import BaselineRemoval


# For cleaning a rastascan csv from junk info and sorting data into data coordinates and wavelengths
def cleanse_n_sort(path, slicer=None):  # Slicer is a tuple for slicing rastascan into slice.. eg[10:20]
    data = []
    coordinates = []
    wavelength = []
    counter = 0
    with open(path, "r") as f:
        reader = csv.reader(f)
        while True:
            try:
                x = next(reader)
                if x[0] == 'id':
                    wavelength.extend([float(i) for i in x[3:-1]])
                else:
                    if slicer is None:
                        coordinates.append([int(i) for i in x[1:3]])
                        data.append([float(i) for i in x[3:-1]])
                    elif counter >= slicer[0]:
                        coordinates.append([int(i) for i in x[1:3]])
                        data.append([float(i) for i in x[3:-1]])
                    if slicer is not None:
                        counter += 1
            except:
                break
            if slicer is not None and counter > slicer[1]:  # break if end in slice tuple reached
                break

    return (wavelength,coordinates,data)


# Function for measuring length of data in rastascan and for counting rows
def measure_data_lenghts(path):
    row_count = None
    data_len = None
    with open(path, "r") as f:
        row_count = sum(1 for row in f) - 1
    with open(path, 'r') as f:
        reader = csv.reader(f)
        x = next(reader)
        x = next(reader)
        data_len = len([float(i) for i in x[3:-1]])

    return row_count, data_len


# For saving npy file, (not used)
def save_as_npy(directory, name, arr):
  with open(directory+ "/" + name, "wb") as f:
    np.save(f, arr)

# Legacy function for condensing data to specified size (not used)
def condense_to_1000(arr, reversed=False, length=1000):
  l14 = arr
  l10 = np.ones(length)

  fillindex = 0
  targetmover = 0

  if reversed == False:
    targetindex = -1
    targetmover = 1
  else:
    targetindex = len(arr)
    targetmover = -1

  value = 0
  sum = 0
  left = 0
  targetsize = len(arr)/length

  while True:
    if left == 0:
      left = 1.0
      targetindex += targetmover
      try:
        l14[targetindex]
        if targetindex < 0:
           l10[fillindex] = value
           return l10
      except:
        if fillindex < length:
          l10[fillindex] = value
        return l10

    if (targetsize - sum) < left:
      taken = (targetsize - sum)
      value += l14[targetindex]*taken
      left -= taken
      sum += taken
    else:
      value += l14[targetindex]*left
      sum += left
      left = 0

    if sum == len(arr)/length:
      l10[fillindex] = value
      fillindex+=1
      value = 0
      sum = 0


# function for preparing data for model, it can reshape data apply baselineremoval and it normalizes to (0-1)
def clean_and_convert(arr, zhang=True, reshape_length=False):  # zhang is br, reshape_length is the desired data length
    ars_cleaned = []

    for i in range(len(arr)):
        if reshape_length is False:
            temp = arr[i]
        else:
            temp = arr[i][:-(len(arr[i])-reshape_length)]

        if zhang is True:
            baseObj = BaselineRemoval(temp)
            temp = baseObj.ZhangFit()

        temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        ars_cleaned.append(temp)

    return np.array(ars_cleaned)
