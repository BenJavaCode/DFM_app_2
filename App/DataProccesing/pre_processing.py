import csv
import numpy as np
from BaselineRemoval import BaselineRemoval


def cleanse_n_sort(path, slicer=None):

    """
    cleanse_n_sort(path, slicer=None)
    Description: For cleaning a rastascan .csv file from junk info,
                 and sorting data into data, coordinates and wavelengths.
    Params: path = path to rastascan .csv file.
            slicer = Slicer is a tuple for slicing rastascan into slice.. eg[10:20]
                     Lets user determine what traces, in a scan they want to examine.
    Latest update: 03-06-2021. Added more comments.
                               Refactored  [3:-1] to x[3:]
    """

    data = []
    coordinates = []
    wavelength = []
    counter = 0
    with open(path, "r") as f:
        reader = csv.reader(f)
        while True:
            try:
                x = next(reader)
                if x[0] == 'id':  # Take wavelength data
                    wavelength.extend([float(i) for i in x[3:]])
                else:
                    if slicer is None:  # If the user wants to examine whole rastascan
                        coordinates.append([int(i) for i in x[1:3]])
                        data.append([float(i) for i in x[3:]])
                    elif counter >= slicer[0]:  # For slicing rastascan
                        coordinates.append([int(i) for i in x[1:3]])
                        data.append([float(i) for i in x[3:]])
                    if slicer is not None:
                        counter += 1
            except:
                break
            if slicer is not None and counter > slicer[1]:  # break if end in slice tuple reached
                break

    return wavelength, coordinates, data


def measure_data_lengths(path):

    """
    measure_data_lengths(path)
    Description: For measuring length of traces in rastascan, and for counting rows.
    Params: path = path to .csv file containing rastascan.
    Latest update: 03-06-2021. Added comments.
                               Refactored data len from x[3:-1] to x[3:]
    """

    row_count = None
    data_len = None
    # Count rows
    with open(path, "r") as f:
        row_count = sum(1 for row in f) - 1
    # Determining trace length
    with open(path, 'r') as f:
        reader = csv.reader(f)
        x = next(reader)
        x = next(reader)
        data_len = len([float(i) for i in x[3:]])

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


def clean_and_convert(arr, zhang=True, reshape_length=False):

    """
    clean_and_convert(arr, zhang=True, reshape_length=False)
    Description: For preparing data for model,
                 it can reshape data, apply baseline-removal and it normalizes to (0-1)
    Params: arr = Input array.
            zhang = Boolean value. If True baseline will be removed.
                    If false baseline will not be removed
            reshape_length = An int.
                             Determines the length of each trace, after cleaning.
                             If False, original length will be kept.
    Latest update: 03-06-2021. Added more comments.
    """

    ars_cleaned = []

    for i in range(len(arr)):

        # RESHAPE
        if reshape_length is False:
            temp = arr[i]
        else:
            temp = arr[i][:- (len(arr[i]) - int(reshape_length))]
        # -

        # BASELINE-REMOVAL
        if zhang is True:
            baseObj = BaselineRemoval(temp)
            temp = baseObj.ZhangFit()
        # -

        temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))  # Normalizes traces between 0-1
        ars_cleaned.append(temp)

    return np.array(ars_cleaned)
