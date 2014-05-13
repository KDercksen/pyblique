import csv


def get_data(fname):
    """Return a dictionary records/class-labels.
    Data should be accessed by their integer identifier first, data label
    second:
        # Get the first record/class-label
        first = data[0]
        # Get the records from first
        first_records = first["record"]
        # Get the class-label of first
        first_label = first["class"]
        # Get the second record
        second_record = data[1]["record"]

    Arguments:
    ----------
    fname: string
        The filename of the csv file to read data from.
    """
    data = {}
    with open(fname) as f:
        for i, row in enumerate(csv.reader(f)):
            data[i]["record"] = list(map(float, row[:-1]))
            data[i]["class"] = row[-1]
    return data
