import csv
from io import StringIO


def csv_to_data(bytes: bytes, data_id: int, target_id: int) -> dict:
    reader = csv.reader(StringIO(bytes.decode()))
    next(reader)  # skip header TODO: define csv format

    data: list[float] = []
    target: list[float] = []
    for row in reader:
        data.extend(map(float, row[1:-1]))
        target.append(float(row[-1]))

    result = {data_id: data, target_id: target}
    return result
