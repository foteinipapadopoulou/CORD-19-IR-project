import re
import numpy as np

"""
Calculate the average execution query time based on the output logs during retrieval.
The logs contain for each query a phrase of 'Finished executing query 48 in 151ms - 1000 results retrieved'
Here we are collecting the execution time for each query from the logs and we calculate the average.
The logs are stored at the './retrieval_query_time_logs' and the txt name of each log indicates the used method
"""
def extract_execution_times(file_path):
    times = []
    pattern = r'Finished executing query \d+ in (\d+)ms'

    # read the log file
    with open(file_path, 'r') as file:
        for line in file:
            # check if the line contains the pattern
            match = re.search(pattern, line)

            if match:
                # get only the second number
                time_ms = int(match.group(1))
                times.append(time_ms)

    return times


def calculate_average(query_exec_times):
    if query_exec_times:
        # get the average execution time
        return np.mean(query_exec_times)
    else:
        return None


if __name__ == '__main__':
    # file path to be read
    # change it according to which log file you want to process
    file_path = '../../retrieval_query_time_logs/std_BM25_narrative_logs.txt'

    execution_times = extract_execution_times(file_path)
    avg_time = calculate_average(execution_times)
    print(f'Average execution time: {avg_time} ms')
