import requests
import time
from collections import deque

url = "http://202.115.17.253:53724/RealTimeData/csvFile"
csv_file_path = "C:/Users/LYin/learnpy/test.csv"
headers = {
    "Authorization": "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjozLCJ1c2VyX25hbWUiOiLmnIDkuIrlt50iLCJleHAiOjE3NDM2Mjk4Mjl9.-FbTZDn_HG5u8UzkFYzNOTw3CgYc3_ZNIHPtxLHlMZ4"
}

seq_len = 20

buffer = deque(maxlen=seq_len)

def stream_csv_data(file_path, data_lines):
    is_header = True
    with open(file_path, "r", encoding="utf-8") as file:
        header = file.readline()
        buffer.append(header)

        for line in file:
            if is_header:
                is_header = False
            else:
                buffer.append(line)

            if len(buffer) == data_lines:
                yield "".join([header] + list(buffer)).encode("utf-8")
                time.sleep(10)

for chunk in stream_csv_data(csv_file_path, seq_len):
    files = {"file": ("test.csv", chunk, "text/csv")}
    response = requests.post(url, files=files, headers=headers)

    print("Response Status Code:", response.status_code)
    print("Response Body:", response.text)