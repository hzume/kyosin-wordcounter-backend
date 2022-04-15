import requests
from requests import RequestException
import json
data={"name": "hoge", "text": "the a huga", "output": "C:/Users/user/Desktop/output.csv"}
res = requests.post("http://127.0.0.1:2000/read_item", json=data)
try:
    res.raise_for_status()
except RequestException as e:
    print(e.response.text)