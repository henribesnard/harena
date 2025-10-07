import requests

url = "http://localhost:8004/api/v1/metrics/expenses/yoy"

payload = {}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
