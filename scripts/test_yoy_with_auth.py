import requests

# CORRECT: Port 8000 (local_app.py - tous les services intégrés)
url = "http://localhost:8000/api/v1/metrics/expenses/yoy"

# Token JWT pour user_id=100
headers = {
  'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjAzODg0NDcsInN1YiI6IjEwMCIsInBlcm1pc3Npb25zIjpbImNoYXQ6d3JpdGUiXX0.P6Uga0xm3RgRCWDv96stimmYv2Ow36As-Am4SVDKqMU'
}

response = requests.get(url, headers=headers)

print("Status:", response.status_code)
print("\nResponse:")
print(response.json())
