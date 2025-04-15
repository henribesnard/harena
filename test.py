import requests

url = "https://api.bridgeapi.io/v3/aggregation/accounts"

headers = {
    "accept": "application/json",
    "Bridge-Version": "2025-01-15",
    "Client-Id": "sandbox_id_16d8bcba6b1341b786189b8b5f42670e",
    "Client-Secret": "sandbox_secret_nk3Nok2l7sNOjQH8tzsSnakQrN6aNsqCcQkHdnev2HGsbJtf59uKkujwTzhYORQM",
    "authorization": "Bearer 8df1f692a29f5f05e7215c53dd788a4f5acdaa96-0de80455-bd22-4d89-b010-2bdd48b1ed49"
}

response = requests.get(url, headers=headers)

print(response.text)