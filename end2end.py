import json
import requests

queries = []
with open('./data/requests.json', 'r') as f:
    for line in f:
        queries.append(json.loads(line))

endpoint = "http://0.0.0.0/predict"
s = requests.Session()
for q in queries:
    response = s.post(endpoint, data=json.dumps(q)).json()
