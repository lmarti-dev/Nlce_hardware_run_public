import json

with open("file.json", "r") as f:
    data = json.load(f)

print(json.dumps(data, indent=4))