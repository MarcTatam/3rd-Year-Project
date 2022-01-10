import json
with open("301113tweets.json","r") as f:
    tweets1 = json.load(f)
with open("301113tweets2.json","r") as f:
    tweets2 = json.load(f)

print("1: " + str(len(tweets1["tweets"])))
print("2: " + str(len(tweets2["tweets"])))


