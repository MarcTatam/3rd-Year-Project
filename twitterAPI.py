import requests
import json
import datetime as dt
CIRCLE = "[9.189731 45.464157 8km]"
START_DATE = "201311010000"
END_DATE = "201401010000"

def get_tweets(query:str, max_results:int):
    with open("keys.json", "r") as f:
        keys = json.load(f)
    url = "https://api.twitter.com/1.1/tweets/search/fullarchive/dev.json"
    #x = requests.post(url, json = data_builder_count(1), headers = {"Authorization" : "Bearer " + keys["Bearer Token"]})
    return request(url, json = data_builder_count(1), headers = {"Authorization" : "Bearer " + keys["Bearer Token"]})
    #return x.json()

def get_tweets_next(query:str, max_results:int, next:str):
    with open("keys.json", "r") as f:
        keys = json.load(f)
    url = "https://api.twitter.com/1.1/tweets/search/fullarchive/dev.json"
    #x = requests.post(url, json = data_builder_count(1), headers = {"Authorization" : "Bearer " + keys["Bearer Token"]})
    return request(url, json = data_builder_count(1, next), headers = {"Authorization" : "Bearer " + keys["Bearer Token"]})
    #return x.json()

def data_builder_count(max_results: int , fromDate = START_DATE, toDate = END_DATE):
    data_dict = {"query" : "point_radius:"+CIRCLE,
                 "fromDate" : fromDate,
                 "toDate" : toDate}
    print(data_dict)
    with open("temp.json", "r") as f:
        x = json.load(f)
    return x

def data_builder_count_next(max_results: int , fromDate = START_DATE, toDate = END_DATE, next = None):
    data_dict = {"query" : "point_radius:"+CIRCLE,
                 "fromDate" : fromDate,
                 "toDate" : toDate,
                 "next" : next}
    print(data_dict)
    with open("temp.json", "r") as f:
        x = json.load(f)
    return x

def request(url = None, json = None, headers = None):
    return 

def handle_response(response: dict):
    next = response["next"]
    results = response["results"]
    for result in results:
        tweet_time = dt.strptime()

if __name__ == "__main__":
    get_tweets("s",10)