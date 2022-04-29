import requests
import json
import datetime as dt
import time
CIRCLE = "[9.189731 45.464157 12km]"
START_DATE = "2013-12-01T00:00:00Z"
END_DATE = "2014-01-01T23:59:59.000Z"

def get_tweets(max_results:int):
    """Gets the first set of Tweets from the Twitter API
    
    Args
    max_results - number of tweets that should be obtained per page
    
    Returns
    JSON from the API response"""
    with open("keys.json", "r") as f:
        keys = json.load(f)
    url = "https://api.twitter.com/2/tweets/search/all?"+query_builder(max_results)
    x = requests.get(url, headers = {"Authorization" : "Bearer " + keys["Bearer Elevated"]})
    print(x.headers["x-rate-limit-reset"])
    if x.status_code == 428:
        wait_time = int(x.headers["x-rate-limit-reset"])
        print("Rate exceeded proceeding in " +str(wait_time))
        time.sleep(wait_time)
        print("Proceeding")
        x = requests.get(url, headers = {"Authorization" : "Bearer " + keys["Bearer Elevated"]})
    return x.json()

def get_tweets_next(max_results:int, next:str):
    """Gets the subsequent sets of Tweets from the Twitter API
    
    Args
    max_results - number of tweets that should be obtained per page
    next - next page token
    
    Returns
    JSON from the API response"""
    with open("keys.json", "r") as f:
        keys = json.load(f)
    url = "https://api.twitter.com/2/tweets/search/all?"+query_builder(max_results, next = next)
    x = requests.get(url, headers = {"Authorization" : "Bearer " + keys["Bearer Elevated"]})
    print(x.headers["x-rate-limit-reset"])
    if x.status_code == 428:
        wait_time = int(x.headers["x-rate-limit-reset"])
        print("Rate exceeded proceeding in " +str(wait_time))
        time.sleep(wait_time)
        print("Proceeding")
        x = requests.get(url, headers = {"Authorization" : "Bearer " + keys["Bearer Elevated"]})
    return x.json()

def query_builder(max_results: int , fromDate = START_DATE, toDate = END_DATE, next = None):
    """Builds a query for a get request
    
    Args
    max_results - Number of results to retrieve per request
    
    Keyword Args
    fromDate - Date to start the retreival form. Default is the 1st of November 2013
    toDate - Date to perform the retreival to. Default is the 1st of January 2013
    next - next page token"""
    #query = "query=point_radius:"+CIRCLE+"&end_time=" + toDate +"&max_results=10&user.fields=created_at"
    if next != None:
        query = "query=point_radius:"+CIRCLE+"&end_time=" + toDate +"&max_results="+str(max_results)+"&next_token="+next+"&tweet.fields=created_at"
    else:
        query = "query=point_radius:"+CIRCLE+"&end_time=" + toDate +"&max_results="+str(max_results)+"&tweet.fields=created_at"
    return query

def data_builder_count_next(max_results: int , fromDate = START_DATE, toDate = END_DATE, next = None):
    """Function used to test, obsolete"""
    data_dict = {"query" : "point_radius:"+CIRCLE,
                 "fromDate" : fromDate,
                 "toDate" : toDate,
                 "next" : next}
    print(data_dict)
    with open("temp.json", "r") as f:
        x = json.load(f)
    return x

def request(url = None, json = None, headers = None):
    """Function used to test, obsolete"""
    return 

def handle_response(response: dict):
    """Process the response from a request
    
    Args
    response- response to proccess"""
    next = response["meta"]["next_token"]
    with open("next.txt", "w") as f:
        f.write(next)
    results = response["data"]
    for result in results:
        print(result)
        tweet_time = dt.datetime.strptime(result["created_at"], "%Y-%m-%dT%H:%M:%S.000Z")
        filename = tweet_time.strftime("%d%m%y")+"tweets.json"
        try:
            with open(filename,"x") as f:
                file_json = {"tweets":[]}
                file_json["tweets"].append(result)
                json.dump(file_json, f) 
        except:
            with open(filename, "r+") as f:
                file_json = json.load(f)
                file_json["tweets"].append(result)
            with open(filename, "w+") as f:
                json.dump(file_json, f)       

def save_tweets(tweets:dict):
    """Temporary test function, obsolete"""
    with open("temp.json", "w") as f:
        json.dump(tweets, f)

def load_tweets():
    """Temporary test function, obsolete"""
    with open("temp.json", "r") as f:
         out = json.load(f)
    return out
if __name__ == "__main__":
    #with open("next.txt", "r") as f:
    #    next = f.read()
    #print(next)
    #tweets = get_tweets(500)
    #handle_response(tweets)
    for i in range(0,200):
        with open("next.txt", "r") as f:
            next = f.read()
        print(next)
        tweets = get_tweets_next(500,next)
        print(tweets)
        handle_response(tweets)
