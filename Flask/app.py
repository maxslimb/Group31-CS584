import os
from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
import json
import dateutil.relativedelta
from dateutil import *
from datetime import date
import pandas as pd
import requests
from datetime import datetime as dt

app = Flask(__name__)
CORS(app)

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API accepts only POST request
'''


@app.route('/api/github', methods=['POST'])
def github():
    body = request.get_json()
    # Extract the choosen repositories from the request
    repo_name = body['repository']
    # Add your own GitHub Token to run it local
    token = os.environ.get(
        'GITHUB_TOKEN', 'GITHUB_TOKEN')
    GITHUB_URL = f"https://api.github.com/"
    headers = {
        "Authorization": f'token {token}'
    }
    params = {
        "state": "open"
    }
    repository_url = GITHUB_URL + "repos/" + repo_name
    # Fetch GitHub data from GitHub API
    repository = requests.get(repository_url, headers=headers)
    # Convert the data obtained from GitHub API to JSON format
    repository = repository.json()


    today = date.today()

    issues_reponse = []
    print("REPO: ",repo_name)
    
    #Fetch ISSUES DATA
    today = date.today()
    for i in range(24):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        types = 'type:issue'
        repo = 'repo:' + repo_name
        ranges = 'created:' + str(last_month) + '..' + str(today)
        per_page = 'per_page=100'
        search_query = types + ' ' + repo + ' ' + ranges

        # Append the search query to the GitHub API URL 
        query_url = GITHUB_URL + "search/issues?q=" + search_query + "&" + per_page
      
        search_issues = requests.get(query_url, headers=headers, params=params)
        search_issues = search_issues.json()
        
       
        issues_items = []
        try:
            # Extract "items" from search issues
            issues_items = search_issues.get("items")
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        if issues_items is None:
            continue
        for issue in issues_items:
            label_name = []
            data = {}
            current_issue = issue
            # Get issue number
            
            data['issue_number'] = current_issue["number"]
            # Get created date of issue
            data['created_at'] = current_issue["created_at"][0:10]
            if current_issue["closed_at"] == None:
                data['closed_at'] = current_issue["closed_at"]
            else:
                # Get closed date of issue
                data['closed_at'] = current_issue["closed_at"][0:10]
            for label in current_issue["labels"]:
                # Get label name of issue
                label_name.append(label["name"])
            data['labels'] = label_name
            # It gives state of issue like closed or open
            data['State'] = current_issue["state"]
            issues_reponse.append(data)

        today = last_month
    df = pd.DataFrame(issues_reponse)
    print("DF: ",df)

    # Daily Created Issues
    df_created_at = df.groupby(['created_at'], as_index=False).count()
    dataFrameCreated = df_created_at[['created_at', 'issue_number']]
    dataFrameCreated.columns = ['date', 'count']

    '''
    Monthly Created Issues
    Format the data by grouping the data by month
    ''' 
    created_at = df['created_at']
    month_issue_created = pd.to_datetime(
        pd.Series(created_at), format='%Y-%m-%d')
    month_issue_created.index = month_issue_created.dt.to_period('m')
    month_issue_created = month_issue_created.groupby(level=0).size()
    month_issue_created = month_issue_created.reindex(pd.period_range(
        month_issue_created.index.min(), month_issue_created.index.max(), freq='m'), fill_value=0)
    month_issue_created_dict = month_issue_created.to_dict()
    created_at_issues = []
    for key in month_issue_created_dict.keys():
        array = [str(key), month_issue_created_dict[key]]
        created_at_issues.append(array)
    #print(created_at_issues)


    created_at_body = {
        "issues": issues_reponse,
        "type": "created_at",
        "repo": repo_name.split("/")[1]
    }
   
    LSTM_API_URL = "http://127.0.0.1:8080/" + "api/forecast"
    LSTM_API_URL_FINAL = "http://127.0.0.1:8080/"
   
    
    created_at_response = requests.post(LSTM_API_URL,
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})

    created_at_response_stat = requests.post(LSTM_API_URL_FINAL+"api/statmis",
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})

    created_at_response_fb = requests.post(LSTM_API_URL_FINAL+"api/fbprophetis",
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})



    json_response = {
       
        "stat_createdAtImageUrls": {
            **created_at_response_stat.json(),
        },

         "fb_createdAtImageUrls": {
            **created_at_response_fb.json(),
        },

        "createdAtImageUrls": {
            **created_at_response.json(),
        },
    }
    # Return the response back to client (React App)
    return jsonify(json_response)


# Run flask app server on port 9000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)
