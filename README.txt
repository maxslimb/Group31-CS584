The Project is divided into 3 parts:
➡ Flask
      • Flask will take the repository_name from the body of the api(i.e. from React) and will fetch the created issues for the given repository for past 2 year
      • It will use group_by to group the created and closed issues for a given month and will return back the data to client
      • It will then use the data obtained from the GitHub and pass it as a input request in the POST body to Forecasting App
          to predict and to forecast the data
      • The response obtained from Forecasting service is also return back to react client app

➡ Forecasting
      1. Forecasting will accept the GitHub data from flask app and will forecast the data for past 2 year based on past 30 days
      2. It will also plot three different graph (i.e.  "model_loss", "Forecast_generated_data", "all_issues_data")using matplot lib 
      3. It will create observations and forcasting graphs using StatsModel
      4. It will create Forecasting and forecasting components (trend,weekly,daily) graphs
      3. This graph will be stored as image in gcloud storage bucket
      4. The image URL are then returned back to flask app as a json response 

➡ React
      1. React will retrieve GitHub all the requirements for a given repository and will display the bar-charts, line charts and stack bars of same using high-charts        
      2. It will also display the images of the forecasted data for the given GitHub repository and images are being retrieved from GCP storage
      3. React will make a fetch api call to flask app.


Implementation: 
      ‣ Firstly, fetched the data from Github for issues for all the 10 repository
      ‣ Further, sent the the data a json body to Forecasting service hosted on 8080 port
      ‣ Forecasting service will then do Time-Series Forecasting using Tensorflow, Facebook Prophet and Statsmodel and save all the graphs to Google Cloud Storage
      ‣ Flask then sends all the image url as a json request to the React app hosted on port 3000
      ‣ React collects all the links and then populates the images on the web browser

Run Commands:
To run & compile Flask & Forecasting:
    Please add an GITHUB_TOKEN at Line 42 in /FLask/app.py 
    Also, you have to create an Google Cloud Project and put the credentials in the Forecasting directory
    Set the following Environment variables before running Forecasting app:
            a. GOOGLE_APPLICATION_CREDENTIALS     "<Project_credntials_filename>.json"
            b. BASE_IMAGE_PATH                    "https://storage.googleapis.com/PROJECT-NAME/"
            c. BUCKET_NAME                         "Bucket name"
    Run Commands Flask & Forecasting App:
    pip install -r requirements.txt
    python app.py 

To run & install all the lib for React:
    npm install
    npm start

All the versions of the libraries used are there in the requirements.txt and package.json file 


