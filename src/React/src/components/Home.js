import * as React from "react";
import { useState } from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import AppBar from "@mui/material/AppBar";
import CssBaseline from "@mui/material/CssBaseline";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Loader from "./Loader";
import { ListItemButton } from "@mui/material";

const drawerWidth = 240;
// List of GitHub repositories 
const repositories = [
  {
    key: "angular/angular",
    value: "Angular",
  },
  {
    key: "SeleniumHQ/selenium",
    value: "Selenium",
  },
  {
    key: "openai/openai-python",
    value: "Openai-Python",
  },
  {
    key: "angular/angular-cli",
    value: "Angular-cli",
  },
  {
    key: "d3/d3",
    value: "D3",
  },
  {
    key: "facebook/react",
    value: "React",
  },
  {
    key: "pallets/flask",
    value: "Flask",
  },
  {
    key: "tensorflow/tensorflow",
    value: "Tensorflow",
  },
];

export default function Home() {
  
  const [loading, setLoading] = useState(true);
  
  const [repository, setRepository] = useState({
    key: "angular/angular",
    value: "Angular",
  });

  const [githubRepoData, setGithubData] = useState([]);
  // Updates the repository to newly selected repository
  const eventHandler = (repo) => {
    setRepository(repo);
  };
  


  React.useEffect(() => {
    // set loading to true to display loader
    setLoading(true);
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // Append the repository key to request body
      body: JSON.stringify({ repository: repository.key }),
    };

    /*
    Fetching the GitHub details from flask microservice
    */
    fetch("/api/github", requestOptions)
      .then((res) => res.json())
      .then(
        // On successful response from flask microservice
        (result) => {
          // On success set loading to false to display the contents of the resonse
          setLoading(false);
          setGithubData(result); 
          
        },
        // On failure from flask microservice
        (error) => {
          // Set state on failure response from the API
          console.log(error);
          // On failure set loading to false to display the error message
          setLoading(false);
          setGithubData([]);
        }
      );
  }, [repository]);
  
  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      {/* Application Header */}
      <AppBar
        position="fixed"
        sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Timeseries Forecasting
          </Typography>
        </Toolbar>
      </AppBar>
      {/* Left drawer of the application */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: {
            width: drawerWidth,
            boxSizing: "border-box",
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: "auto" }}>
          <List>
            {/* Iterate through the repositories list */}
            {repositories.map((repo) => (
              <ListItem
                button
                key={repo.key}
                onClick={() => eventHandler(repo)}
                disabled={loading && repo.value !== repository.value}
              >
                <ListItemButton selected={repo.value === repository.value}>
                  <ListItemText primary={repo.value} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        {/* Render loader component if loading is true else render charts and images */}
        {loading ? (
          <Loader />
        ) : (

           <div>
            
            <Divider
              sx={{ borderBlockWidth: "3px", borderBlockColor: "#FFA500" }}
            />
            {/* Rendering Timeseries Forecasting of Created Issues using Tensorflow and
                Keras LSTM */}

            <div>
              <Typography variant="h5" component="div" gutterBottom>
                Timeseries Forecasting of Created Issues using Tensorflow and
                Keras LSTM based on past 2 years
              </Typography>

              <div>
                <Typography component="h4">
                  Model Loss for Created Issues
                </Typography>
                {/* Render the model loss image for created issues */}
                <img
                  src={githubRepoData?.createdAtImageUrls?.model_loss_image_url}
                  alt={"Model Loss for Created Issues"}
                  loading={"lazy"}
                />
              </div>
              <div>
                <Typography component="h4">
                  LSTM Generated Data for Created Issues
                </Typography>
                {/* Render the LSTM generated image for created issues*/}
                <img
                  src={
                    githubRepoData?.createdAtImageUrls?.lstm_generated_image_url
                  }
                  alt={"LSTM Generated Data for Created Issues"}
                  loading={"lazy"}
                />
              </div>
              <div>
                <Typography component="h4">
                  All Issues Data for Created Issues
                </Typography>
                {/* Render the all issues data image for created issues*/}
                <img
                  src={
                    githubRepoData?.createdAtImageUrls?.all_issues_data_image
                  }
                  alt={"All Issues Data for Created Issues"}
                  loading={"lazy"}
                />
              </div>
            </div>


            <div>
              <Divider
                sx={{ borderBlockWidth: "3px", borderBlockColor: "#FFA500" }}
              />
              <Typography variant="h5" component="div" gutterBottom>
                Timeseries Forecasting of Created Issues using Facebook/Prophet 
                based on past 2 years
              </Typography>

              <div>
                <Typography component="h4">
                  Forecast of Created Issues
                </Typography>
                {/* Render the model loss image Created AT  */}
                <img
                  src={githubRepoData?.fb_createdAtImageUrls?.fbprophet_forecast_url}
                  alt={"Forecast of Created Issues"}
                  loading={"lazy"}
                />
              </div>
              <div>
                <Typography component="h4">
                Forecast Components of Created Issues
                </Typography>
                {/* Render the LSTM generated image Created At */}
                <img
                  src={
                    githubRepoData?.fb_createdAtImageUrls?.fbprophet_forecast_components_url
                  }
                  alt={"Forecast Components of Created Issues"}
                  loading={"lazy"}
                />
              </div>
              
            </div>

            

            <div>
              <Divider
                sx={{ borderBlockWidth: "3px", borderBlockColor: "#FFA500" }}
              />
              <Typography variant="h5" component="div" gutterBottom>
                Timeseries Forecasting of Created Issues using StatsModel 
                based on past 2 years
              </Typography>

              <div>
                <Typography component="h4">
                Observation Graph of Created Issues
                </Typography>
                {/* Render the model loss image Created AT  */}
                <img
                  src={githubRepoData?.stat_createdAtImageUrls?.stats_observation_url}
                  alt={"Observation Graph of Created Issues"}
                  loading={"lazy"}
                />
              </div>
              <div>
                <Typography component="h4">
                Time Series Forecasting of Created Issues
                </Typography>
                {/* Render the LSTM generated image Created At */}
                <img
                  src={
                    githubRepoData?.stat_createdAtImageUrls?.stats_forecast_url
                  }
                  alt={"Time Series Forecasting of Created Issues"}
                  loading={"lazy"}
                />
              </div>
              
            </div>

          </div>
        )}
      </Box>
    </Box>
  );
}