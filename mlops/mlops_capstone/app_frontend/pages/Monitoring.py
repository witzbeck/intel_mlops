from streamlit import title, markdown, image, caption, divider
import requests
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

title("Monitoring Dashboard")
image = Image.open("./assets/stats.png")
image(image)
markdown(
    "###### A simple tool for monitoring the performance of our model. This simple monitoring dashboard will help us track the inference latency and evaluate trends in prediction results."
)

markdown("### Record of Inference Results")
caption("A table containing metadata about each inference request made.")

# Logic for inference metadata table

divider()

markdown("### Chart of Inference Time in Milliseconds (ms) vs Request DateTime Stamps")
caption("A line graph depicting the change inference time over time. ")

# Logic for inference latency line chart

divider()

markdown("### Chart of Predicted Labels vs Request DateTime Stamps")
caption("A plot depicting the change predictions over time. ")

# Logic for predictions over time

divider()

markdown("### Histogram of Results")
caption("A histogram showing the frequency of each prediction label.")

# Logic for predictions histogram
