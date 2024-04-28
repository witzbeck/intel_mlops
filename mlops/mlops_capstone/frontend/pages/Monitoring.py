from pathlib import Path
from PIL import Image
from streamlit import caption, divider, image, markdown, title

PROJECT_PATH = Path(__file__).parent.parent
ASSETS_PATH = PROJECT_PATH / "assets"

title("Monitoring Dashboard")
stats_image = Image.open(ASSETS_PATH / "stats.png")
image(stats_image)
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
