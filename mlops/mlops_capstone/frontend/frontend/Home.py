from pathlib import Path

from PIL import Image
from streamlit import (
    caption,
    columns,
    divider,
    header,
    image,
    markdown,
    subheader,
    title,
)

PROJECT_PATH = Path(__file__).parent.parent
ASSETS_PATH = PROJECT_PATH / "assets"


title("The Prototype")
header("Pharmaceutical Manufacturing Business")
markdown("Building a Prototype for the MLOps Certifcation Capstone Project.")

divider()

col1, col2 = columns(2)

with col1:
    subheader("Robotics Maintenance")
    forecasting_image = Image.open(ASSETS_PATH / "robot_arm.png")
    image(forecasting_image)
    caption(
        "Computer vision quality inspection tool to flag and remove bad pills from production line"
    )

with col2:
    subheader("Monitoring Dashboard")
    forecasting_image = Image.open(ASSETS_PATH / "stats.png")
    image(forecasting_image)
    caption(
        "Customer support chatbot based on pre-trained gpt-j large language model"
    )

divider()

markdown("##### Notices & Disclaimers")
caption(
    "Performance varies by use, configuration, and other factors. Learn more on the Performance \
    Index site. Performance results are based on testing as of dates shown in configurations and may not\
        reflect all publicly available updates. See backup for configuration details. No product or component\
            can be absolutely secure. Your costs and results may vary. Intel technologies may require enabled\
                hardware, software, or service activation. © Intel Corporation. Intel, the Intel logo, and other\
                    Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may\
                        be claimed as the property of others."
)
