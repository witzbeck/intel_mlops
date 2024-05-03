from os import environ
from sys import path

import frontend.constants as const

# Add the project path to the system path
path.append(str(const.PROJECT_PATH))

# Load the environment variables from the .env file
dotenv = const.PROJECT_PATH / ".env"
lines = [
    x.split("=") for x in dotenv.read_text().split("\n") if not x.startswith("#") and x
]

# Update the environment variables
environ.update({x[0]: x[-1] for x in lines if len(x) > 1})
