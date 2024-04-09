from argparse import ArgumentParser
from json import dumps

from requests import get, post

from model.data_model import GenPayload, headers, url_base


if __name__ == "__main__":
    print("URL_BASE: ", url_base)
    parser = ArgumentParser()
    parser.add_argument("--ping", action="store_true", help="Ping the server")
    parser.add_argument(
        "--predict", action="store_true", help="Predict using the model"
    )
    args = parser.parse_args()
    if args.ping:
        request_type = "ping"
        response = get(f"{url_base}/ping")
    elif args.predict:
        print(f"headers: {headers}")
        request_type = "prediction"
        print(f"sending {GenPayload}")
        response = post(
            f"{url_base}/predict", headers=headers, data=dumps(GenPayload)
        )
    else:
        raise ValueError(f"No valid arguments passed | {parser.print_help()}")

    if args.generate or response.status_code == 200:
        print(f"{request_type} request was successful\n{response.json()}")
    else:
        print(f"{request_type} request failed")
        print(response.status_code, response.text)
        exit(1)
