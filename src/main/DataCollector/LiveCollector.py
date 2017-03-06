# Standard Library
# import json
import time
# 3rd Party
import pandas as pd
import numpy as np
import websocket
import requests

# Constants TODO add to config file
web_socket_url_base = 'ws://livestats.proxy.lolesports.com/stats?jwt='
token_issuer = 'http://api.lolesports.com/api/issueToken'


def get_token():
    """Gets a token to use in the livestats websocket"""
    token_json = requests.get(token_issuer)
    return token_json.json()['token']


class WebSocketLogger(object):

    def __init__(self, request_interval=2):
        """request_interval: int or float: time to wait in seconds between requests"""
        self.__get_token()
        self.request_interval = request_interval

    def __get_token(self):
        """Get token for this session"""
        self.token = get_token()

    def logger(self):
        with websocket.create_connection(web_socket_url_base + self.token) as ws:
            try:
                while True:
                    new_row = ws.recv()  # Currently unused
                    time.sleep(self.request_interval)
            except websocket._exceptions.WebSocketConnectionClosedException:  # TODO unsure if correct error
                pass
