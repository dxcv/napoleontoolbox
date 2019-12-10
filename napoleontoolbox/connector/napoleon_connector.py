#!/usr/bin/env python3
# coding: utf-8

""" Object to connect to the NaPoleonX database. """

# Built-in packages
import requests
import json

# Third party packages
import pandas as pd

# Local packages

__all__ = ['NaPoleonXConnector']


class NaPoleonXConnector(object):
    """ Object to connect to the NaPoleonX database.

    Parameters
    ----------
    username, password, client_id, client_secret : str
        Identifier to request the API.

    Attributes
    ----------
    token : str
        token to identify client.

    Methods
    -------
    get_data
    get_dataframe

    """

    url_auth = 'https://api.napoleonx.ai/auth-service/oauth/token'
    url = 'https://api.napoleonx.ai/quote-service/v1/eod-quote/filter'

    data = {
        'grant_type': 'password',
        # 'username': 'arthur',
        # 'password': 'arthurBernard-coiffeur!digital_with-machine&learning',
        'scope': 'read write'
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
    }

    # client_id = 'postman'
    # client_secret = 'bKTn8a2vp4'

    def __init__(self, username, password, client_id, client_secret):
        """ Initialize object. """
        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)

        self.data.update({'username': username, 'password': password})

        response = requests.post(
            self.url_auth,
            data=self.data,
            headers=self.headers,
            auth=auth
        ).json()

        # Set token
        self.token = response['access_token']
        self.token_type = response['token_type']
        self._set_header()

    def _set_header(self):
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': "{} {}".format(
                self.token_type,
                self.token
            ),
            'cache-control': "no-cache"
        }

    def _post_request(self, url, **kwargs):
        data = json.dumps(kwargs)
        return requests.post(url, data=data, headers=self.headers).json()

    def _get_request(self, url, **kwargs):
        data = json.dumps(kwargs)
        return requests.get(url, data=data, headers=self.headers).json()

    def get_data(self, productCodes, minDate, maxDate):
        """ Request data.

        Parameters
        ----------
        productCodes : list of str
            List of codes of assets.
        minDate, maxDate : str
            Respectively the first and last date of the data.
        kwargs
            Key word arguments for requests the database.

        Returns
        -------
        dict
            Data.

        """
        return self._post_request(
            self.url,
            productCodes=productCodes,
            minDate=minDate,
            maxDate=maxDate
        )

    def _set_dataframe(self, data, assets, keep=['close']):
        """ Clean, sort and set data into the dataframe.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe of not ordered data.
        assets : list of str
            List of code of assets.
        keep : list of str
            List of columns to keep into the dataframe.

        Returns
        -------
        pandas.DataFrame
            Where each column is an asset price and each row is a date.

        """
        columns, asset_col = {}, []
        for asset in assets:
            if len(keep) > 1:
                columns[asset] = {k: k + '_' + asset for k in keep}
                asset_col += [k + '_' + asset for k in keep]

            else:
                columns[asset] = {keep[0]: asset}
                asset_col += [asset]

        df = pd.DataFrame(
            columns=asset_col,
            index=sorted(data.date.drop_duplicates()),
        )

        for asset in assets:
            sub_df = (data.loc[data.productCode == asset, keep + ['date']]
                          .set_index('date')
                          .sort_index()
                          .rename(columns=columns[asset]))
            asset_col = list(columns[asset].values())
            df.loc[sub_df.index, asset_col] = sub_df.loc[:, asset_col]

        return df

    def get_dataframe(self, productCodes, minDate, maxDate, keep=['close'],
                      process_na=None):
        """ Request data, clean, sort, and set into pandas dataframe.

        Parameters
        ----------
        productCodes : list of str
            List of codes of assets.
        minDate, maxDate : str
            Respectively the first and last date of the data.
        keep : list of str
            List of columns to keep into the dataframe.
        process_na : {None, 'fill', 'drop'}
            - If None don't return dataframe with nan values.
            - If 'fill' replace nan by the last observation or by the next one.
            - If 'drop' drop the nan values.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe.

        """
        data = pd.DataFrame(self.get_data(productCodes, minDate, maxDate))
        df = self._set_dataframe(data, assets=productCodes, keep=keep)
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

        if process_na == 'fill':

            return df.fillna(method='ffill').fillna(method='bfill')

        elif process_na == 'drop':

            return df.dropna()

        return df
