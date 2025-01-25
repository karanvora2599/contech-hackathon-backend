import boto3
import base64
import json
from iamauth import Sigv4Auth
import os
import hashlib
import pandas as pd
import requests
from typing import List, Dict, Union
import awswrangler as wr
from IPython.display import display, JSON
from pprint import pprint


def get_aws_auth_token_from_sso_profile(
    profile_name: str, region_name: str = "us-east-1"
):
    base_session = boto3.Session(profile_name=profile_name, region_name=region_name)
    sigv4 = Sigv4Auth(boto3_session=base_session)
    return sigv4


def get_account_hash_from_account_id(account_id: str):
    return hashlib.md5(account_id.encode("utf-8")).hexdigest()


def get_account_hash_from_sso_profile(
    profile_name: str, region_name: str = "us-east-1"
):
    base_session = boto3.Session(profile_name=profile_name, region_name=region_name)
    sts_client = base_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    return get_account_hash_from_account_id(account_id)


def get_intelligence_base_url_from_sso_profile(
    profile_name: str, region_name: str = "us-east-1"
):
    account_hash = get_account_hash_from_sso_profile(profile_name, region_name)
    return f"https://intelligence.{account_hash}.gryps.io"


class IMSQueryHandler:
    def __init__(self, profile_name: str, region_name: str = "us-east-1"):
        self.base_session = boto3.Session(
            profile_name=profile_name, region_name=region_name
        )

    def list_of_databases(self):
        return wr.catalog.databases(boto3_session=self.base_session)

    def list_of_tables(self, database):
        return wr.catalog.tables(database=database, boto3_session=self.base_session)

    def query(self, query: str, database: str):
        return wr.athena.read_sql_query(
            query, database=database, boto3_session=self.base_session, ctas_approach=False, workgroup="AmazonAthenaLakeFormation"
        )


class NeptuneQueryHandler:
    def __init__(self, profile_name: str):
        self.auth = get_aws_auth_token_from_sso_profile(profile_name)
        self.base_url = get_intelligence_base_url_from_sso_profile(profile_name)

    def query(self, query, output_format="json"):
        payload = json.dumps({"query": query})
        headers = {"Content-Type": "application/json"}
        response = requests.request(
            "POST",
            url=f"{self.base_url}/sparql",
            headers=headers,
            data=payload,
            auth=self.auth,
            timeout=180,
        )

        if output_format == "json":
            return response.json()
        elif output_format == "pandas":
            return self._convert_to_pandas(response.json())
        else:
            print("Invalid output format. Please use 'json' or 'pandas'.")

    def _convert_to_pandas(self, response):

        lst = []
        df = pd.DataFrame()
        for item in response["results"]["bindings"]:
            d = {}
            for k in item.keys():
                d[k] = item[k]["value"]
            lst.append(d)
        if len(lst) > 0:
            df = pd.json_normalize(lst)

        return df