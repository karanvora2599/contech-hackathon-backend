import logging
import time
import traceback  # Ensure traceback is imported
import os
import json
from typing import Optional, Union, List, Any

from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from cerebras.cloud.sdk import Cerebras  # Ensure this SDK is installed

import boto3
import hashlib
import pandas as pd
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
import awswrangler as wr

# Configure logging
LOG_DIR = os.path.abspath("logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, "RDF.log")

# Create handlers
file_handler = TimedRotatingFileHandler(
    filename=log_file_path,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)

console_handler = logging.StreamHandler()

# Create formatters with more context
detailed_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
)

file_handler.setFormatter(detailed_formatter)
console_handler.setFormatter(detailed_formatter)

# Configure root logger
logger = logging.getLogger("RDF")
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all logs
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Pydantic model for parse request
class ParseRequest(BaseModel):
    document_text: str = Field(..., example="Your document text here.")
    system_prompt: Optional[str] = Field(
        None,
        example="You are an assistant that extracts key information from documents."
    )
    temperature: Optional[float] = Field(1.0, example=0.7)
    max_tokens: Optional[int] = Field(2048, example=1500)
    top_p: Optional[float] = Field(0.9, example=0.8)
    stream: Optional[bool] = Field(False, example=True)
    response_format: Optional[dict] = Field({"type": "json_object"}, example={"type": "json_object"})
    stop: Optional[Union[str, List[str]]] = Field(None, example=["\n"])

# Pydantic model for parse response
class ParseResponse(BaseModel):
    status: str
    details: Optional[dict] = None
    
# ===========================
# Pydantic Models for Query
# ===========================
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: Any

# Load API Key from environment
CEREBRAS_API_KEY = os.getenv(
    "CEREBRAS_API_KEY"  # Replace with your actual API key or set as environment variable
)

credentials = {
    'AccessKeyId': 'YOUR_ACCESS_KEY_ID',           # Replace with your Access Key ID
    'SecretAccessKey': 'YOUR_SECRET_ACCESS_KEY',   # Replace with your Secret Access Key
    'Token': 'YOUR_SESSION_TOKEN',                 # Replace with your Session Token if using temporary credentials
    'Expiration': '2025-01-25T21:22:53Z',          # Example expiration timestamp
    'Code': 'Success',
    'Message': None
}

# Define prompts or other constants if needed
class Prompts:
    DOCUMENT_SYSTEM_PROMPT = "You are a helpful assistant that parses documents into structured JSON."
    
# ===========================
# AWS Session Creation
# ===========================

def create_session_with_credentials(credentials: dict, region: str = "us-east-1"):
    """
    Create a boto3 session using provided credentials.
    """
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials.get('Token'),
        region_name=region
    )
    
# ===========================
# Utility Functions
# ===========================

def get_aws_auth_token(session):
    """
    Retrieve AWS authentication token details from the session.
    """
    credentials = session.get_credentials().get_frozen_credentials()
    region = session.region_name or "us-east-1"  # Default to us-east-1 if region is None
    service_name = "execute-api"  # Correct service name for Neptune queries
    return {
        "credentials": Credentials(credentials.access_key, credentials.secret_key, credentials.token),
        "service_name": service_name,
        "region": region
    }

def get_account_hash_from_account_id(account_id: str):
    """
    Generate an MD5 hash from the AWS account ID.
    """
    return hashlib.md5(account_id.encode("utf-8")).hexdigest()

def get_account_hash_from_session(session):
    """
    Retrieve the AWS account ID from the session and generate its hash.
    """
    sts_client = session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    return get_account_hash_from_account_id(account_id)

def get_intelligence_base_url_from_session(session):
    """
    Construct the base URL for the intelligence service based on the account hash.
    """
    account_hash = get_account_hash_from_session(session)
    return f"https://intelligence.{account_hash}.gryps.io"

# ===========================
# Neptune Query Handler
# ===========================

class NeptuneQueryHandler:
    def __init__(self, session):
        """
        Initialize the NeptuneQueryHandler with AWS session details.
        """
        auth_details = get_aws_auth_token(session)
        self.credentials = auth_details["credentials"]
        self.service_name = auth_details["service_name"]
        self.region = auth_details["region"]
        self.base_url = get_intelligence_base_url_from_session(session)

    def query(self, query, output_format="json"):
        """
        Execute a SPARQL query against Amazon Neptune and return the results.
        """
        payload = json.dumps({"query": query})
        headers = {"Content-Type": "application/json"}
        aws_request = AWSRequest(
            method="POST",
            url=f"{self.base_url}/sparql",
            data=payload,
            headers=headers
        )
        SigV4Auth(self.credentials, self.service_name, self.region).add_auth(aws_request)
        signed_headers = dict(aws_request.headers)

        response = requests.post(
            aws_request.url,
            headers=signed_headers,
            data=payload,
            timeout=180,
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")  # Debugging
            response.raise_for_status()

        if output_format == "json":
            return response.json()
        elif output_format == "pandas":
            return self._convert_to_pandas(response.json())
        else:
            raise ValueError("Invalid output format. Please use 'json' or 'pandas'.")

    def _convert_to_pandas(self, response):
        """
        Convert the JSON response from Neptune into a pandas DataFrame.
        """
        lst = []
        for item in response.get("results", {}).get("bindings", []):
            d = {k: v.get("value") for k, v in item.items()}
            lst.append(d)
        if lst:
            df = pd.json_normalize(lst)
            return df
        return pd.DataFrame()
    
session = create_session_with_credentials(credentials)
neptune_client = NeptuneQueryHandler(session=session)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application Startup Event
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing resources.")

# Application Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown: Releasing resources.")

# Middleware to log request and response details
@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Request received from {client_host}: {request.method} {request.url}")

    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request error: {str(e)}", exc_info=True)
        raise

    process_time = time.time() - start_time
    response_log = f"Response: {response.status_code} | Processing time: {process_time:.2f}s"

    if response.status_code >= 500:
        logger.error(response_log)
    elif response.status_code >= 400:
        logger.warning(response_log)
    else:
        logger.info(response_log)

    return response

# Global exception handler for HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        f"HTTPException: {exc.status_code} - {exc.detail} - Path: {request.url.path}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

# Global exception handler
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {exc} - Path: {request.url.path}"
    )
    logger.debug(traceback.format_exc())  # Now, traceback is defined
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )

# Health check endpoint with detailed logging
@app.get("/health")
async def health_check():
    logger.info("Starting health check")
    try:
        # Add actual health checks here (e.g., database connection)
        logger.debug("Performing health check validations")
        return {"status": "healthy", "details": "All systems operational"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Service unavailable",
        ) from e
    finally:
        logger.info("Completed health check")

def LLM_Text_Parse(
    document_text: str,
    api_key: str,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    top_p: float = 1.0,
    stream: bool = False,
    response_format: dict = {"type": "json_object"},
    stop: Optional[Union[str, List[str]]] = None
) -> Optional[dict]:
    """
    Parses the extracted text using Cerebras' LLM and returns structured JSON.
    """
    logger.info("Starting parsing of extracted text with Cerebras' LLM.")
    try:
        client = Cerebras(api_key=api_key)
        
        # Use the provided system prompt or default if not provided
        prompt = system_prompt if system_prompt else Prompts.DOCUMENT_SYSTEM_PROMPT.strip()
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": document_text
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            response_format=response_format,
            stop=stop,
        )

        # Gather and return the output
        parsed_content = completion.choices[0].message.content
        try:
            JSONOutput = json.loads(parsed_content)
            logger.info("Successfully parsed text with Cerebras' LLM.")
            return JSONOutput
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Cerebras' response: {e}")
            return None

    except Exception as e:
        logger.error(f"Error generating system prompt for Cerebras' LLM: {e}", exc_info=True)
        return None

# New endpoint to parse text using LLM with customizable parameters
@app.post("/parse", response_model=ParseResponse)
async def parse_document(request: ParseRequest):
    logger.info("Received request to parse document.")
    logger.debug(f"Document text received: {request.document_text[:100]}...")  # Log first 100 chars

    try:
        parsed_result = LLM_Text_Parse(
            document_text=request.document_text,
            api_key=CEREBRAS_API_KEY,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream,
            response_format=request.response_format,
            stop=request.stop
        )
        if parsed_result is None:
            logger.error("Parsing failed due to invalid JSON response.")
            raise HTTPException(status_code=500, detail="Failed to parse document.")

        logger.info("Document parsed successfully.")
        return ParseResponse(status="success", details=parsed_result)

    except HTTPException as he:
        logger.warning(f"HTTPException during parsing: {he.detail}")
        raise he  # Re-raise to be handled by global handlers

    except Exception as e:
        logger.error(f"Unexpected error during document parsing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during parsing.")
    
@app.post("/query", response_model=QueryResponseModel)
def execute_query(request: QueryRequestModel):
    logger.info("Received Neptune query request.")
    logger.debug(f"Neptune query received: {request.query}")

    try:
        data = neptune_client.query(request.query, output_format="json")
        logger.info("Neptune query executed successfully.")
        return QueryResponseModel(query=request.query, response=data)

    except HTTPException as he:
        # Re-raise HTTPExceptions to be handled by the global exception handler
        logger.warning(f"HTTPException during Neptune query: {he.detail}")
        raise he

    except Exception as e:
        # Log unexpected exceptions and raise a generic HTTPException
        logger.error(f"Unexpected error during Neptune query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during Neptune query.")