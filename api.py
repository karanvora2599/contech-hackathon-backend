import logging
import time
import traceback  # Ensure traceback is imported
import os
import json
from typing import Optional

from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from cerebras.cloud.sdk import Cerebras  # Ensure this SDK is installed
from gryps_utils import NeptuneQueryHandler, IMSQueryHandler  # Import your utility classes

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
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Pydantic model for parse request
class ParseRequest(BaseModel):
    document_text: str = Field(..., example="Your document text here.")

# Pydantic model for parse response
class ParseResponse(BaseModel):
    status: str
    details: Optional[dict] = None
    
# Pydantic model for query request
class QueryRequest(BaseModel):
    query: str = Field(..., example="Your SPARQL query here.")

# Pydantic model for query response
class QueryResponse(BaseModel):
    query: str
    response: Union[dict, list, str]  # Adjust based on expected response types

# Load API Key from environment
CEREBRAS_API_KEY = os.getenv(
    "CEREBRAS_API_KEY"  # Replace with your actual API key or set as environment variable
)

# Define prompts or other constants if needed
class Prompts:
    DOCUMENT_SYSTEM_PROMPT = "You are a helpful assistant that parses documents into structured JSON."

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

def LLM_Text_Parse(document_text: str, api_key: str) -> Optional[dict]:
    """
    Parses the extracted text using Cerebras' LLM and returns structured JSON.
    """
    logger.info("Starting parsing of extracted text with Cerebras' LLM.")
    try:
        client = Cerebras(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {
                    "role": "system",
                    "content": Prompts.DOCUMENT_SYSTEM_PROMPT.strip()
                },
                {
                    "role": "user",
                    "content": document_text
                }
            ],
            temperature=1,
            max_tokens=8192,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
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

# New endpoint to parse text using LLM
@app.post("/parse", response_model=ParseResponse)
async def parse_document(request: ParseRequest):
    logger.info("Received request to parse document.")
    logger.debug(f"Document text received: {request.document_text[:100]}...")  # Log first 100 chars

    try:
        parsed_result = LLM_Text_Parse(document_text=request.document_text, api_key=CEREBRAS_API_KEY)
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

# New endpoint to execute SPARQL queries against Neptune
@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    logger.info("Received request to execute SPARQL query.")
    logger.debug(f"SPARQL query received: {request.query[:100]}...")  # Log first 100 chars

    try:
        # Initialize NeptuneQueryHandler with the appropriate AWS profile
        neptune_client = NeptuneQueryHandler(profile_name="default")  # Adjust profile_name as needed

        # Execute the query
        query_result = neptune_client.query(request.query, output_format="json")

        logger.info("SPARQL query executed successfully.")

        return QueryResponse(query=request.query, response=query_result)

    except HTTPException as he:
        logger.warning(f"HTTPException during SPARQL query execution: {he.detail}")
        raise he  # Re-raise to be handled by global handlers

    except Exception as e:
        logger.error(f"Unexpected error during SPARQL query execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during SPARQL query execution.")

# Entry point
if __name__ == "__main__":
    logger.info("Starting application server")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        logger.critical(f"Server failed to start: {str(e)}", exc_info=True)
        raise