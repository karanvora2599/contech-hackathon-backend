import logging
import platform
import traceback
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os

# Configure logging
LOG_DIR = os.path.abspath("logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file_path = os.path.join(LOG_DIR, "RDF.log")

# Create a timed rotating file handler (daily rotation)
file_handler = TimedRotatingFileHandler(
    filename=log_file_path,
    when="midnight",       # Rotate at midnight
    interval=1,            # Every 1 day
    backupCount=7,         # Keep 7 days of logs
    encoding="utf-8"
)

file_handler.setLevel(logging.DEBUG)  # Capture all levels
file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Console shows INFO and above
console_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

logger = logging.getLogger("RDF")
logger.setLevel(logging.DEBUG)  # Capture all levels
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize FastAPI app
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
    logger.info(f"Incoming request: {request.method} {request.url}")
    # Optionally log request headers
    logger.debug(f"Request headers: {dict(request.headers)}")
    
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception during request: {e}")
        logger.debug(traceback.format_exc())
        raise e  # Re-raise the exception after logging
    
    logger.info(f"Response status: {response.status_code}")
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

# Example endpoint with additional logging and error handling
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    logger.debug(f"Processing request for item_id: {item_id}, query: {q}")
    if item_id < 0:
        logger.warning(f"Invalid item_id received: {item_id}")
        raise HTTPException(status_code=400, detail="Item ID must be non-negative")
    # Simulate processing
    item = {"item_id": item_id, "q": q}
    logger.info(f"Returning item: {item}")
    return item

# Global exception handler for HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        f"HTTPException: {exc.status_code} - {exc.detail} - Path: {request.url.path}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {exc} - Path: {request.url.path}"
    )
    logger.debug(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")