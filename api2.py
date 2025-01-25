import logging
import time
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os

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

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced request logging middleware
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

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        logger.warning(f"HTTP Exception: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.detail},
        )
    
    logger.critical(f"Unhandled exception: {str(exc)}", exc_info=True)
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

if __name__ == "__main__":
    logger.info("Starting application server")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        logger.critical(f"Server failed to start: {str(e)}", exc_info=True)
        raise