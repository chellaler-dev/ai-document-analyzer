from pathlib import Path
import hashlib
import time
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def get_file_hash(file_path: Path) -> str:
    """Generate MD5 hash of file for caching/deduplication"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def clean_text(text: str) -> str:
    """
    Clean extracted text
    - Remove extra whitespace
    - Remove special characters
    - Normalize line breaks
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Basic cleaning
    text = text.replace("\x00", "")  # Remove null bytes
    
    return text.strip()

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()

def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed"""
    return get_file_extension(filename) in allowed_extensions

def format_confidence(score: float) -> str:
    """Format confidence score as percentage"""
    return f"{score * 100:.2f}%"

class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} took {duration:.2f} seconds")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

def create_response(
    success: bool,
    data: dict = None,
    message: str = "",
    error: Optional[str] = None
) -> dict:
    """
    Create standardized API response
    
    Args:
        success: Whether operation was successful
        data: Response data
        message: Success message
        error: Error message if failed
    
    Returns:
        dict: Standardized response
    """
    response = {
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if success:
        response["message"] = message
        response["data"] = data or {}
    else:
        response["error"] = error or "An error occurred"
        response["data"] = None
    
    return response

def extract_key_info(text: str) -> dict:
    """
    Extract common key information from document text
    (Simple regex-based extraction for demo)
    """
    import re
    
    info = {
        "emails": [],
        "phone_numbers": [],
        "dates": [],
        "amounts": []
    }
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    info["emails"] = re.findall(email_pattern, text)
    
    # Extract phone numbers (simple pattern)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    info["phone_numbers"] = re.findall(phone_pattern, text)
    
    # Extract dates (simple pattern: MM/DD/YYYY or DD-MM-YYYY)
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    info["dates"] = re.findall(date_pattern, text)
    
    # Extract dollar amounts
    amount_pattern = r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?'
    info["amounts"] = re.findall(amount_pattern, text)
    
    return info