from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import uvicorn
from pathlib import Path
import shutil
from typing import List, Optional
import logging

# Import our custom modules
from models import classifier, text_extractor, ner
from utils import (
    get_file_hash, clean_text, get_file_extension,
    Timer, create_response, extract_key_info
)
from cache import cache
from monitoring import (
    RequestTracker, record_document_processed,
    get_metrics_summary, initialize_api_info, active_requests
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Document Intelligence API",
    description="AI-powered document classification, OCR, and entity extraction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Global state
models_loaded = False

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "AI Document Intelligence API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "classify": "/classify - Classify document type",
            "extract": "/extract - Extract text from document",
            "analyze": "/analyze - Full document analysis",
            "entities": "/entities - Extract named entities"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    cache_stats = cache.get_stats()
    return {
        "status": "healthy",
        "api": "operational",
        "models_loaded": models_loaded,
        "ocr_available": text_extractor.ocr_available,
        "cache": cache_stats
    }

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return cache.get_stats()

@app.post("/cache/clear")
async def clear_cache():
    """Clear all cache (admin endpoint)"""
    success = cache.clear()
    return {
        "success": success,
        "message": "Cache cleared" if success else "Cache not available"
    }

@app.get("/metrics/summary")
async def metrics_summary():
    """Get metrics summary"""
    return get_metrics_summary()

@app.post("/classify")
async def classify_document(
    file: UploadFile = File(...),
    categories: Optional[str] = Form(None)
):
    """
    Classify uploaded document into categories
    
    - **file**: Document file (image or PDF)
    - **categories**: Optional comma-separated list of categories
    """
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    file_path = None
    try:
        with Timer("Document classification"):
            # Validate file
            ext = get_file_extension(file.filename)
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported. Use: {', '.join(ALLOWED_EXTENSIONS)}"
                )
            
            # Save file
            file_path = UPLOAD_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract text based on file type
            if ext == ".pdf":
                text = text_extractor.extract_text_from_pdf(file_path)
            else:
                text = text_extractor.extract_text_from_image(file_path)
            
            if not text:
                raise HTTPException(status_code=400, detail="No text found in document")
            
            # Clean text
            text = clean_text(text)
            
            # Parse categories if provided
            candidate_labels = None
            if categories:
                candidate_labels = [cat.strip() for cat in categories.split(",")]
            
            # Classify
            classification_result = classifier.classify_document(text, candidate_labels)
            
            return create_response(
                success=True,
                message="Document classified successfully",
                data={
                    "filename": file.filename,
                    "classification": classification_result,
                    "text_length": len(text),
                    "text_preview": text[:200] + "..." if len(text) > 200 else text
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if file_path and file_path.exists():
            file_path.unlink()

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded document (OCR for images, text extraction for PDFs)
    
    - **file**: Document file (image or PDF)
    """
    file_path = None
    try:
        with Timer("Text extraction"):
            # Validate file
            ext = get_file_extension(file.filename)
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported. Use: {', '.join(ALLOWED_EXTENSIONS)}"
                )
            
            # Save file
            file_path = UPLOAD_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract text
            if ext == ".pdf":
                text = text_extractor.extract_text_from_pdf(file_path)
            else:
                text = text_extractor.extract_text_from_image(file_path)
            
            if not text:
                raise HTTPException(status_code=400, detail="No text found in document")
            
            # Clean text
            text = clean_text(text)
            
            # Extract key information
            key_info = extract_key_info(text)
            
            return create_response(
                success=True,
                message="Text extracted successfully",
                data={
                    "filename": file.filename,
                    "text": text,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "key_info": key_info
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if file_path and file_path.exists():
            file_path.unlink()

@app.post("/entities")
async def extract_entities(file: UploadFile = File(...)):
    """
    Extract named entities (people, organizations, locations) from document
    
    - **file**: Document file (image or PDF)
    """
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    file_path = None
    try:
        with Timer("Entity extraction"):
            # Validate file
            ext = get_file_extension(file.filename)
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported. Use: {', '.join(ALLOWED_EXTENSIONS)}"
                )
            
            # Save file
            file_path = UPLOAD_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract text
            if ext == ".pdf":
                text = text_extractor.extract_text_from_pdf(file_path)
            else:
                text = text_extractor.extract_text_from_image(file_path)
            
            if not text:
                raise HTTPException(status_code=400, detail="No text found in document")
            
            # Clean text
            text = clean_text(text)
            
            # Extract entities
            entities = ner.extract_entities(text)
            
            return create_response(
                success=True,
                message="Entities extracted successfully",
                data={
                    "filename": file.filename,
                    "entities": entities,
                    "text_preview": text[:200] + "..." if len(text) > 200 else text
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if file_path and file_path.exists():
            file_path.unlink()

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    categories: Optional[str] = Form(None)
):
    """
    Complete document analysis: classification + extraction + entities
    
    - **file**: Document file (image or PDF)
    - **categories**: Optional comma-separated list of categories
    """
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    file_path = None
    try:
        with Timer("Full document analysis"):
            # Validate file
            ext = get_file_extension(file.filename)
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported. Use: {', '.join(ALLOWED_EXTENSIONS)}"
                )
            
            # Save file
            file_path = UPLOAD_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Check cache first
            cache_key = cache.generate_key(file_path, "analyze")
            cached_result = cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached analysis result")
                return cached_result
            
            # Extract text
            if ext == ".pdf":
                text = text_extractor.extract_text_from_pdf(file_path)
            else:
                text = text_extractor.extract_text_from_image(file_path)
            
            if not text:
                raise HTTPException(status_code=400, detail="No text found in document")
            
            # Clean text
            text = clean_text(text)
            
            # Parse categories
            candidate_labels = None
            if categories:
                candidate_labels = [cat.strip() for cat in categories.split(",")]
            
            # Run all analyses
            classification = classifier.classify_document(text, candidate_labels)
            entities = ner.extract_entities(text)
            key_info = extract_key_info(text)
            
            response = create_response(
                success=True,
                message="Document analyzed successfully",
                data={
                    "filename": file.filename,
                    "classification": classification,
                    "entities": entities,
                    "key_info": key_info,
                    "text_statistics": {
                        "character_count": len(text),
                        "word_count": len(text.split()),
                        "line_count": text.count('\n') + 1
                    },
                    "text_preview": text[:300] + "..." if len(text) > 300 else text,
                    "cached": False
                }
            )
            
            # Cache the result
            cache.set(cache_key, response, ttl=3600)
            
            return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if file_path and file_path.exists():
            file_path.unlink()

@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    global models_loaded
    logger.info("=" * 50)
    logger.info("Starting AI Document Intelligence API...")
    logger.info("=" * 50)
    
    # Initialize monitoring
    initialize_api_info(version="1.0.0", environment="development")
    
    try:
        # Load models
        logger.info("Loading ML models (this may take a minute)...")
        
        classifier_loaded = classifier.load_model()
        ner_loaded = ner.load_model()
        
        if classifier_loaded and ner_loaded:
            models_loaded = True
            logger.info("✓ All models loaded successfully!")
        else:
            logger.error("✗ Some models failed to load")
        
        logger.info(f"✓ OCR available: {text_extractor.ocr_available}")
        logger.info(f"✓ Upload directory ready: {UPLOAD_DIR.absolute()}")
        logger.info("=" * 50)
        logger.info("API is ready! Visit http://localhost:8000/docs")
        logger.info("Metrics available at: http://localhost:8000/metrics")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        models_loaded = False

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    
    # Clean up uploads directory
    for file in UPLOAD_DIR.glob("*"):
        if file.is_file():
            file.unlink()
    
    logger.info("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )