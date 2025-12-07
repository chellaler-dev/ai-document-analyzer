from prometheus_client import Counter, Histogram, Gauge, Info
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# METRICS DEFINITIONS
# ============================================================================

# Request Counters
requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status']
)

# ML Model Metrics
ml_predictions_total = Counter(
    'ml_predictions_total',
    'Total number of ML predictions',
    ['model_type', 'status']
)

ml_inference_duration = Histogram(
    'ml_inference_duration_seconds',
    'ML inference duration in seconds',
    ['model_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

ml_confidence_score = Histogram(
    'ml_confidence_score',
    'ML prediction confidence scores',
    ['model_type'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

low_confidence_predictions = Counter(
    'ml_low_confidence_predictions_total',
    'Number of predictions with confidence below threshold',
    ['model_type', 'threshold']
)

# Cache Metrics
cache_operations = Counter(
    'cache_operations_total',
    'Total number of cache operations',
    ['operation', 'status']  # operation: get/set/delete, status: hit/miss/error
)

# Document Processing Metrics
documents_processed = Counter(
    'documents_processed_total',
    'Total number of documents processed',
    ['doc_type', 'status']  # doc_type: pdf/image, status: success/error
)

ocr_operations = Counter(
    'ocr_operations_total',
    'Total number of OCR operations',
    ['status']
)

# Resource Metrics
model_memory_usage = Gauge(
    'model_memory_usage_bytes',
    'Memory usage of loaded models',
    ['model_name']
)

active_requests = Gauge(
    'api_active_requests',
    'Number of requests currently being processed'
)

# API Info
api_info = Info(
    'api',
    'API information'
)

# ============================================================================
# MONITORING DECORATORS
# ============================================================================

def track_inference_time(model_type: str):
    """
    Decorator to track ML inference time
    
    Usage:
        @track_inference_time("classifier")
        def classify_document(text):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics
                ml_inference_duration.labels(model_type=model_type).observe(duration)
                ml_predictions_total.labels(
                    model_type=model_type,
                    status="success"
                ).inc()
                
                logger.info(f"{model_type} inference took {duration:.2f}s")
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                ml_predictions_total.labels(
                    model_type=model_type,
                    status="error"
                ).inc()
                logger.error(f"{model_type} inference failed after {duration:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator

def track_confidence(model_type: str, threshold: float = 0.7):
    """
    Decorator to track prediction confidence scores
    
    Usage:
        @track_confidence("classifier", threshold=0.7)
        def classify_document(text):
            return {"confidence": 0.95, ...}
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Extract confidence score
            confidence = result.get("confidence", 0.0)
            
            # Record confidence distribution
            ml_confidence_score.labels(model_type=model_type).observe(confidence)
            
            # Track low confidence predictions
            if confidence < threshold:
                low_confidence_predictions.labels(
                    model_type=model_type,
                    threshold=str(threshold)
                ).inc()
                logger.warning(
                    f"Low confidence prediction: {model_type} = {confidence:.2f}"
                )
            
            return result
        
        return wrapper
    return decorator

class RequestTracker:
    """Context manager to track active requests"""
    
    def __enter__(self):
        active_requests.inc()
        return self
    
    def __exit__(self, *args):
        active_requests.dec()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_cache_operation(operation: str, status: str):
    """
    Record cache operation
    
    Args:
        operation: get, set, delete
        status: hit, miss, error
    """
    cache_operations.labels(operation=operation, status=status).inc()

def record_document_processed(doc_type: str, status: str):
    """
    Record document processing
    
    Args:
        doc_type: pdf, image
        status: success, error
    """
    documents_processed.labels(doc_type=doc_type, status=status).inc()

def record_ocr_operation(status: str):
    """
    Record OCR operation
    
    Args:
        status: success, error
    """
    ocr_operations.labels(status=status).inc()

def set_model_memory(model_name: str, memory_bytes: int):
    """
    Set model memory usage
    
    Args:
        model_name: Name of the model
        memory_bytes: Memory usage in bytes
    """
    model_memory_usage.labels(model_name=model_name).set(memory_bytes)

def initialize_api_info(version: str, environment: str = "production"):
    """
    Initialize API information
    
    Args:
        version: API version
        environment: Deployment environment
    """
    api_info.info({
        'version': version,
        'environment': environment
    })

# ============================================================================
# METRICS COLLECTION
# ============================================================================

def get_metrics_summary() -> dict:
    """
    Get summary of key metrics for API response
    
    Returns:
        dict: Metrics summary
    """
    return {
        "active_requests": active_requests._value.get(),
        "total_predictions": {
            "classifier": ml_predictions_total.labels(
                model_type="classifier",
                status="success"
            )._value.get(),
            "ner": ml_predictions_total.labels(
                model_type="ner",
                status="success"
            )._value.get()
        },
        "cache": {
            "hits": cache_operations.labels(
                operation="get",
                status="hit"
            )._value.get(),
            "misses": cache_operations.labels(
                operation="get",
                status="miss"
            )._value.get()
        }
    }