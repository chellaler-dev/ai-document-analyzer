import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pytesseract
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """Handles document classification using pre-trained models"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load pre-trained zero-shot classification model"""
        try:
            logger.info("Loading classification model...")
            # Using zero-shot classification - no training needed!
            self.model = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Classification model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def classify_document(self, text: str, candidate_labels: list = None):
        """
        Classify document text into categories
        
        Args:
            text: Document text to classify
            candidate_labels: List of possible categories
            
        Returns:
            dict: Classification results with scores
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Default document types if none provided
        if candidate_labels is None:
            candidate_labels = [
                "invoice",
                "contract",
                "receipt",
                "legal document",
                "email",
                "report",
                "resume",
                "letter"
            ]
        
        try:
            # Truncate text if too long (BART has 1024 token limit)
            max_chars = 3000
            if len(text) > max_chars:
                text = text[:max_chars]
            
            result = self.model(text, candidate_labels, multi_label=False)
            
            return {
                "predicted_category": result["labels"][0],
                "confidence": float(result["scores"][0]),
                "all_scores": {
                    label: float(score) 
                    for label, score in zip(result["labels"], result["scores"])
                }
            }
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            raise

class TextExtractor:
    """Handles OCR and text extraction from images/PDFs"""
    
    def __init__(self):
        self.ocr_available = self._check_tesseract()
    
    def _check_tesseract(self):
        """Check if Tesseract is installed"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {str(e)}")
            return False
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            str: Extracted text
        """
        if not self.ocr_available:
            raise RuntimeError("Tesseract OCR is not installed")
        
        try:
            image = Image.open(image_path)
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            logger.info(f"Extracted {len(text)} characters from image")
            return text.strip()
        
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            from PyPDF2 import PdfReader
            
            reader = PdfReader(pdf_path)
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text.strip()
        
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise

class NamedEntityRecognizer:
    """Extract key entities from text (names, dates, amounts, etc.)"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load NER model"""
        try:
            logger.info("Loading NER model...")
            self.model = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("NER model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading NER model: {str(e)}")
            return False
    
    def extract_entities(self, text: str):
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            dict: Extracted entities by type
        """
        if not self.model:
            raise RuntimeError("NER model not loaded")
        
        try:
            # Truncate if too long
            max_chars = 3000
            if len(text) > max_chars:
                text = text[:max_chars]
            
            results = self.model(text)
            
            # Organize entities by type
            entities = {
                "persons": [],
                "organizations": [],
                "locations": [],
                "miscellaneous": []
            }
            
            for entity in results:
                entity_type = entity["entity_group"]
                entity_text = entity["word"]
                confidence = float(entity["score"])
                
                entity_info = {
                    "text": entity_text,
                    "confidence": confidence
                }
                
                if entity_type == "PER":
                    entities["persons"].append(entity_info)
                elif entity_type == "ORG":
                    entities["organizations"].append(entity_info)
                elif entity_type == "LOC":
                    entities["locations"].append(entity_info)
                else:
                    entities["miscellaneous"].append(entity_info)
            
            return entities
        
        except Exception as e:
            logger.error(f"NER error: {str(e)}")
            raise

# Global instances (singleton pattern)
classifier = DocumentClassifier()
text_extractor = TextExtractor()
ner = NamedEntityRecognizer()