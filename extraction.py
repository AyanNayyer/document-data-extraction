import os
import io
import json
import re
import time
import logging
import gc
import multiprocessing
from typing import List, Dict, Any, Iterator, Tuple, Optional
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np

# Set multiprocessing method for macOS compatibility with PaddleOCR
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Set up logging first
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# OCR libraries
try:
    import pytesseract  # pip install pytesseract
except Exception:
    pytesseract = None

try:
    from easyocr import Reader  # pip install easyocr
    EASY_OCR_LANG = ['en']  # add more if needed
    easyocr_reader = Reader(EASY_OCR_LANG, gpu=False)
except Exception:
    easyocr_reader = None

# PaddleOCR for better handwritten text recognition (re-enabled with safe single-page processing)
paddle_ocr = None
paddle_ocr_initialized = False
logging.info("✓ OCR engines initialized: Tesseract + EasyOCR + PaddleOCR (process-isolated)")

def init_paddleocr():
    """Initialize PaddleOCR instance once"""
    global paddle_ocr, paddle_ocr_initialized
    if not paddle_ocr_initialized:
        try:
            from paddleocr import PaddleOCR
            paddle_ocr = PaddleOCR(lang='en')
            paddle_ocr_initialized = True
            logging.info("PaddleOCR initialized successfully")
        except Exception as e:
            paddle_ocr = None
            paddle_ocr_initialized = True
            logging.warning(f"PaddleOCR initialization failed: {e}")
    return paddle_ocr

# Optional OpenCV for advanced preprocessing
try:
    import cv2  # pip install opencv-python
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Install with: pip install opencv-python")

# Optional dependencies
try:
    from pdf2image import convert_from_path  # pip install pdf2image pypdfium2
except Exception:
    convert_from_path = None

# Gemini LLM client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

generative_model = None
try:
    import google.generativeai as genai  # pip install google-generativeai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        generative_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logging.info(f"✓ LLM initialized: {GEMINI_MODEL_NAME}")
    else:
        logging.warning("⚠ GEMINI_API_KEY not found - LLM extraction disabled")
except Exception as e:
    logging.warning(f"⚠ Failed to initialize LLM: {e}")
    generative_model = None

SUPPORTED_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp')

def pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

def render_pdf_pages(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    if convert_from_path is None:
        raise RuntimeError("pdf2image not installed.")
    return convert_from_path(pdf_path, dpi=dpi)

def load_documents(document_folder: str) -> Iterator[Tuple[str, int, bytes]]:
    for fname in sorted(os.listdir(document_folder)):
        fpath = os.path.join(document_folder, fname)
        if os.path.isdir(fpath):
            continue
        lower = fname.lower()
        if lower.endswith(SUPPORTED_IMAGE_EXTS):
            with Image.open(fpath) as img:
                img = img.convert('RGB')
                yield fname, 1, pil_to_png_bytes(img)
        elif lower.endswith('.pdf'):
            try:
                pages = render_pdf_pages(fpath, dpi=300)
                for idx, page in enumerate(pages, start=1):
                    yield f"{fname}#page{idx}", idx, pil_to_png_bytes(page)
            except Exception as e:
                logging.exception(f"Failed to render PDF {fname}: {e}")

def preprocess_versions(image: Image.Image) -> List[Image.Image]:
    # Various preprocessing strategies for OCR robustness with handwriting focus
    versions = []
    
    # Original grayscale
    versions.append(ImageOps.grayscale(image))
    
    # Contrast enhanced
    versions.append(ImageEnhance.Contrast(ImageOps.grayscale(image)).enhance(1.7))
    
    # Adaptive thresholding (better for uneven lighting)
    if OPENCV_AVAILABLE:
        try:
            arr = np.array(ImageOps.grayscale(image))
            # Adaptive threshold for better handwriting
            adaptive_thresh = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            versions.append(Image.fromarray(adaptive_thresh))
            
            # Otsu thresholding
            _, otsu_thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            versions.append(Image.fromarray(otsu_thresh))
            
            # Morphological operations for handwriting
            kernel = np.ones((2,2), np.uint8)
            # Dilation to thicken thin strokes
            dilated = cv2.dilate(otsu_thresh, kernel, iterations=1)
            versions.append(Image.fromarray(dilated))
            
            # Opening to remove noise
            opened = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
            versions.append(Image.fromarray(opened))
            
            # Deskewing for tilted text
            coords = np.column_stack(np.where(otsu_thresh > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                if abs(angle) > 0.5:  # Only deskew if angle is significant
                    (h, w) = otsu_thresh.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    deskewed = cv2.warpAffine(otsu_thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    versions.append(Image.fromarray(deskewed))
                    
        except Exception as e:
            logging.warning(f"OpenCV preprocessing failed: {e}")
    else:
        # Fallback without OpenCV
        arr = np.array(ImageOps.grayscale(image))
        # Simple thresholding without OpenCV
        threshold = 128
        binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
        versions.append(Image.fromarray(binary))
    
    # Sharpened with different parameters
    versions.append(ImageOps.grayscale(image).filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2)))
    versions.append(ImageOps.grayscale(image).filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3)))
    
    # Inverted
    versions.append(ImageOps.invert(ImageOps.grayscale(image)))
    
    # Brightness enhanced
    versions.append(ImageEnhance.Brightness(ImageOps.grayscale(image)).enhance(1.2))
    
    return versions

def paddleocr_worker(page_image_bytes: bytes) -> str:
    """
    Worker function that runs in a separate process for PaddleOCR isolation
    """
    import numpy as np
    from PIL import Image
    import io
    import gc
    import warnings
    
    # Suppress warnings to reduce noise
    warnings.filterwarnings('ignore')
    
    paddle_ocr = None
    try:
        from paddleocr import PaddleOCR
        # Initialize with minimal options for stability
        paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
        
        # Open and process image
        with Image.open(io.BytesIO(page_image_bytes)) as img:
            # Resize if too large to reduce resource consumption
            max_dimension = 1500
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to proper format for PaddleOCR
            img_array = np.array(img)
            
            # PaddleOCR requires 3D array (H, W, C) - convert grayscale to RGB if needed
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # Single channel
                img_array = np.repeat(img_array, 3, axis=2)  # Convert to RGB
            
            # Run OCR using the correct API without deprecated parameters
            result = paddle_ocr.ocr(img_array)
            
            if result and len(result) > 0 and result[0]:
                ocr_result = result[0]
                
                # Handle new PaddleOCR API - OCRResult object with rec_texts as dictionary key
                if hasattr(ocr_result, 'keys') and 'rec_texts' in ocr_result:
                    texts = ocr_result['rec_texts']
                    if texts:
                        combined_text = " ".join(str(text) for text in texts if text and str(text).strip())
                        if combined_text.strip():
                            return combined_text.strip()
                
                # Fallback: check if it has rec_texts as attribute
                elif hasattr(ocr_result, 'rec_texts'):
                    texts = ocr_result.rec_texts
                    if texts:
                        combined_text = " ".join(str(text) for text in texts if text and str(text).strip())
                        if combined_text.strip():
                            return combined_text.strip()
                
                # Fallback: try traditional format (for compatibility)
                elif isinstance(ocr_result, list):
                    texts = []
                    for line in ocr_result:
                        if len(line) >= 2:
                            # line[0] is bbox, line[1] is (text, confidence)
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                                text = text_info[0]
                            else:
                                text = str(text_info)
                            texts.append(text)
                    
                    combined_text = " ".join(texts)
                    if combined_text.strip():
                        return combined_text.strip()
            
            return ""
            
    except Exception as e:
        return f"PaddleOCR worker error: {e}"
    finally:
        # Clean up PaddleOCR resources explicitly
        if paddle_ocr is not None:
            try:
                del paddle_ocr
            except:
                pass
        
        # Force garbage collection
        gc.collect()

def run_paddleocr_separate(page_image_bytes: bytes) -> str:
    """
    Run PaddleOCR in a completely separate process to avoid segmentation faults
    """
    pool = None
    try:
        # Use spawn method for complete process isolation
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(1)
        
        # Use apply_async with timeout to prevent hanging
        async_result = pool.apply_async(paddleocr_worker, (page_image_bytes,))
        result = async_result.get(timeout=30)  # 30 second timeout
        
        if result and not result.startswith("PaddleOCR worker error:"):
            return result
        else:
            logging.debug(f"PaddleOCR failed: {result}")
            return ""
            
    except multiprocessing.TimeoutError:
        logging.debug("PaddleOCR process timed out")
        return ""
    except Exception as e:
        logging.debug(f"PaddleOCR process failed: {e}")
        return ""
    finally:
        # Ensure pool is properly terminated
        if pool is not None:
            try:
                # First try graceful shutdown
                pool.close()  # No more tasks
                pool.join(timeout=3)  # Wait for completion
                
                # Then force termination if needed
                if hasattr(pool, '_pool') and pool._pool:
                    for p in pool._pool:
                        if p.is_alive():
                            p.terminate()
                            p.join(timeout=1)
                
                # Final termination
                pool.terminate()
                
            except Exception as cleanup_error:
                logging.debug(f"Pool cleanup error: {cleanup_error}")
        
        # Force garbage collection in main process too
        gc.collect()

def paddleocr_single_page(image_bytes: bytes) -> str:
    """
    Use PaddleOCR with complete process isolation to avoid segmentation faults
    """
    return run_paddleocr_separate(image_bytes)

def ensemble_ocr(image_bytes: bytes, lang: str = 'en') -> str:
    # Combine Tesseract and EasyOCR on multiple preprocessed versions
    final_texts = []
    with Image.open(io.BytesIO(image_bytes)) as img:
        versions = preprocess_versions(img)
        
        # Try PaddleOCR first (now using process isolation to avoid segfaults)
        use_paddleocr = os.getenv("USE_PADDLEOCR", "1") == "1"  # Enabled with process isolation
        if use_paddleocr:
            paddle_result = paddleocr_single_page(image_bytes)
            if paddle_result:
                return paddle_result  # Return immediately on success
        
        # Comprehensive Tesseract OCR - maximize quality over speed
        if pytesseract:
            tesseract_lang = 'eng' if lang == 'en' else lang
            
            # Use all preprocessing versions for maximum accuracy
            for v in versions:
                try:
                    # Use comprehensive PSM modes for medical documents
                    psm_modes = ['6', '7', '8', '11', '13']  # All effective modes
                    for psm in psm_modes:
                        # Try multiple OCR engine modes for robustness
                        for oem in ['3', '1']:  # LSTM and Legacy engines
                            config = f'--psm {psm} --oem {oem}'
                            txt = pytesseract.image_to_string(v, lang=tesseract_lang, config=config)
                            if txt.strip() and len(txt.strip()) > 3:  # Accept shorter results
                                final_texts.append(txt.strip())
                            
                            # Additional config for handwritten text
                            if oem == '1':  # Legacy engine better for handwriting
                                config_hw = f'--psm {psm} --oem 1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-/: ()'
                                txt_hw = pytesseract.image_to_string(v, lang=tesseract_lang, config=config_hw)
                                if txt_hw.strip() and len(txt_hw.strip()) > 3:
                                    final_texts.append(txt_hw.strip())
                                    
                except Exception as e:
                    logging.debug(f"Tesseract OCR failed (PSM {psm}, OEM {oem}): {e}")
        
        # EasyOCR for additional accuracy - always run
        if easyocr_reader:
            for v in versions:
                try:
                    # Run EasyOCR with different confidence thresholds for robustness
                    results = easyocr_reader.readtext(np.array(v), detail=1)
                    
                    # Extract text with different confidence levels
                    high_conf_texts = [item[1] for item in results if len(item) > 2 and item[2] > 0.5]
                    med_conf_texts = [item[1] for item in results if len(item) > 2 and item[2] > 0.3]
                    all_texts = [item[1] for item in results if len(item) > 1]
                    
                    # Add all confidence levels for maximum coverage
                    for txt_list in [high_conf_texts, med_conf_texts, all_texts]:
                        if txt_list:
                            txt = ' '.join(txt_list)
                            if txt.strip() and len(txt.strip()) > 3:
                                final_texts.append(txt.strip())
                                
                except Exception as e:
                    logging.debug(f"EasyOCR failed: {e}")
    
    # Smart deduplication - remove similar texts while preserving quality variations
    if not final_texts:
        return ""
    
    # Remove exact duplicates first
    unique_texts = list(dict.fromkeys([t for t in final_texts if t.strip()]))
    
    # If we have many results, filter by length and content quality
    if len(unique_texts) > 10:
        # Prefer longer, more complete results
        unique_texts.sort(key=lambda x: len(x), reverse=True)
        # Keep top results but ensure we don't lose short but important text
        filtered_texts = unique_texts[:5]  # Top 5 longest
        # Add any short texts that might be important (like numbers, names)
        for text in unique_texts[5:]:
            if len(text.strip()) < 20 and any(char.isdigit() for char in text):
                filtered_texts.append(text)
        unique_texts = filtered_texts
    
    # Combine with better formatting
    combined = "\n".join(unique_texts)
    return combined.strip() if combined else ""

def build_gemini_prompt(ocr_text: str) -> str:
    prompt = (
        "You are an information extraction agent for medical/insurance documents.\n"
        "Extract ALL possible fields, values, and tables from the OCR text below, "
        "including handwritten, garbled, or unclear fields. For tables (like bills, lab results), "
        "extract as a list of records with columns inferred from context.\n"
        "IMPORTANT: Multiple page images are provided for visual context. Use the images to interpret "
        "handwritten text and verify OCR accuracy, especially for medical terms, medications, and clinical notes.\n"
        "you have to mention all the details of the patient, doctor, hospital, diagnosis, medications( including the vaccine etc), treatments, lab results, billing, dates, and any other medical information not covered by the above fields.\n"
        "the ocr are of the processed information from the image, so you can improve accuracy leveraging both the ocr and the image.\n"
        "go through every image and if you find any information that is not present in the ocr, then you can use the image to extract the information.\n"
        "choose the best information from the ocr and the image to extract the information, also verify the information to medical terms and abbreviations\n"
        "Output ONLY JSON using this structured medical schema, AND include any additional fields you find:\n"
        "{\n"
        "  \"patient_info\": {\n"
        "    \"name\": \"\",\n"
        "    \"age\": \"\",\n"
        "    \"gender\": \"\",\n"
        "    \"uhid\": \"\",\n"
        "    \"ipd_opd_no\": \"\",\n"
        "    \"ward\": \"\",\n"
        "    \"bed_no\": \"\"\n"
        "  },\n"
        "  \"hospital_info\": {\n"
        "    \"name\": \"\",\n"
        "    \"address\": \"\",\n"
        "    \"phone\": \"\",\n"
        "    \"email\": \"\"\n"
        "  },\n"
        "  \"doctor_info\": {\n"
        "    \"name\": \"\",\n"
        "    \"department\": \"\",\n"
        "    \"referred_by\": \"\"\n"
        "  },\n"
        "  \"diagnoses\": [\n"
        "    {\n"
        "      \"diagnosis\": \"\",\n"
        "      \"type\": \"provisional|final|differential\",\n"
        "      \"certainty\": \"high|medium|low\"\n"
        "    }\n"
        "  ],\n"
        "  \"medications\": [\n"
        "    {\n"
        "      \"name\": \"\",\n"
        "      \"dose\": \"\",\n"
        "      \"route\": \"oral|injection|topical|inhalation\",\n"
        "      \"frequency\": \"\",\n"
        "      \"duration\": \"\"\n"
        "    }\n"
        "  ],\n"
        "  \"treatments\": [\n"
        "    {\n"
        "      \"treatment\": \"\",\n"
        "      \"date\": \"\",\n"
        "      \"type\": \"procedure|therapy|surgery\"\n"
        "    }\n"
        "  ],\n"
        "  \"lab_results\": [\n"
        "    {\n"
        "      \"test_name\": \"\",\n"
        "      \"result\": \"\",\n"
        "      \"unit\": \"\",\n"
        "      \"reference_range\": \"\",\n"
        "      \"date\": \"\",\n"
        "      \"lab_name\": \"\"\n"
        "    }\n"
        "  ],\n"
        "  \"billing\": {\n"
        "    \"total_amount\": \"\",\n"
        "    \"currency\": \"\",\n"
        "    \"items\": [\n"
        "      {\n"
        "        \"item\": \"\",\n"
        "        \"quantity\": \"\",\n"
        "        \"rate\": \"\",\n"
        "        \"amount\": \"\"\n"
        "      }\n"
        "    ]\n"
        "  },\n"
        "  \"dates\": {\n"
        "    \"admission_date\": \"\",\n"
        "    \"discharge_date\": \"\",\n"
        "    \"report_date\": \"\",\n"
        "    \"prescription_date\": \"\"\n"
        "  },\n"
        "  \"additional_fields\": {\n"
        "    \"doctor_notes\": \"\",\n"
        "    \"clinical_observations\": \"\",\n"
        "    \"vital_signs\": \"\",\n"
        "    \"symptoms\": \"\",\n"
        "    \"recommendations\": \"\",\n"
        "    \"follow_up\": \"\",\n"
        "    \"any_other_medical_info\": \"\"\n"
        "  }\n"
        "}\n"
        "Rules:\n"
        "- Normalize dates to YYYY-MM-DD format\n"
        "- For handwritten/garbled text, try to interpret context and extract meaningful information\n"
        "- Use empty string \"\" for missing values\n"
        "- Pay special attention to medical abbreviations (ART, CT, IPD, OPD, etc.)\n"
        "- Extract medication names even if partially garbled\n"
        "- Look for treatment protocols and clinical observations\n"
        "- IMPORTANT: Extract ANY additional medical information not covered by the above fields\n"
        "- Include doctor's notes, clinical observations, vital signs, symptoms, recommendations, follow-up instructions\n"
        "- If you find medical information that doesn't fit the defined fields, put it in additional_fields\n"
        "- Be comprehensive - extract everything medical-related you can find\n\n"
        f"OCR text:\n{ocr_text}\n"
    )
    return prompt

def call_gemini_image_ocr_extract(page_images: list, ocr_text: str) -> str:
    if os.getenv("SKIP_LLM"):
        logging.debug("SKIP_LLM=1 set; skipping LLM call.")
        return ""
    if generative_model is None:
        logging.debug("LLM not available.")
        return ""
    
    # Build parts list with prompt and all page images
    parts = [build_gemini_prompt(ocr_text)]
    
    # Add all page images
    for i, img_bytes in enumerate(page_images):
        parts.append({
            "mime_type": "image/png", 
            "data": img_bytes
        })
    
    generation_config = {
        "temperature": 0,
        "response_mime_type": "application/json"
    }
    
    # Retry logic for robustness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = generative_model.generate_content(parts, generation_config=generation_config)
            result = getattr(resp, "text", "") or ""
            if result.strip():
                return result
            else:
                logging.warning(f"⚠ LLM returned empty response (attempt {attempt + 1})")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():  # Rate limit
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logging.warning(f"⚠ Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
            else:
                logging.warning(f"⚠ LLM extraction failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:  # Last attempt
                    return ""
                time.sleep(1)  # Brief pause before retry
    
    return ""

JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")

def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = JSON_OBJECT_RE.search(text)
    if not match:
        return None
    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except Exception:
        return None

def process_documents(document_folder: str, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)

    def finalize_document(doc_key: str, ocr_chunks: list, page_images: list) -> None:
        if not ocr_chunks:
            return
        
        doc_stem = os.path.splitext(os.path.basename(doc_key))[0]
        ocr_out = os.path.join(output_folder, f"{doc_stem}.ocr.txt")
        llm_raw_out = os.path.join(output_folder, f"{doc_stem}.llm.txt")
        llm_json_out = os.path.join(output_folder, f"{doc_stem}.json")
        
        # Combine all OCR text
        full_ocr = "\n".join(ocr_chunks)
        with open(ocr_out, "w", encoding="utf-8") as f:
            f.write(full_ocr)
        
        # Single LLM call per document with all page images
        if page_images:
            llm_response = call_gemini_image_ocr_extract(page_images, full_ocr)
        else:
            llm_response = ""
        
        # Save LLM response if not empty
        if llm_response:
            with open(llm_raw_out, "w", encoding="utf-8") as f:
                f.write(llm_response)
            llm_json = extract_first_json(llm_response) or {}
        else:
            # Create empty JSON structure when SKIP_LLM is set
            llm_json = {
                "patient_info": {"name": "", "age": "", "gender": "", "uhid": "", "ipd_opd_no": "", "ward": "", "bed_no": ""},
                "hospital_info": {"name": "", "address": "", "phone": "", "email": ""},
                "doctor_info": {"name": "", "department": "", "referred_by": ""},
                "diagnoses": [],
                "medications": [],
                "treatments": [],
                "lab_results": [],
                "billing": {"total_amount": "", "currency": "", "items": []},
                "dates": {"admission_date": "", "discharge_date": "", "report_date": "", "prescription_date": ""},
                "additional_fields": {
                    "doctor_notes": "",
                    "clinical_observations": "",
                    "vital_signs": "",
                    "symptoms": "",
                    "recommendations": "",
                    "follow_up": "",
                    "any_other_medical_info": ""
                }
            }
        
        with open(llm_json_out, "w", encoding="utf-8") as f:
            json.dump(llm_json, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✓ Completed: {doc_key}")

    current_doc_key = None
    current_ocr_chunks = []
    current_page_images = []

    for source_id, page_idx, img_bytes in load_documents(document_folder):
        logging.info(f"📄 Processing {source_id}")
        
        # Extract document key (remove page suffix)
        doc_key = source_id.split('#', 1)[0]
        
        if current_doc_key is None:
            current_doc_key = doc_key
        elif doc_key != current_doc_key:
            # Finished previous document; flush its outputs
            finalize_document(current_doc_key, current_ocr_chunks, current_page_images)
            current_doc_key = doc_key
            current_ocr_chunks = []
            current_page_images = []

        # OCR per page (sequential processing) with error handling
        try:
            ocr_text = ensemble_ocr(img_bytes)
            if ocr_text.strip():
                current_ocr_chunks.append(f"\n\n===== PAGE {page_idx} : {doc_key} =====\n\n{ocr_text}")
                current_page_images.append(img_bytes)
            else:
                logging.warning(f"⚠ No text extracted from {source_id}")
                # Still append empty result to maintain page order
                current_ocr_chunks.append(f"\n\n===== PAGE {page_idx} : {doc_key} =====\n\n[No text extracted]")
                current_page_images.append(img_bytes)
        except Exception as e:
            logging.error(f"❌ OCR failed for {source_id}: {e}")
            current_ocr_chunks.append(f"\n\n===== PAGE {page_idx} : {doc_key} =====\n\n[OCR failed: {e}]")
            current_page_images.append(img_bytes)
        
        # Force garbage collection after each page to prevent memory issues
        gc.collect()

    # Flush last document
    if current_doc_key is not None:
        finalize_document(current_doc_key, current_ocr_chunks, current_page_images)

if __name__ == "__main__":
    logging.info("🚀 Starting Medical Document Extraction Pipeline")
    
    # Configuration
    docs_in = "/Users/ayann/Documents/GitHub/autoflow-data-extraction/AutoFlow/docs_in"
    docs_out = "/Users/ayann/Documents/GitHub/autoflow-data-extraction/AutoFlow/docs_out"
    
    # Validate input directory
    if not os.path.exists(docs_in):
        logging.error(f"❌ Input directory not found: {docs_in}")
        exit(1)
    
    # Create output directory if needed
    os.makedirs(docs_out, exist_ok=True)
    
    # Show configuration
    paddle_status = "✓ Enabled" if os.getenv("USE_PADDLEOCR", "1") == "1" else "⚠ Disabled"
    llm_status = "✓ Enabled" if generative_model and not os.getenv("SKIP_LLM") else "⚠ Disabled"
    
    logging.info(f"📁 Input: {docs_in}")
    logging.info(f"📁 Output: {docs_out}")
    logging.info(f"🔧 PaddleOCR: {paddle_status}")
    logging.info(f"🤖 LLM: {llm_status}")
    logging.info("=" * 50)
    
    try:
        process_documents(docs_in, docs_out)
        logging.info("=" * 50)
        logging.info("✅ Pipeline completed successfully!")
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        exit(1)
