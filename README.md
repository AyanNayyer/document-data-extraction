Medical document data extraction system using computer vision and AI

### **1. Multi-Engine OCR Pipeline**

- **Tesseract OCR**: Primary engine with multiple PSM (Page Segmentation Mode) configurations optimized for medical documents
- **EasyOCR**: Secondary engine for enhanced accuracy and confidence filtering
- **PaddleOCR**: Advanced engine for handwritten text recognition

### **2. Advanced Image Preprocessing**

- **OpenCV-based enhancement**: Adaptive thresholding, morphological operations, deskewing
- **PIL-based preprocessing**: Grayscale conversion, contrast enhancement, sharpening, brightness adjustment
- **Multiple image versions**: Creates 8+ different preprocessed versions for ensemble accuracy

### **3. AI-Powered Information Extraction**

- **Google Gemini LLM**: Analyzes OCR text + original images to extract structured medical data
- **Comprehensive medical schema**: Extracts patient info, hospital details, medications, diagnoses, lab results, billing, etc.

### **4. Output Generation**

- **OCR text files**: Raw extracted text from each page
- **LLM analysis files**: AI-processed structured data
- **JSON output**: Structured medical information in standardized format

## **Technologies Used 🛠️**

### **Computer Vision & OCR**

- **Tesseract**: Traditional OCR with medical document optimization
- **EasyOCR**: Deep learning-based OCR with confidence scoring
- **PaddleOCR**: Advanced handwritten text recognition (currently disabled)
- **OpenCV**: Image preprocessing and enhancement
- **PIL/Pillow**: Image manipulation and processing

### **AI & Machine Learning**

- **Google Gemini 1.5 Flash**: Large Language Model for information extraction
- **Multi-engine ensemble**: Combines multiple OCR results for maximum accuracy

### **Document Processing**

- **pdf2image**: PDF to image conversion
- **Multiprocessing**: Parallel processing for performance
- **JSON**: Structured data output

### **Development & Deployment**

- **Python**: Core programming language
- **Environment variables**: Configuration management
- **Logging**: Comprehensive error tracking and debugging

## **Current Capabilities ✅**

- **Multi-format support**: PDF and image files (JPG, PNG, TIFF, BMP)
- **Handwritten text recognition**: Through enhanced Tesseract + EasyOCR
- **Medical document specialization**: Optimized for hospital records, prescriptions, lab reports
- **Structured output**: JSON format with comprehensive medical schema
- **Error handling**: Robust processing with fallback mechanisms

## **Next Steps for Improvement 🚀**

**1. Azure Vision Integration**

**2. Azure Form Recognizer (For structured documents)**

### **3. Enhanced AI Pipeline**

- **Azure OpenAI**: Replace Gemini with Azure OpenAI for better integration
- **Custom medical models**: Fine-tune models on medical document datasets
- **Real-time processing**: Stream processing for large document volumes

### **4. Production Improvements**

- **Docker containerization**: For consistent deployment
- **API endpoints**: REST API for integration
- **Database storage**: Store extracted data in medical databases
- **Security**: HIPAA compliance and data encryption
- **Monitoring**: Performance metrics and accuracy tracking

### **5. Advanced Features**

- **Multi-language support**: For international medical documents
- **Document classification**: Auto-categorize document types
- **Quality assurance**: Confidence scoring and human review workflows
- **Batch processing**: Handle large document volumes efficiently

## Working

- **Place input docs**: Put PDFs/images in AutoFlow/docs_in/
- **Outputs location**: Results appear in AutoFlow/docs_out/ as:
    - <name>.ocr.txt (raw OCR)
    - <name>.llm.txt (LLM raw response)
    - <name>.json (final structured data)
- **API keys/config (.env)**: Create .env in repo root with:
    - GEMINI_API_KEY=your_key
    - GEMINI_MODEL_NAME=gemini-1.5-flash (optional)
- USE_PADDLEOCR=1 to try PaddleOCR (optional; process-isolated, may be slow)
- SKIP_LLM=1 to run OCR-only (optional)
- **Install deps**:
    - Python packages: pip install pillow numpy opencv-python easyocr pytesseract pdf2image pypdfium2 python-dotenv google-generativeai
    - System: Tesseract OCR, and poppler/pdfium as required by pdf2image
- **Run**:
    - python3 /Users/ayann/Documents/GitHub/autoflow-data-extraction/extraction_fixed.py
    
- **How it works** :

Advanced preprocessing (8+ variants per page) → multi-engine OCR (Tesseract + EasyOCR, optional PaddleOCR) → combine text → Gemini LLM uses OCR + original images → emit structured medical JSON.

The current system is already quite sophisticated with its multi-engine OCR approach and AI-powered extraction. Azure Vision would be a natural next step to improve accuracy and scalability, especially for handwritten medical text recognition.