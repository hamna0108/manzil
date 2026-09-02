Manzil:
AI-Powered Real Estate Search EngineManzil is an end-to-end intelligent real estate discovery platform designed to close the gap between how people naturally describe what they want and how property inventory is structured. Instead of rigid dropdown filters, Manzil extracts structured search parameters from natural language using a fine-tuned, locally hosted language model, mapping user intent directly to active listings.

🚀 Key Features
Conversational NLU Engine: Powered by a fine-tuned Qwen-2.5 model (quantized to 4-bit GGUF via llama.cpp) to accurately parse complex, bilingual queries (e.g., "10 marla house in DHA under 3.5 crore"), with a Gemini 1.5 Flash fallback. 
Automated Daily ETL Pipeline: An isolated background worker using Playwright scrapes fresh property listings every 24 hours, normalizes local currency and area units, and geographically maps them using LocationIQ.  
Multi-Stage Retrieval System: Executes a three-stage search consisting of strict hard constraints, graceful relaxation (automatically expanding budget/radius if results are low), and semantic vector re-ranking using all-MiniLM-L6-v2. 
Interactive UI: A modern, responsive frontend built with React and Tailwind CSS, featuring voice-to-text search and interactive Mapbox visualizations.  

🛠️ Technical Stack
Frontend: React 18, TypeScript, Tailwind CSS  
Backend: FastAPI (Python 3.12), SQLAlchemy (Async), Uvicorn  
Data Pipeline: Playwright, APScheduler, Pandas  
AI & Search: Llama.cpp (Qwen-2.5 LoRA), Sentence-Transformers, Numpy  
