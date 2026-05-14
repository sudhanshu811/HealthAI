"""
HealthAI — Production Backend
Orchestrates 6 LlamaIndex pipelines via a single /chat endpoint.
Pipelines: basic_rag | multi_doc_agent | multimodal | react_agent | router | subquestion
"""

import os, asyncio, base64, hashlib, logging, time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── env ───────────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthai")

# ─── in-memory LRU cache (cheap, no Redis needed) ──────────────────────────
from collections import OrderedDict

class LRUCache:
    def __init__(self, maxsize=200):
        self._d = OrderedDict()
        self._max = maxsize

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def set(self, key, val):
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = val
        if len(self._d) > self._max:
            self._d.popitem(last=False)

_cache = LRUCache()

# ─── pipeline registry (lazy-loaded) ───────────────────────────────────────
_engines: dict = {}

def _hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE 1 — Basic RAG (health documents)
# ═══════════════════════════════════════════════════════════════════════════
def _build_basic_rag():
    from llama_index.llms.mistralai import MistralAI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

    llm = MistralAI(model="mistral-small-latest", api_key=MISTRAL_API_KEY)
    embed = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed
    Settings.chunk_size = 512

    data_path = "./data/health"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        logger.warning("No health data found — using fallback stub")
        return None

    docs = SimpleDirectoryReader(data_path).load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine(similarity_top_k=3)


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE 2 — Multi-Document Agent (disease outbreaks)
# ═══════════════════════════════════════════════════════════════════════════
def _build_multi_doc_agent():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings, VectorStoreIndex, SummaryIndex
    from llama_index.core.schema import Document, IndexNode
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
    from llama_index.core.llms.callbacks import llm_completion_callback

    class MistralLLM(CustomLLM):
        model: str = "mistral-small-latest"
        temperature: float = 0.0
        max_tokens: int = 1024
        api_key: str = MISTRAL_API_KEY

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(num_output=self.max_tokens, model_name=self.model)

        @llm_completion_callback()
        def complete(self, prompt: str, **kwargs) -> CompletionResponse:
            with httpx.Client(timeout=60) as client:
                r = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                          "temperature": self.temperature, "max_tokens": self.max_tokens},
                )
                r.raise_for_status()
                return CompletionResponse(text=r.json()["choices"][0]["message"]["content"])

        @llm_completion_callback()
        def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
            yield self.complete(prompt, **kwargs)

    embed = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    llm = MistralLLM()
    Settings.llm = llm
    Settings.embed_model = embed

    disease_config = {
        "covid19": "COVID-19",
        "malaria": "Malaria",
        "ebola": "Ebola",
        "cholera": "Cholera",
        "tuberculosis": "Tuberculosis",
    }

    data_path = "./data/disease_outbreaks"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        logger.warning("No disease data found — using fallback stub")
        return None

    parser = MarkdownNodeParser()
    disease_engines = {}
    agent_summaries = {}

    for key, name in disease_config.items():
        filepath = os.path.join(data_path, f"{key}.txt")
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            text = f.read()
        doc = Document(text=text, metadata={"disease": name, "source": key})
        nodes = parser.get_nodes_from_documents([doc])
        v_idx = VectorStoreIndex(nodes)
        s_idx = SummaryIndex(nodes)
        v_tool = QueryEngineTool(
            query_engine=v_idx.as_query_engine(similarity_top_k=2),
            metadata=ToolMetadata(name=f"{key}_vector", description=f"Factual details about {name}")
        )
        s_tool = QueryEngineTool(
            query_engine=s_idx.as_query_engine(),
            metadata=ToolMetadata(name=f"{key}_summary", description=f"Summary/overview of {name}")
        )
        disease_engines[key] = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[v_tool, s_tool],
        )
        agent_summaries[key] = f"{name} disease intelligence agent"

    objects = []
    for key, name in disease_config.items():
        if key in disease_engines:
            objects.append(IndexNode(text=agent_summaries[key], index_id=key, obj=disease_engines[key]))

    if not objects:
        return None

    top_index = VectorStoreIndex(objects=objects)
    return top_index.as_query_engine(similarity_top_k=2)


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE 4 — ReAct Agent (clinical decision support)
# ═══════════════════════════════════════════════════════════════════════════
def _build_react_agent():
    import nest_asyncio
    nest_asyncio.apply()

    from llama_index.llms.mistralai import MistralAI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.tools import FunctionTool, QueryEngineTool
    from llama_index.core.agent import ReActAgent
    from llama_index.core.node_parser import SentenceSplitter  # noqa — used below

    llm = MistralAI(model="mistral-small-latest", api_key=MISTRAL_API_KEY)
    embed = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed
    Settings.chunk_size = 512

    def calculate_bmi(weight_kg: float, height_m: float) -> str:
        """Calculate Body Mass Index (BMI) given weight in kilograms and height in meters."""
        if height_m <= 0:
            return "Error: Height must be greater than 0."
        bmi = round(weight_kg / (height_m ** 2), 2)
        if bmi < 18.5: cat = "Underweight"
        elif bmi < 25: cat = "Normal weight"
        elif bmi < 30: cat = "Overweight"
        elif bmi < 35: cat = "Obese Class I"
        elif bmi < 40: cat = "Obese Class II"
        else: cat = "Obese Class III (Severe)"
        return f"BMI = {bmi} kg/m² — Category: {cat}"

    def classify_blood_pressure(systolic: int, diastolic: int) -> str:
        """Classify blood pressure per AHA/ACC 2017 guidelines."""
        if systolic > 180 or diastolic > 120:
            return "Hypertensive Crisis — Seek emergency care immediately."
        elif systolic >= 140 or diastolic >= 90:
            return "Stage 2 Hypertension — Medication + lifestyle changes needed."
        elif systolic >= 130 or diastolic >= 80:
            return "Stage 1 Hypertension — Lifestyle changes and possible medication."
        elif systolic >= 120:
            return "Elevated — Lifestyle changes recommended."
        return "Normal Blood Pressure — Maintain healthy habits."

    def calculate_drug_dosage(drug_name: str, weight_kg: float, dose_mg_per_kg: float) -> str:
        """Calculate weight-based drug dosage."""
        if weight_kg <= 0 or dose_mg_per_kg <= 0:
            return "Error: weight and dose must be positive."
        total = round(weight_kg * dose_mg_per_kg, 2)
        return f"Drug: {drug_name} | Weight: {weight_kg} kg | Dose: {dose_mg_per_kg} mg/kg → Total: {total} mg"

    def estimate_egfr(creatinine_mg_dl: float, age: int, is_female: bool = False) -> str:
        """Estimate eGFR using MDRD formula for kidney function assessment."""
        egfr = 175 * (creatinine_mg_dl ** -1.154) * (age ** -0.203)
        if is_female:
            egfr *= 0.742
        egfr = round(egfr, 1)
        if egfr >= 90: stage = "G1 — Normal"
        elif egfr >= 60: stage = "G2 — Mildly decreased"
        elif egfr >= 45: stage = "G3a — Mildly to moderately decreased"
        elif egfr >= 30: stage = "G3b — Moderately to severely decreased"
        elif egfr >= 15: stage = "G4 — Severely decreased"
        else: stage = "G5 — Kidney failure"
        return f"eGFR = {egfr} mL/min/1.73m² — CKD Stage: {stage}"

    tools = [
        FunctionTool.from_defaults(fn=calculate_bmi),
        FunctionTool.from_defaults(fn=classify_blood_pressure),
        FunctionTool.from_defaults(fn=calculate_drug_dosage),
        FunctionTool.from_defaults(fn=estimate_egfr),
    ]

    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=60, paragraph_separator="\n\n")

    # Load diabetes protocol (Notebook 4 Part 2 — mirrors uber_2021.pdf)
    diabetes_path = "./data/clinical_guidelines/diabetes_protocol.txt"
    if os.path.exists(diabetes_path):
        d_docs = SimpleDirectoryReader(input_files=[diabetes_path]).load_data()
        d_nodes = splitter.get_nodes_from_documents(d_docs)
        d_idx = VectorStoreIndex(d_nodes)
        tools.append(QueryEngineTool.from_defaults(
            query_engine=d_idx.as_query_engine(similarity_top_k=3),
            name="diabetes_protocol",
            description=(
                "Provides clinical guidelines for Type 2 Diabetes management. "
                "Covers diagnosis criteria, glycemic targets, first-line Metformin dosing, "
                "second-line agents (GLP-1, SGLT-2, DPP-4, insulin), monitoring schedule, "
                "lifestyle modifications, complications, and hypoglycemia Rule of 15. "
                "Use a detailed plain text question as input."
            ),
        ))

    # Load hypertension protocol (Notebook 4 Part 2 — mirrors lyft_2021.pdf)
    htn_path = "./data/clinical_guidelines/hypertension_protocol.txt"
    if os.path.exists(htn_path):
        h_docs = SimpleDirectoryReader(input_files=[htn_path]).load_data()
        h_nodes = splitter.get_nodes_from_documents(h_docs)
        h_idx = VectorStoreIndex(h_nodes)
        tools.append(QueryEngineTool.from_defaults(
            query_engine=h_idx.as_query_engine(similarity_top_k=3),
            name="hypertension_protocol",
            description=(
                "Provides clinical guidelines for Hypertension management. "
                "Covers BP classification (Normal/Elevated/Stage1/Stage2/Crisis), "
                "lifestyle modifications, four first-line drug classes (ACE inhibitors, ARBs, "
                "CCBs, diuretics), special populations (diabetes, CKD, pregnancy, elderly), "
                "and hypertensive emergencies. "
                "Use a detailed plain text question as input."
            ),
        ))

    # Fallback: load any remaining guidelines files
    extra_path = "./data/clinical_guidelines"
    if os.path.exists(extra_path):
        loaded = {"diabetes_protocol.txt", "hypertension_protocol.txt"}
        extras = [f for f in os.listdir(extra_path) if f not in loaded and f.endswith(".txt")]
        if extras:
            e_docs = SimpleDirectoryReader(input_files=[
                os.path.join(extra_path, f) for f in extras
            ]).load_data()
            e_idx = VectorStoreIndex.from_documents(e_docs)
            tools.append(QueryEngineTool.from_defaults(
                query_engine=e_idx.as_query_engine(similarity_top_k=3),
                name="clinical_guidelines_rag",
                description="Search general clinical guidelines for healthcare questions.",
            ))

    return ReActAgent.from_tools(tools, llm=llm, verbose=False, max_iterations=10)


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE 5 — Router Query Engine (clinical guidelines)
# ═══════════════════════════════════════════════════════════════════════════
def _build_router():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.tools.query_engine import QueryEngineTool
    from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
    from llama_index.core.selectors.llm_selectors import LLMMultiSelector
    from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
    from llama_index.core.llms.callbacks import llm_completion_callback

    class MistralLLM(CustomLLM):
        model: str = "mistral-small-latest"
        temperature: float = 0.0
        max_tokens: int = 1024
        api_key: str = MISTRAL_API_KEY

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(num_output=self.max_tokens, model_name=self.model)

        @llm_completion_callback()
        def complete(self, prompt: str, **kwargs) -> CompletionResponse:
            with httpx.Client(timeout=60) as client:
                r = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                          "temperature": self.temperature, "max_tokens": self.max_tokens},
                )
                r.raise_for_status()
                return CompletionResponse(text=r.json()["choices"][0]["message"]["content"])

        @llm_completion_callback()
        def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
            yield self.complete(prompt, **kwargs)

    embed = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    llm = MistralLLM()
    Settings.llm = llm
    Settings.embed_model = embed

    data_path = "./data/clinical_guidelines"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        return None

    docs = SimpleDirectoryReader(data_path).load_data()
    splitter = SemanticSplitterNodeParser(embed_model=embed)
    nodes = splitter.get_nodes_from_documents(docs)

    v_idx = VectorStoreIndex(nodes)
    s_idx = SummaryIndex(nodes)

    v_engine = v_idx.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )
    s_engine = s_idx.as_query_engine(response_mode="compact")

    v_tool = QueryEngineTool.from_defaults(
        query_engine=v_engine,
        description=(
            "Useful for specific factual questions about clinical guidelines. "
            "If the answer is not in the guidelines, say so. Do not invent information."
        ),
    )
    s_tool = QueryEngineTool.from_defaults(
        query_engine=s_engine,
        description="Useful for summaries or overviews of clinical guidelines.",
    )

    return RouterQueryEngine(
        selector=LLMMultiSelector.from_defaults(max_outputs=2),
        query_engine_tools=[v_tool, s_tool],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE 6 — SubQuestion (hospital comparator)
# ═══════════════════════════════════════════════════════════════════════════
def _build_subquestion():
    import nest_asyncio
    nest_asyncio.apply()

    from llama_index.llms.mistralai import MistralAI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.node_parser import SentenceWindowNodeParser
    from llama_index.core.postprocessor import MetadataReplacementPostProcessor
    from llama_index.core.query_engine import SubQuestionQueryEngine
    from llama_index.core.tools import QueryEngineTool, ToolMetadata

    llm = MistralAI(model="mistral-small-latest", api_key=MISTRAL_API_KEY)
    embed = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed
    Settings.chunk_size = 512

    data_path = "./data/hospital_reports"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        return None

    hospitals = {
        "city_general": "City General Hospital",
        "metro_health": "Metro Health Centre",
        "sunrise_medical": "Sunrise Medical Institute",
    }

    parser = SentenceWindowNodeParser.from_defaults(window_size=3, window_metadata_key="window")
    pp = MetadataReplacementPostProcessor(target_metadata_key="window")

    tools = []
    for key, name in hospitals.items():
        filepath = os.path.join(data_path, f"hospital_{key}.txt")
        if not os.path.exists(filepath):
            continue
        docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
        nodes = parser.get_nodes_from_documents(docs)
        idx = VectorStoreIndex(nodes)
        engine = idx.as_query_engine(similarity_top_k=3, node_postprocessors=[pp])
        tools.append(QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(name=key, description=f"Annual performance report for {name}"),
        ))

    if not tools:
        return None

    return SubQuestionQueryEngine.from_defaults(query_engine_tools=tools, use_async=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE 3 — Multimodal (pharmaceutical labels) — direct API call
# ═══════════════════════════════════════════════════════════════════════════
async def _multimodal_query(image_b64: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "pixtral-12b-2409",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": 1024,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ─── query classifier ──────────────────────────────────────────────────────
def _classify_query(query: str) -> str:
    """Auto-route query to the best pipeline."""
    q = query.lower()

    if any(w in q for w in ["compare", "comparison", "vs", "versus", "difference between",
                             "hospital", "readmission", "mortality rate", "staffing"]):
        return "subquestion"

    if any(w in q for w in ["bmi", "blood pressure", "dosage", "egfr", "kidney",
                             "calculate", "compute", "mmhg", "systolic", "diastolic",
                             "weight", "creatinine", "clinical decision"]):
        return "react_agent"

    if any(w in q for w in ["outbreak", "ebola", "malaria", "covid", "cholera",
                             "tuberculosis", "epidemic", "transmission", "fatality rate"]):
        return "multi_doc_agent"

    if any(w in q for w in ["guideline", "protocol", "management", "treatment plan",
                             "diabetes management", "hypertension management"]):
        return "router"

    return "basic_rag"


# ─── lazy engine loader ────────────────────────────────────────────────────
_build_map = {
    "basic_rag":      _build_basic_rag,
    "multi_doc_agent": _build_multi_doc_agent,
    "react_agent":    _build_react_agent,
    "router":         _build_router,
    "subquestion":    _build_subquestion,
}

def _get_engine(mode: str):
    if mode not in _engines:
        logger.info(f"Building pipeline: {mode}")
        engine = _build_map[mode]()
        _engines[mode] = engine
    return _engines[mode]


# ─── app ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Generate drug label images and ensure data dirs exist
    try:
        from backend.data_init import ensure_data_dirs, generate_drug_labels
        ensure_data_dirs()
        generate_drug_labels()
    except Exception as e:
        logger.warning(f"data_init skipped: {e}")

    logger.info("HealthAI starting up — warming basic_rag pipeline")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_engine, "basic_rag")
    except Exception as e:
        logger.warning(f"Warm-up skipped: {e}")
    yield
    logger.info("HealthAI shut down")


app = FastAPI(title="HealthAI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── request / response models ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    mode: str = "auto"   # auto | basic_rag | multi_doc_agent | react_agent | router | subquestion
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    mode: str
    latency_ms: int
    cached: bool


# ─── /chat endpoint ────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    mode = req.mode if req.mode != "auto" else _classify_query(req.query)

    cache_key = f"{mode}:{_hash(req.query)}"
    cached = _cache.get(cache_key)
    if cached:
        return ChatResponse(answer=cached, mode=mode,
                            latency_ms=int((time.time()-start)*1000), cached=True)

    loop = asyncio.get_event_loop()
    try:
        engine = await loop.run_in_executor(None, _get_engine, mode)
        if engine is None:
            answer = (
                "This pipeline requires data files that are not yet uploaded. "
                "Please add your .txt documents to the appropriate data folder."
            )
        else:
            response = await loop.run_in_executor(None, engine.query, req.query)
            answer = str(response)
    except Exception as e:
        logger.error(f"Pipeline {mode} error: {e}", exc_info=True)
        answer = f"I encountered an error processing your request. ({type(e).__name__})"

    _cache.set(cache_key, answer)
    return ChatResponse(
        answer=answer,
        mode=mode,
        latency_ms=int((time.time()-start)*1000),
        cached=False,
    )


# ─── /chat/image endpoint (multimodal) ─────────────────────────────────────
@app.post("/chat/image")
async def chat_image(
    file: UploadFile = File(...),
    prompt: str = Form(default="Analyze this pharmaceutical label and extract drug information."),
):
    start = time.time()
    img_bytes = await file.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    cache_key = f"multimodal:{_hash(prompt + img_b64[:100])}"
    cached = _cache.get(cache_key)
    if cached:
        return {"answer": cached, "mode": "multimodal",
                "latency_ms": int((time.time()-start)*1000), "cached": True}

    try:
        answer = await _multimodal_query(img_b64, prompt)
    except Exception as e:
        logger.error(f"Multimodal error: {e}", exc_info=True)
        answer = f"Image analysis failed. ({type(e).__name__})"

    _cache.set(cache_key, answer)
    return {"answer": answer, "mode": "multimodal",
            "latency_ms": int((time.time()-start)*1000), "cached": False}


# ─── health check ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "loaded_pipelines": list(_engines.keys())}


# ─── serve frontend ────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="./frontend", html=True), name="frontend")
