# enhanced_qa_graph.py

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import httpx
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable not found")

# Configuration
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL = "meta-llama/Llama-3-70b-chat-hf"
SUMMARY_FILE = "summaries.json"
OUTPUT_FILE = "qa_dataset.jsonl"
MIN_SUMMARY_WORDS = 100
REVIEW_REWRITE_ITERATIONS = 4
MAX_RETRIES = 3
TIMEOUT = 120.0

# State type definition
class QAState:
    def __init__(self):
        self.summaries: List[Dict[str, Any]] = []
        self.qa_results: List[Dict[str, Any]] = []
        self.index: int = 0
        self.qa_pairs: List[Dict[str, str]] = []
        self.metadata: Dict[str, Any] = {}
        self.doc_id: str = ""
        self.review_suggestions: str = ""
        self.iteration_count: int = 0
        self.end: bool = False
        self.error: Optional[str] = None

# ---------- Enhanced LLM Caller ----------
async def call_together_ai(messages: List[Dict[str, str]], max_retries: int = MAX_RETRIES) -> str:
    """Enhanced API caller with retry logic and better error handling"""
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stream": False
    }
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(TOGETHER_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    raise ValueError("Invalid response format from Together.ai")
                    
        except httpx.TimeoutException:
            logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error on attempt {attempt + 1}/{max_retries}: {e}")
            if e.response.status_code == 429:  # Rate limiting
                await asyncio.sleep(5 * (attempt + 1))
                continue
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
            
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    
    # This should never be reached due to the raise statements above, but ensures all paths return
    raise RuntimeError(f"Failed to get response after {max_retries} attempts")

# ---------- Validation Functions ----------
def validate_qa_pairs(qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Validate and clean Q/A pairs"""
    valid_pairs = []
    for pair in qa_pairs:
        if isinstance(pair, dict) and "question" in pair and "answer" in pair:
            question = str(pair["question"]).strip()
            answer = str(pair["answer"]).strip()
            
            # Basic validation
            if len(question) > 10 and len(answer) > 20:
                valid_pairs.append({
                    "question": question,
                    "answer": answer
                })
    return valid_pairs

def parse_json_response(response: str) -> List[Dict[str, str]]:
    """Parse JSON response with fallback handling"""
    try:
        # Try direct JSON parsing
        return json.loads(response)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try extracting JSON without code blocks
        json_match = re.search(r'(\[.*?\])', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Failed to parse JSON response: {response[:200]}...")
        return []

# ---------- Node: Start ----------
async def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the workflow by loading summaries"""
    logger.info("Starting Q/A generation workflow")
    
    try:
        if not Path(SUMMARY_FILE).exists():
            raise FileNotFoundError(f"Summary file {SUMMARY_FILE} not found")
        
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Filter summaries based on word count
        valid_summaries = []
        for d in data:
            if isinstance(d, dict) and "structured_summary" in d:
                word_count = len(d["structured_summary"].split())
                if word_count >= MIN_SUMMARY_WORDS:
                    valid_summaries.append(d)
        
        state["summaries"] = valid_summaries
        state["qa_results"] = []
        state["index"] = 0
        state["iteration_count"] = 0
        state["end"] = False
        state["error"] = None
        
        logger.info(f"Loaded {len(valid_summaries)} valid summaries")
        return state
        
    except Exception as e:
        logger.error(f"Error in start_node: {e}")
        state["error"] = str(e)
        state["end"] = True
        return state

# ---------- Node: Generate Q/A ----------
async def generate_qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Q/A pairs from HR policy summary"""
    if state.get("error") or state.get("end"):
        return state
    
    try:
        current_summary = state["summaries"][state["index"]]
        doc_id = current_summary.get("doc_id", f"doc_{state['index']}")
        
        logger.info(f"Generating Q/A for document: {doc_id}")
        
        # Enhanced prompt with better instructions
        prompt = f"""You are an expert HR consultant. Based on the following HR policy summary, generate exactly 5 realistic employee questions and comprehensive answers.

REQUIREMENTS:
1. Questions should be practical and likely to be asked by employees
2. Answers must be detailed, grounded in the policy, and actionable
3. Cover different aspects: eligibility, procedures, exceptions, penalties, etc.
4. Use professional but accessible language
5. Return ONLY valid JSON format

FORMAT (return exactly this structure):
[
  {{"question": "What are the eligibility criteria for...", "answer": "According to the policy..."}},
  {{"question": "How do I apply for...", "answer": "The application process involves..."}},
  {{"question": "What happens if I...", "answer": "In case of..."}},
  {{"question": "Are there any exceptions to...", "answer": "The policy states that exceptions..."}},
  {{"question": "What are the penalties for...", "answer": "Penalties for violations include..."}}
]

### HR Policy Summary:
{current_summary['structured_summary']}

### Context:
- Document ID: {doc_id}
- Chapter: {current_summary.get('metadata', {}).get('chapter_title', 'Unknown')}
- Section: {current_summary.get('metadata', {}).get('subtopic_title', 'Unknown')}

Generate the 5 Q/A pairs now:"""

        result = await call_together_ai([{"role": "user", "content": prompt}])
        qa_pairs = parse_json_response(result)
        qa_pairs = validate_qa_pairs(qa_pairs)
        
        if len(qa_pairs) < 3:  # Minimum requirement
            logger.warning(f"Generated only {len(qa_pairs)} valid Q/A pairs for {doc_id}")
            # Retry with simplified prompt
            simple_prompt = f"""Generate 5 simple HR questions and answers based on this policy summary. Return as JSON array:

{current_summary['structured_summary'][:1000]}

Format: [{{"question": "...", "answer": "..."}}]"""
            
            result = await call_together_ai([{"role": "user", "content": simple_prompt}])
            qa_pairs = parse_json_response(result)
            qa_pairs = validate_qa_pairs(qa_pairs)
        
        state["qa_pairs"] = qa_pairs
        state["metadata"] = current_summary.get("metadata", {})
        state["doc_id"] = doc_id
        state["iteration_count"] = 0
        
        logger.info(f"Generated {len(qa_pairs)} Q/A pairs for {doc_id}")
        return state
        
    except Exception as e:
        logger.error(f"Error in generate_qa_node: {e}")
        state["error"] = str(e)
        return state

# ---------- Node: Review ----------
async def review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Review Q/A pairs and provide improvement suggestions"""
    if state.get("error") or state.get("end"):
        return state
    
    try:
        state["iteration_count"] += 1
        logger.info(f"Reviewing Q/A pairs - iteration {state['iteration_count']}")
        
        prompt = f"""You are an expert HR policy reviewer. Analyze the following {len(state['qa_pairs'])} question-answer pairs and provide specific, actionable suggestions for improvement.

EVALUATION CRITERIA:
1. **Authenticity**: Do questions sound like real employee inquiries?
2. **Accuracy**: Are answers grounded in the provided policy?
3. **Completeness**: Are answers comprehensive and actionable?
4. **Clarity**: Are both questions and answers clear and professional?
5. **Diversity**: Do questions cover different policy aspects?

CURRENT Q/A PAIRS:
{json.dumps(state['qa_pairs'], indent=2)}

PROVIDE SUGGESTIONS:
For each Q/A pair, suggest specific improvements. Focus on:
- Making questions more realistic and employee-focused
- Enhancing answer detail and practical guidance
- Improving professional tone and clarity
- Adding missing policy references or procedures

Return your analysis as structured feedback for each pair."""

        suggestions = await call_together_ai([{"role": "user", "content": prompt}])
        state["review_suggestions"] = suggestions
        
        return state
        
    except Exception as e:
        logger.error(f"Error in review_node: {e}")
        state["error"] = str(e)
        return state

# ---------- Node: Rewrite ----------
async def rewrite_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite Q/A pairs based on review suggestions"""
    if state.get("error") or state.get("end"):
        return state
    
    try:
        logger.info(f"Rewriting Q/A pairs - iteration {state['iteration_count']}")
        
        prompt = f"""You are an expert HR writer. Apply the following improvement suggestions to enhance each Q/A pair. Maintain the original structure but improve quality, authenticity, and completeness.

ORIGINAL Q/A PAIRS:
{json.dumps(state['qa_pairs'], indent=2)}

IMPROVEMENT SUGGESTIONS:
{state['review_suggestions']}

REQUIREMENTS:
1. Keep the same number of Q/A pairs
2. Maintain JSON format
3. Improve based on suggestions while keeping core content
4. Ensure answers are policy-grounded and actionable
5. Make questions more employee-focused

Return the improved Q/A pairs in the exact same JSON format:
[
  {{"question": "improved question", "answer": "improved answer"}},
  ...
]"""

        improved_result = await call_together_ai([{"role": "user", "content": prompt}])
        improved_pairs = parse_json_response(improved_result)
        improved_pairs = validate_qa_pairs(improved_pairs)
        
        if len(improved_pairs) >= len(state["qa_pairs"]) * 0.8:  # Accept if at least 80% valid
            state["qa_pairs"] = improved_pairs
            logger.info(f"Successfully rewrote {len(improved_pairs)} Q/A pairs")
        else:
            logger.warning("Rewrite produced insufficient valid pairs, keeping original")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in rewrite_node: {e}")
        state["error"] = str(e)
        return state

# ---------- Node: Save ----------
async def save_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Save the final Q/A pairs and move to next document"""
    if state.get("error"):
        logger.error(f"Skipping save due to error: {state['error']}")
        state["index"] += 1
        return check_completion(state)
    
    try:
        # Ensure output directory exists
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        saved_count = 0
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for pair in state["qa_pairs"]:
                qa_record = {
                    "doc_id": state["doc_id"],
                    "metadata": state["metadata"],
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "generated_at": timestamp,
                    "iterations": state["iteration_count"]
                }
                json.dump(qa_record, f, ensure_ascii=False)
                f.write("\n")
                saved_count += 1
        
        logger.info(f"Saved {saved_count} Q/A pairs for document {state['doc_id']}")
        state["qa_results"].append({
            "doc_id": state["doc_id"],
            "qa_count": saved_count,
            "iterations": state["iteration_count"]
        })
        
        # Move to next document
        state["index"] += 1
        return check_completion(state)
        
    except Exception as e:
        logger.error(f"Error in save_node: {e}")
        state["error"] = str(e)
        return state

def check_completion(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if processing is complete"""
    if state["index"] >= len(state["summaries"]):
        state["end"] = True
        logger.info(f"Processing complete! Generated Q/A pairs for {len(state['qa_results'])} documents")
        
        # Log summary statistics
        total_pairs = sum(r["qa_count"] for r in state["qa_results"])
        avg_iterations = sum(r["iterations"] for r in state["qa_results"]) / len(state["qa_results"]) if state["qa_results"] else 0
        
        logger.info(f"Summary: {total_pairs} total Q/A pairs, {avg_iterations:.1f} avg iterations")
    
    return state

# ---------- Routing Functions ----------
def should_continue_review(state: Dict[str, Any]) -> str:
    """Determine if should continue review-rewrite loop"""
    if state.get("error") or state.get("end"):
        return "save"
    
    if state["iteration_count"] >= REVIEW_REWRITE_ITERATIONS:
        return "save"
    
    return "rewrite"

def should_continue_processing(state: Dict[str, Any]) -> str:
    """Determine if should continue processing or end"""
    if state.get("error") or state.get("end"):
        return END
    
    return "generate_qa"

# ---------- LangGraph Workflow Definition ----------
def create_workflow():
    """Create and configure the LangGraph workflow"""
    workflow = StateGraph(dict) # type: ignore
    
    # Add nodes
    workflow.add_node("start", RunnableLambda(start_node))
    workflow.add_node("generate_qa", RunnableLambda(generate_qa_node))
    workflow.add_node("review", RunnableLambda(review_node))
    workflow.add_node("rewrite", RunnableLambda(rewrite_node))
    workflow.add_node("save", RunnableLambda(save_node))
    
    # Define edges
    workflow.set_entry_point("start")
    workflow.add_edge("start", "generate_qa")
    workflow.add_edge("generate_qa", "review")
    
    # Review-rewrite loop with conditional routing
    workflow.add_conditional_edges(
        "review",
        should_continue_review,
        {
            "rewrite": "rewrite",
            "save": "save"
        }
    )
    workflow.add_edge("rewrite", "review")
    
    # Continue processing or end
    workflow.add_conditional_edges(
        "save",
        should_continue_processing,
        {
            "generate_qa": "generate_qa",
            END: END
        }
    )
    
    return workflow.compile()

# ---------- Main Execution ----------
async def main():
    """Main execution function"""
    logger.info("Starting Enhanced Q/A Generation Pipeline")
    
    try:
        # Initialize empty output file
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            pass
        
        # Create and run workflow
        app = create_workflow()
        
        # Run the workflow
        final_state = await app.ainvoke({})
        
        if final_state.get("error"):
            logger.error(f"Workflow completed with error: {final_state['error']}")
            return False
        
        logger.info("Q/A generation pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)