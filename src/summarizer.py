"""
HR Policy Document Summarization Chain for RAG Integration
Refactored for cleaner architecture and better model utilization
Uses locally downloaded Llama-3.2-3B-Instruct model
"""

import torch
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
from rich.console import Console
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from typing import List, Dict, Any, Optional
import re
import logging
from dataclasses import dataclass , field
from pydantic import PrivateAttr
from langchain_core.callbacks import Callbacks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class SummarizationConfig:
    """Configuration for summarization behavior"""
    temperature: float = 0.1
    repetition_penalty: float = 1.1
    preserve_structure: bool = True
    redact_sensitive_info: bool = True
    include_metadata: bool = True

class LlamaLLM(LLM):
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _config: SummarizationConfig = PrivateAttr()
    callbacks: Callbacks = None  # ✅ Still public for LangChain

    def __init__(self, model, tokenizer, config: SummarizationConfig, callbacks: Callbacks = None):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self.callbacks = callbacks or []

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    temperature=self._config.temperature,
                    do_sample=self._config.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    repetition_penalty=self._config.repetition_penalty,
                    max_new_tokens=4000
                )

            response = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]

            return response.strip()

        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return f"Error: Could not generate response - {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "llama"


class ModelManager:
    """Manages Llama model lifecycle"""
    
    def __init__(self, model_path: str = None): # type: ignore
        self.model_path = model_path or "C:\\Users\\MayanksPotato\\Desktop\\DMRC_Chatbot\\chatbot\\models\\meta-llama\\Llama-3.2-3B-Instruct"
        self.model = None
        self.tokenizer = None
        self.llm_wrapper = None
        
    def load_model(self, config: SummarizationConfig) -> bool:
        """Load the Llama model and tokenizer"""
        try:
            console.print("\n🔄 [yellow]Loading Llama model...[/yellow]")
            
            # Load tokenizer first
            console.print("📝 [yellow]Loading tokenizer...[/yellow]")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            console.print("🧠 [yellow]Loading model...[/yellow]")
            
            # Memory configuration
            max_memory = {
                0: "6GiB",
                "cpu": "6GiB"
            }
            console.print("🔧 [yellow]Configuring model memory...[/yellow]")
            # Load model without device mapping first
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None
            )
            
            console.print("🔧 [yellow]Applying device mapping...[/yellow]")
            
            # Apply device mapping
            device_map = infer_auto_device_map(
                self.model,
                max_memory=max_memory, # type: ignore
                no_split_module_classes=["LlamaDecoderLayer"]
            )
            
            self.model = dispatch_model(self.model, device_map=device_map)
            
            console.print("🔗 [yellow]Creating LLM wrapper...[/yellow]")
            
            # Create LLM wrapper only after model and tokenizer are fully loaded
            self.llm_wrapper = LlamaLLM(self.model, self.tokenizer, config)
            
            console.print("✅ [green]Model loaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"❌ [red]Failed to load model: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up model resources"""
        console.print("\n🧹 [yellow]Cleaning up resources...[/yellow]")
        
        for attr in ['model', 'tokenizer', 'llm_wrapper']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        console.print("✅ [green]Resources cleaned up[/green]")
    
    def get_llm(self) -> Optional[LlamaLLM]:
        """Get the LLM wrapper"""
        return self.llm_wrapper

class ContentPreprocessor:
    """Handles content preprocessing and privacy redaction"""
    
    @staticmethod
    def clean_content(content: str) -> str:
        """Clean and normalize content"""
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Fix common formatting issues
        content = re.sub(r'(\d+)\.\s*([A-Z])', r'\1. \2', content)
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
        
        # Preserve structural elements
        content = re.sub(r'(\n|^)([A-Z\s]{3,})\n', r'\1**\2**\n', content)
        
        return content
    
    @staticmethod
    def extract_structure_hints(content: str) -> Dict[str, Any]:
        """Extract structural hints from content"""
        hints = {
            'has_numbered_steps': bool(re.search(r'\d+\.\s', content)),
            'has_bullet_points': bool(re.search(r'[•\-\*]\s', content)),
            'has_sections': bool(re.search(r'\n\s*[A-Z][A-Z\s]+\n', content)),
            'has_procedures': bool(re.search(r'(process|procedure|steps|requirements)', content, re.IGNORECASE)),
            'estimated_complexity': 'high' if len(content.split()) > 500 else 'medium' if len(content.split()) > 200 else 'low'
        }
        return hints

class PromptBuilder:
    """Builds dynamic prompts based on content characteristics"""
    
    def __init__(self, config: SummarizationConfig):
        self.config = config
    
    def build_summarization_prompt(self, content: str, metadata: Dict[str, Any], structure_hints: Dict[str, Any]) -> str:
        """Build a dynamic prompt based on content characteristics"""

        base_prompt = """You are a highly accurate and detail-oriented HR policy summarizer. Your job is to generate a **detailed, complete summary** that preserves all critical HR rules and procedures in structured bullet points.

## INPUT CONTENT:
{content}

## DOCUMENT CONTEXT:
- Chapter: {chapter_title}
- Section: {subtopic_title}

## ✳️ TASK GOAL:
Summarize the content while preserving:
- All key actions, responsibilities, and rules
- Procedural sequences and if-then conditions
- Timeframes, deadlines, exceptions, and penalties
- Any legal, financial, or disciplinary consequences

The summary must **not miss any numbered step**, **must not merge unrelated actions**, and **must retain specificity** of rules, amounts, and conditions.

## 🔒 PRIVACY REDACTION
Automatically replace any sensitive info with `[REDACTED]`, including:
- Names, emails, phone numbers
- Addresses, employee IDs
- Financial figures or salary info
- Legal identifiers or document codes

## ✅ FORMAT REQUIREMENTS:

### 📘 Overview
1-2 lines about the purpose of this policy.

### 🔑 Key Procedures
Use **grouped bullet points** to describe:
- Eligibility and allocation
- Responsibilities of the employee
- Procedures for leave, resignation, retirement, etc.
- Financial obligations (rent, fees, deductions)
- Rights and restrictions (use of premises, subletting, guests)
- Penalties and enforcement conditions

Use bullets like:
- ✅ Clear obligations
- ❗ Conditions and exceptions
- 🔁 Re-entry or reallocation conditions

### 📎 Requirements & Deadlines
Explicitly mention:
- Timeframes (e.g. 30-day notice)
- Retention periods
- Cut-off rules and decision-making authorities
- Approval conditions or documentation needed

## ⛔ DO NOT:
- Skip any important clause, rule, or timeline
- Use abstract language or vague generalizations
- Merge unrelated concepts into single bullets
- Omit license fees, overstay rules, or approval conditions


## ✍️ TARGET OUTPUT:
Adjust summary length **based on input length**:
- For long documents (2,000+ words), retain more detail; compression should not exceed **40–60%**.
- Preserve full procedural clarity, legal implications, and nested clauses.
- Goal: Summary may be **800–1,200 words** for 3,000–4,000 word documents.

Now generate the summary:"""

        # Customize prompt based on structure hints
        if structure_hints.get('has_numbered_steps'):
            base_prompt = base_prompt.replace(
                "## Key Procedures",
                "## Step-by-Step Procedures"
            )
        
        if structure_hints.get('estimated_complexity') == 'high':
            base_prompt = base_prompt.replace(
                "[Concise summary",
                "[Comprehensive summary"
            )
        
        return base_prompt

class HRPolicySummarizer:
    """Main summarization engine"""
    
    def __init__(self, llm: LlamaLLM, config: SummarizationConfig = None): # type: ignore
        self.llm = llm
        self.config = config or SummarizationConfig()
        self.preprocessor = ContentPreprocessor()
        self.prompt_builder = PromptBuilder(self.config)
        
    def summarize_document(self, document: Document) -> Dict[str, Any]:
        """Summarize a single document"""
        try:
            # Preprocess content
            clean_content = self.preprocessor.clean_content(document.page_content)
            structure_hints = self.preprocessor.extract_structure_hints(clean_content)
            
            # Build dynamic prompt
            prompt_template = self.prompt_builder.build_summarization_prompt(
                clean_content, 
                document.metadata, 
                structure_hints
            )
            from langchain_core.runnables import RunnableSequence

            # Create prompt using LangChain's modern interface
            prompt = PromptTemplate.from_template(prompt_template)
            # Compose the full pipeline
            chain = prompt | self.llm

            # Run the composed prompt -> LLM pipeline
            summary = chain.invoke({
                "content": clean_content,
                "chapter_title": document.metadata.get("chapter_title", "Unknown"),
                "subtopic_title": document.metadata.get("subtopic_title", "Unknown"),
                "source": document.metadata.get("source", "HR Policy Manual")
            })
            # Build result
            result = {
                "original_content": document.page_content,
                "structured_summary": summary,
                "metadata": {
                    **document.metadata,
                    "structure_hints": structure_hints,
                    "summary_stats": {
                        "original_words": len(document.page_content.split()),
                        "summary_words": len(summary.split()),
                        "compression_ratio": len(summary.split()) / len(document.page_content.split()) if document.page_content else 0
                    }
                },
                "privacy_redacted": "[REDACTED]" in summary,
                "processing_successful": True
            }
            
            logger.info(f"Summarized: {document.metadata.get('subtopic_title', 'Unknown')} "
                        f"({result['metadata']['summary_stats']['compression_ratio']:.2f} compression)")
            
            return result
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {
                "original_content": document.page_content,
                "structured_summary": f"Error: Summarization failed - {str(e)}",
                "metadata": document.metadata,
                "error": str(e),
                "processing_successful": False
            }
    
    def summarize_batch(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Process multiple documents"""
        results = []
        
        console.print(f"\n📝 [cyan]Processing {len(documents)} documents...[/cyan]")
        
        for i, doc in enumerate(documents, 1):
            console.print(f"Processing document {i}/{len(documents)}")
            result = self.summarize_document(doc)
            results.append(result)
            
            # Brief pause to prevent overwhelming
            time.sleep(0.1)
        
        successful = sum(1 for r in results if r.get('processing_successful', False))
        console.print(f"✅ [green]Completed: {successful}/{len(documents)} successful[/green]")
        
        return results

class RAGIntegrator:
    """Integrates summarization with RAG pipeline"""
    
    def __init__(self, summarizer: HRPolicySummarizer, vector_store=None):
        self.summarizer = summarizer
        self.vector_store = vector_store
    
    def prepare_qa_context(self, summary_result: Dict[str, Any]) -> str:
        """Convert summary to Q&A generation context"""
        summary = summary_result["structured_summary"]
        metadata = summary_result["metadata"]
        
        context = f"""
DOCUMENT SECTION: {metadata.get('chapter_title', 'Unknown')} - {metadata.get('subtopic_title', 'Unknown')}
SOURCE: {metadata.get('source', 'HR Policy Manual')}

CONTENT SUMMARY:
{summary}

PROCESSING INFO:
- Privacy Status: {'Redacted' if summary_result.get('privacy_redacted') else 'Standard'}
- Content Type: {metadata.get('structure_hints', {}).get('estimated_complexity', 'Unknown')}
"""
        return context.strip()
    
    def update_vector_store(self, documents: List[Document]) -> None:
        """Add summaries to vector store"""
        if not self.vector_store:
            logger.warning("No vector store provided")
            return
        
        summaries = self.summarizer.summarize_batch(documents)
        
        summary_docs = []
        for summary in summaries:
            if summary.get('processing_successful'):
                doc = Document(
                    page_content=summary["structured_summary"],
                    metadata={
                        **summary["metadata"],
                        "document_type": "summary",
                        "privacy_redacted": summary.get("privacy_redacted", False)
                    }
                )
                summary_docs.append(doc)
        
        if summary_docs:
            self.vector_store.add_documents(summary_docs)
            logger.info(f"Added {len(summary_docs)} summaries to vector store")

def create_test_document() -> Document:
    """Create a test document for demonstration"""
    content = """
    The following types of houses are available for staff housing: Type III Available for the employees in the Pay level of 11 and above. Total nine houses are available. Type II Available for the employees in the Pay Level 10. Total eight houses are available. Type IIA Available for the employees in the Pay Level 6 to 9. Total 20 houses are available. Type I Available for the employees in Group C in Pay Level 2 to 5. Total 40 houses are available. Type S Available for the employees in Group D. Total 55 houses are available. Consideration for allotment in all the cases is seniority in the respective pay ranges. GENERAL CONDITIONS 01\t Out of turn allotment may be made in exceptional situations if, in the judgement of the Director, institutional requirements so demand. 02\t If a person does not accept the offer of a house, his/her name will be shifted to the bottom of the waiting list.  Exceptions to this will be made with respect to the following cases. i) When an institutional commitment has been made; for such a situation, institutional commitment will have priority over seniority in the waiting list. ii) When the Institute decides to change over from one system to another and such a change involves marginal adjustments. 03\t If a person goes on a leave of absence or on deputation for a period not exceeding one year, he/she can either retain the house for the period of his/her leave or can let the Institute use the house (full or part) during his/her absence.  In the latter case, the person concerned will have the right to reoccupy the house when he/she returns. 04\t If a person goes on leave for a period exceeding one year he/she will have to surrender the house to the Institute from the date the leave commences, but his/her seniority will be kept intact. 05\t In every case, the allottee shall be deemed to be a licensee and not a tenant. 06\t HR Department shall monitor the availability of faculty houses periodically and inform the faculty concerned accordingly. CHAPTER 12 EMPLOYEES HOUSING – RULES & REGULATIONS 144 IIMA HR Policy Manual 2023 07\t The allottee will have to enter into an agreement with the Institute for the permissive use of the house allotted to him/her on a non-judicial stamp paper worth Rs. 300/-. 08\t An employee, who becomes entitled to a higher category of house, will not be displaced from the house under occupation until alternative accommodation in the next category is made available. 09\t An allottee, whether temporary or permanent or on Tenure Based Scaled Contract, shall cease to draw House Rent Allowance from the date of moving into the allotted house, or in case he/ she does not accept the offer, and there is no other claimant for the house. 10\t In addition to HRA, following license fees will be recovered from the employee: Type of Houses House Nos. Living Area @ IIMA (SQMT) Category as per GoI License Fees INR V 501 to 505 158.88 VI A 1840 IV (Old) 401 to 419 421 to 425 129.28 V A 1490 IV (New) 426 to 434 125.65 III (Old) 301 to 320 129.28 III (New) III-1 to III-9 108.74 Transit House (New) T-21 to T-35 108.74 Transit House (Old) T-5 to T-20 86.38 IV (Sp.) 790 II II-1 to II-8 86.38 IV (Sp.) II-A 201 to 220 65.24 IV 750 I 111 to 130 141 to 160 44.10 II 370 S S-1 to S-55 21.75 I 180 Garages Allotted parking to Type 500, 300, 400, T, III, II-A, I -- -- 50 11\t If an allottee dies, the allotment shall be cancelled from the date of death.  The Director will have the discretion to extend the period of retention of the house by the family of the deceased in appropriate cases for up to 4 months. During such occupation, the rent last paid by the deceased allottee will be payable to the Institute. 12\t If an allottee retires or resigns or is dismissed or removed from service, the allotment shall be cancelled from the date of such retirement, resignation, dismissal or removal. The Director will have the discretion to extend the period of retention of the house in appropriate cases for up to 4 months, and on such terms and conditions, he deems fit. 13\t An allottee who wants to vacate the residence shall give at least thirty days’ notice in writing to the Institute. In the case of shorter notice, he will be charged rent for the number of days by which the notice falls short of 30 days. 145 IIMA HR Policy Manual 2023 14\t An employee shall not sublet or transfer the residence allotted to him or her, or any portion thereof. 15\t The allottee may accommodate guests in his/her house for a period not exceeding three months. For the period exceeding three months, specific approval of the Director needs to be obtained. 16\t The liability for rent shall commence from the date of occupation of the residence. 17\t The employee to whom the house is allotted shall be personally responsible for the rent thereof and for any damage beyond fair wear and tear caused thereto or to services provided therein during the period for which the house is under his/her occupation. 18\t The employee to whom the house has been allotted shall take the possession of the house from the Maintainance/Engineering Office.  Likewise, at the time of vacating the house, he/ she shall hand over the house to the Maintainance/Engineering  Office. 19\t An allottee shall not use the house for any purpose except for residing with his/her family and shall maintain the premises and the compound, if any, attached thereto, in a clean and hygienic condition. 20\t There shall be no improper use of any allotted house.  For the purpose of this rule, `improper use’ shall include: (a) Unauthorised addition to/or alteration of any part of the house or premises; (b) using the house/premises or a portion thereof for purposes other than for strictly residential purposes; (c) unauthorised extension from electricity and water supply and other service connections or tampering therewith; and (d) using the house or any portion in such a way as to be a nuisance to, or as to offend others living on the campus, or using the house in such a way as to detract from the appearance of the campus. Any improper use of a house could lead to a cancellation of the allotment.  In case the residents use the house for any commercial activity, the allotment will be cancelled, and possession of the house will be taken over by the Institute forthwith. 21\t The allottee shall personally be responsible for the loss of or any damage to, beyond fair wear & tear, the building-fixtures, furniture, sanitary fittings, electrical installations, fencing, etc. provided therein, during the period of his or her occupation of the house. 22\t No cattle shall be kept in the house or in the compound of the house. 23\t The allottee shall allow the estate staff of the Institute or the workers of authorised contractors to have access to the house at all reasonable hours to inspect the building, the water supply, sanitary or electricity installation, fixtures, and furniture and to carry out such normal repairs thereto as the Estate Supervisor may consider necessary for the proper maintenance of the house. 24\t The allottee should see that no water is wasted by leakage in the water supply fittings or by careless or extravagant use by the occupants, and shall forthwith report to the maintenance staff any damage to or defect in the building, fixtures and fittings, electrical installations or fencing and gates for necessary action. 146 IIMA HR Policy Manual 2023 25\t Any incidence of infectious disease in the house must immediately be reported to the Medical Officer/Chief Administrative Officer of the Institute, and all precautions must be taken to prevent the spread of the infection. 26\t No inflammable material should be stored in the houses. 27\t The allottee will be responsible for all residents of the house, including servants abiding by these rules. 28\t The rent payable by an employee for any type of quarter occupied by him/her is decided by the Institute from time to time.  There will be additional charges for actual electricity consumption and services like Conservancy, Water Supply, Road & Street Lighting and Government Educational Cess, Municipal Tax etc.  These charges will be deducted from the salaries of the occupants each month. 29\t On any question of interpretation of these rules, the Director’s decision will be final. 30\t The Director will have the authority to modify these rules at any time. 31\t Houses shall be taken possession of from the engineering department and surrendered to the department upon vacating them. Other conditions for allotment and use of the house shall be as laid down in the Housing Agreement (see Annex) to be executed with the Institute. OCCUPATION OF CAMPUS HOUSE BEYOND DATE OF RETIREMENT The charges for overstay of employees is as follows: Type of houses Category House Nos Rate for Overstay INR V A+ 501-505 30000 IV(OLD) A+ 401-419 421-425 24000 IV(NEW) A+ 426-434 24000 III(OLD) A+ 301-320 24000 III(NEW) A III-1TO III-9 20000 TRANSIT HOUSE (NEW) A+ T-21 TO T-35 20000 TRANSIT HOUSE (OLD) A+ T-1 TO T-20 16000 II A II-1 TO II-8 12000 II-A B 201-220 10000 I C 111-130 141-160 6000 S-TYPE D S-1 TO S-55 3000 147 IIMA HR Policy Manual 2023 01 The telephone facility, if any provided in a house shall be withdrawn after two months of the occupant’s retirement even if permission to retain the accommodation for a longer period is granted. 02 The retirement benefits such as gratuity, leave salary etc. would be paid to a superannuated employee only on vacating the Institute house if he/she was allotted one. The charges for overstay will be recovered from the retirement dues. 03 Looking into the overall interest of the Institute, the Director may use his discretion in these matters. 04 Retention of up to two months will be at normal licence fee. 05 The above charges will be reviewed as and when desired by the Institute. 06 No retention beyond six months will be allowed. If an employee continues to occupy an accommodation without official permission, measures like withdrawal of common facilities will be considered. 07 The Institute  will write to the superannuating employees one year in advance about their impending retirement and the formalities they are supposed to complete for getting the retirement benefits like CPF/Pension/Gratuity etc. They may also be informed about the norms for keeping the campus accommodation beyond superannuation period in case they are occupying such houses. Employees who wish to stay in the Institute houses beyond the superannuation date should write to the CAO. 08 Looking into the overall interest of the Institute, Director may use his discretion in these matters. 09 In case of resignation/completion of the tenure for employees appointed on Tenure Based Scaled Contract, the charges for over stay after the last working day will be as per the table. This is subject to approval of the over stay by the competent authority. USE OF CAMPUS HOUSES FOR SOCIAL/RELIGIOUS PURPOSES The Institute may allow the employees to use campus houses for social /religious purposes. The charges applicable for the use of the facility will be as follows Sr. No. Type of House Charges per day 1. Transit house III - series & above 1000 2. II - series 500 3. IIA - series 500 4. I - series 300 5. S - series 200 Policy and norms for use of the guest house 1. The guest house, if available, may be allotted to any faculty coming from another institute either from India or abroad for collaborative research or on sabbatical or any other purpose. The occupant will be charged an amount of Rs.10000-/ month + actual charges for electricity/cable TV/ Telephone 148 IIMA HR Policy Manual 2023 2. After three months’ occupant will bear charges for any refills that may be needed for the cooking gas/cylinder 3. If the occupant wishes to avail boarding facility at IMDC, they will be entitled for the same at the prevailing rates for guest coupons (for breakfast, lunch/dinner). The rates are discounted rates and coupons can be obtained by the occupant from reception, IMDC 4. In case any permanent and Tenure Based Scaled Contract employee desires to use these guest houses for family functions such as marriage, for a short duration, then an amount of Rs.2000-/ per day will be charged. 149 IIMA HR Policy Manual 2023 HOUSING AGREEMENT Agreement for Permissive User This agreement made this __________day of _______________ Two Thousand and in favour of Indian Institute of Management, Ahmedabad (hereinafter called the “Employer”) by ______________________ (Emp.ID: __________) of Indian Institute of Management, Ahmedabad (hereinafter called the “Employee”). Whereas the employer has employed the employee in the services on the post ________________________ from ____________ whereas the employee has to discharge the duties of the employer in conformity with the rules of the Institute and whereas the nature of duties require that the employee should be in the vicinity of the employer’s Institute and whereas to facilitate such discharge of duties by the employee of the employer, the employee has requested for the permissive use of the premises belonging to the employer and more particularly described in the schedule hereunder written or any premises taken in exchange (hereinafter called the said premises) and the employer has agreed to do so. It is hereby agreed by and between the parties as under: 1. That the employer has permitted use of the said premises to the employee to enable the employee to properly discharge his duties of his services with the employer while the said premises shall at all-time be deemed to be in possession and ownership of the employer. 2. That the employee is permitted only to make use of premises for the residence purposes and for such use he shall pay license fee as per rules framed by the Institute from time to time for being such permissive user of the said premises. 3. That this permission of permissive use is purely temporary and the employer reserves the right to revoke it at any time by giving one month’s notice to the employee of its intention to do so. 4. That this agreement of permissive user does not create any right or interest in the property in favour of the employee as the property of the said premises remains in ownership and control of the employer.  The permissive user is permitted by the employer solely with a view to facilitate the employee to properly discharge his duties in the services. 5. That the employee shall not sublet or transfer the residence allotted to him or her, or any portion thereof of the out-houses. 6. That only the family members of the employee are allowed to stay in the campus house (parents, unmarried/dependent children).  Any other guest can be accommodated for a maximum duration of one week and beyond that approval of the CAO/Director and issue of Security Pass is necessary. 7. That the employee to whom the house is allotted shall be personally responsible for the rent thereof and for any damage beyond fair wear and tear caused thereto or to services provided therein during the period for which the house is under his/her occupation, also to keep the interior of the Licensed Premises in good condition. 8. That the employee shall not use the house for any purpose except for residing with his/her family and shall maintain the premises and the compound, if any, attached thereto, in a clean and hygienic condition. 150 IIMA HR Policy Manual 2023 9. That the employee shall not be indulged in any improper use of any allotted house.  For the purpose of this rule, ‘improper use’ shall include: (a) unauthorised addition to/or alteration of any part of the house or premises; (b) using the house/premises or a portion thereof for purposes other than for strictly residential purposes; (c) unauthorised extension from electricity and water supply and other service connections or tampering therewith; (d) using the house or any portion in such a way as to be a nuisance to, or as to offend others living on the campus, or using the house in such a way as to detract from the appearance of the campus; (e) Occupation of unauthorised person and subletting shall be strictly prohibited; (f) Conducts himself in a manner which is prejudicial to the maintenance of the harmonious relations with his neighbors. Any improper use of a house violation of Prohibition Act of the state, such as could lead to cancellation of the allotment.  In case the residents use the house for any commercial activity, the allotment will be cancelled and possession of the house will be taken over by the Institute forthwith. 10.\t That the employee shall not do in the Licensed Premises, any act, deed, matter or thing which may cause or likely to cause nuisance or annoyance to the occupiers of the building or the occupiers of the neighbors; 11.\t That the employee shall personally be responsible for loss of or any damage to, beyond fair wear & tear, the building-fixtures, furniture, sanitary fittings, electrical installations, fencing, etc. provided therein, during the period of his or her occupation of the house. 12.\t That the employee shall not keep cattle in the house or in the compound of the said house. 13.\t That the employee, must immediately report to the Medical Officer/Chief Administrative Officer of the Institute, about any incident of infectious disease in the family and all precautions must be taken to prevent the spread of the infection. 14.\t That the employee, shall not to store any hazardous or inflammable materials except authorised cooking gas cylinders in the Licensed Premises nor to carry on in the Licensed Premises any illegal activity. 15.\t That the employee, not to do or omit or suffer to be done anything whereby its right of use of the Licensed Premises or any part thereof is forfeited or extinguished. 16.\t That the employee shall not to use the Licensed Premises or any part thereof nor permit or suffer the same to be used for illegal, immoral antisocial, obnoxious or improper purposes nor cause or permit or suffer to be done upon the Licensed Premises or any part thereof anything which may offend any law, notification, rules or regulations of the Licensor in the Licensed Premises. 17.\t That the employee without previous consent in writing of the Licensors, not to make or erect or permit or suffer to be made or erected on the Licensed Premises or any part thereof, any structural alterations or additions or any other alterations which could affect or injure the structure of the walls of the Licensed Premises. 151 IIMA HR Policy Manual 2023 18.\t That the, no employee or his/her spouse or dependent(s) is permitted under any circumstances to carry out any commercial activity from the said premises and keep animals like Cow, Buffalo, Goat, Horse etc. Violating this clause will be treated as misconduct on the part of the employee and the allotment of residence will be liable to be cancelled. In addition to this it will be treated as breach of these rules for which the same penalty or damages will be applicable. 19.\t That the employee shall pay by way of recovery from salary, the actual charges for electricity and water consumption, service charges and other admissible charges at such rates as may be decided by the Institute from time to time 20.\t That the employee shall not carry out any additions or alterations in the said premises or electrical or sanitary installations therein and shall not do any damage to the property. 21.\t That the employee shall make good any damage caused to the said premises and shall be liable for the same to the employer. 22.\t That the employee shall use the said premises for his personal use and for no other purposes. The employee shall not permit such premises or any part thereof being used by any other person or family members those who are not dependent, they may entertain as guests for any purpose whatsoever.  The moment the employee is discharged from the services of the employer, this arrangement of permissive user shall stand terminated and the employer can instantaneously stop the employee from such permissive user. 23.\t That this arrangement for permissive user in no way would be construed as creating any right whatsoever in favour of the employee in the said premises or any part thereof. 24.\t That the Institute shall not be responsible or liable for any loss, damage, shortage, theft or destruction of any articles, property or things of any kind or nature whatsoever belonging to the employee and kept in the said Premises. 25.\t That the employee shall abide by the rules, regulations, bye-laws and executive instructions of the Institute governing occupation of residential accommodation for the staff of the Institute and also comply with all directions given from time to time by Institute with regard to the use of the Licensed Premises or any part thereof. Schedule of the premises above referred to: House Type  ,No. ___ bearing Municipal Census No: __________________________Vastrapur Taluka City Survey No.                    -            of ________ sq.mtr. (Name of Employee) Signed and delivered by th employee in the presence of 1. _____________________________________________________________ 2. _________________________________________________________ Chief Administrative Officer Indian Institute of Management Ahmedabad 1. _________________________________________________________ 2. _________________________________________________________
    """
    
    return Document(
        page_content=content,
        metadata={
            "chapter_title": "Employees Housing – Rules & Regulations",
            "subtopic_title": "HOUSES FOR STAFF",
            "source": "HR Policy Manual",
            "page_range": "153-161"
        }
    )