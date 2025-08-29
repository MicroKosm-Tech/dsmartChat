import os
import sys
import json
import colorama
from colorama import Fore, Style
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, Any, List, Tuple, Union
import time
import re
import logging
import uuid
import pandas as pd
import numpy as np
from pprint import pprint
import threading
from decimal import Decimal
import datetime
from pydantic import BaseModel
from sqlalchemy import text


# Add a global json configuration to ensure Arabic text is properly handled
def setup_json_config():
    """Configure JSON to properly handle Arabic text by default"""
    # Store the original dumps function
    _original_dumps = json.dumps

    def _patched_dumps(obj, **kwargs):
        # Always ensure Arabic characters are preserved
        if "ensure_ascii" not in kwargs:
            kwargs["ensure_ascii"] = False
        return _original_dumps(obj, **kwargs)

    # Replace the standard dumps function with our patched version
    json.dumps = _patched_dumps


# Apply the JSON configuration at module import time
setup_json_config()

from .agent_tools import (
    store_patient_details,
    dynamic_doctor_search,
    analyze_symptoms,
    execute_offers_search,
)
from .common import write
from .consts import SYSTEM_AGENT_ENHANCED
from .utils import (
    CustomCallBackHandler,
    thread_local,
    setup_logging,
    log_data_to_csv,
    format_csv_data,
    clear_symptom_analysis_data,
)
from enum import Enum
from .specialty_matcher import detect_symptoms_and_specialties, SpecialtyDataCache
from .query_builder_agent import (
    extract_search_criteria_tool,
    unified_doctor_search_tool,
    unified_doctor_search,
    extract_search_criteria_from_message,
)
from .db import DB

# Initialize thread_local storage
thread_local = threading.local()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the database connection
db = DB()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))


# Helper function to clear symptom analysis data
def clear_symptom_analysis(reason="", session_id=None):
    """
    Clear symptom analysis data from thread_local storage and other possible locations

    Args:
        reason: Optional reason for clearing the data for logging
        session_id: Optional session ID to clear data from specific session history
    """
    # Call our utils version for comprehensive clearing first
    clear_symptom_analysis_data(reason)

    # Also clear from session history if provided
    if session_id and session_id in store:
        history = store[session_id]
        history.clear_symptom_data()
    # If no session_id provided but thread_local has one, use that
    elif hasattr(thread_local, "session_id") and thread_local.session_id in store:
        session_id = thread_local.session_id
        history = store[session_id]
        history.clear_symptom_data()

    # Clear any symptom-related fields that might be in thread_local
    for attr in [
        "specialty",
        "subspecialty",
        "speciality",
        "subspeciality",
        "last_specialty",
        "detected_specialties",
    ]:
        if hasattr(thread_local, attr):
            logger.info(f"🧹 Clearing {attr} from thread_local: {reason}")
            delattr(thread_local, attr)


# Setup detailed logging for debugging
def setup_detailed_logging():
    """
    Configure detailed logging with colorful console output
    for better visibility and debugging
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Create console handler with colorful formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Custom formatter with colors
    class ColorFormatter(logging.Formatter):
        """Custom formatter with colors for different log levels"""

        # ANSI color codes
        COLORS = {
            "INFO": Fore.CYAN,
            "DEBUG": Fore.WHITE,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.MAGENTA,
        }

        def format(self, record):
            # Get the original format string
            log_message = super().format(record)

            # Apply color formatting based on log level
            level_name = record.levelname
            color = self.COLORS.get(level_name, Fore.WHITE)

            # Format the timestamp and add color
            timestamp = datetime.datetime.fromtimestamp(record.created).strftime(
                "%H:%M:%S"
            )

            # Apply colorful formatting
            module_part = f"{Style.DIM}{record.name}{Style.RESET_ALL}"
            level_part = f"{color}{level_name}{Style.RESET_ALL}"
            message_part = record.getMessage()

            # Add emoji prefix based on content
            emoji = "🔄"  # Default
            if "ERROR" in level_name:
                emoji = "❌"
            elif "SYMPTOM" in message_part:
                emoji = "🩺"
            elif "DOCTOR" in message_part:
                emoji = "🔍"
            elif "PATIENT" in message_part:
                emoji = "👤"
            elif "CALL" in message_part:
                emoji = "📞"
            elif "RESPONSE" in message_part:
                emoji = "📤"
            elif "TOOL" in message_part:
                emoji = "🔧"

            # Create final colored output
            return f"{Fore.GREEN}[{timestamp}]{Style.RESET_ALL} {emoji} {level_part} {module_part}: {message_part}"

    # Set formatter on console handler
    console_handler.setFormatter(ColorFormatter("%(message)s"))

    # Add handler to logger
    root_logger.addHandler(console_handler)

    # Print start of application with colorful banner
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*30} MEDICAL ASSISTANT CHAT {'='*30}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

    root_logger.info("🚀 Starting Medical Assistant Chat with detailed console logging")

    return root_logger


# Get database instance for doctor searches
from .db import DB

db_instance = DB()

# Internal utils and helpers
from .utils import (
    store_patient_details as utils_store_patient_details,
    get_language_code,
)

# Load environment variables
load_dotenv()

# Set the OpenAI API key - Try different environment variable names
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise ValueError(
        "OpenAI API key not found. Please set OPENAI_API_KEY or API_KEY environment variable."
    )

os.environ["OPENAI_API_KEY"] = api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = (
    "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"  # Exposed intentionally
)

colorama.init(autoreset=True)

# Initialize logger with detailed settings
logger = setup_detailed_logging()


UNIFIED_MEDICAL_ASSISTANT_PROMPT = """
📋 Dsmart AI System Prompt (Strictly Additive & Reorganized)

You are an intelligent, warm, and multilingual medical assistant named "Dsmart AI" for the Middle East. Help users find doctors using GPS location from our database and datasource only. Support Arabic, English, Roman Urdu, and Urdu script. Respond in the user's exact language and script with a respectful and caring tone.

🌐 LANGUAGE & TONE HANDLING

- Respond in the user's dominant language. If multiple languages are mixed, default to Arabic unless explicitly requested otherwise.
- Maintain tone:
  - Arabic: Respectful and formal
  - English: Warm and friendly
  - Roman Urdu / Urdu: Polite and caring

🎯 CORE MISSION

- Connect users with doctors based on their needs using registered tools only. Never imagine or use external sources.
- Politely remind users you are not a doctor, only a connector to doctors. For emergencies, guide them to contact emergency services immediately.

👋 INITIAL CONVERSATION FLOW (CRITICAL)

- **First Priority**: Start EVERY conversation with a friendly greeting and ask for the user's name, followed by their age. Call `store_patient_details` immediately after receiving name AND age.
- Update with the latest information if user provides contradictory or new details (e.g., new age, name, symptoms).
- If input is unclear (e.g., "I feel bad"), ask clarifying questions in the same language.
- If user requests information about a health concern, provide detailed information and execute [analyze_symptoms] and then [search_doctors_dynamic: specialty=..., subspecialty=...] with relevant parameters from [analyze_symptoms].
- **Exception**: If user starts with a direct doctor or offers request (e.g., "find me dentists," "show me offers from Loran clinic"), skip name/age collection and execute `search_doctors_dynamic` with relevant parameters. Call `store_patient_details` later if name/age provided.

**Example Flow**:
- User: "Hi" or "مرحبا" or "سلام"
  - Response: "Hello! I'm here to help with your healthcare needs. May I know your name?"
- User: "Ali" or "علي"
  - Response: "Nice to meet you, Ali! Could you please tell me your age?"
- User: "25"
  - Response: [store_patient_details: Name="Ali", Age=25, Gender="Male"] "Thank you, Ali! How can I help you today?"
  
IMPORTANT TOOL EXECUTION RULES (CRITICAL):
 - Never show the tool execution in the response. You should only show the results not the internal execution call or process for the tool.
 - As soon as you know which tool to execute, execute it immediately and then respond with the results.
 - Dont give the response with the tool execution call or process for the tool.
 
IMPORTANT RESPONSE RULES (CRITICAL):
- Never say Let me find doctors for you or Please hold on a moment instead convert it to a interactable question, eg: Should I find doctors for you? etc

🛠️ TOOL SELECTION LOGIC

- **MANDATORY TOOL EXECUTION**: For any match to tool triggers, IMMEDIATELY execute the tool using [TOOL_NAME: param1=value1, param2=value2] format at the start of internal reasoning. Only respond naturally AFTER receiving tool results. Never use delaying language, announce actions, or ask for confirmation unless explicitly stated.
- Use context from the final user prompt to decide tool parameters. If unclear, use the last 2–3 prompts only for context related to doctor/clinic/offer mentions.
- If user provides symptoms + doctor request (e.g., "I have gum pain, find me a dentist"), prioritize `search_doctors_dynamic` with specialty/subspecialty over redundant symptom analysis.

**When to Call Each Tool**:

- **`store_patient_details`**:
  - Call when user provides name AND age (e.g., "I am Hammad and 23 years old" , "أنا حماد وعمري 23 سنة") or new personal information (e.g., "I'm 25 now", "أنا الآن 25 سنة").
  - Never call if patient info is already complete.

- **`analyze_symptoms`**:
  - Call when user describes new symptoms (e.g., "I have gum pain", "لدي ألم في اللثة"), health concerns, signs, or procedures.
  - Call when user asks for information about a health issue (e.g., "Tell me about braces", "أخبرني عن تقويم الأسنان").
  - If symptoms are unclear, ask one clarifying question in the same language.
  - Never call if specialty and subspecialty are already detected for the current issue or if user confirms a doctor search (e.g., "yes", "أجل", "أوكي", "يمكن", "نعم"), but if user asks for information about a health issue, call this tool.
  - After receiving specialty/subspecialty, execute the `search_doctors_dynamic` tool with the specialty and subspecialty in params.

- **`search_doctors_dynamic`**:
  - Execute IMMEDIATELY when user requests doctors by specialty, subspecialty, clinic, or name (e.g., ", ,"ابحث لي عن أطباء تقويم الأسنان", "ابحث عن الدكتور عمر", "أطباء ذكور فقط").
  - Execute when user confirms a search after symptom analysis (e.g., "yes," "okay", "please" , "أجل", "أوكي", "يمكن", "نعم") using specialty/subspecialty from `analyze_symptoms` tool result.
  - Also Execute with user message param when user asks questions like "What is the specialty of Dr. Omar?" or "Is Dr Omar a good dentist?" or "Is Dr Omar involved with you?".
  - Use age to select appropriate subspecialty (e.g., for a child, use specialty: "Dentistry", subspecialty: "Pediatric Dentistry").
  - If user requests "other options," use the second subspecialty from `analyze_symptoms` tool result from your context.
  - If no results found for specific doctor/clinic, execute again with only location (lat/long) and respond: "I couldn’t find your exact request, but here are other doctors near you."
  - For "doctors near me" or "offers near me" or "أطباء بالقرب مني" or "عروض بالقرب مني" without specifics, execute with location only.
  - For booking/appointment mentions, execute with relevant parameters and say: "Use the 'Book Appointment', "احجز موعدًا" button on the doctor card to book in the chat or visit dsmart.ai for booking."
  - Never execute without specialty unless it’s a direct location-based request.

🔄 CONVERSATION FLOW HANDLING

- **Scenario 1: Direct Doctor Search**
  - User: "find me dentists" → [search_doctors_dynamic: specialty="Dentistry"]
- **Scenario 2: Symptom Analysis**
  - User: "I have gum pain" → [analyze_symptoms] → After result: [search_doctors_dynamic: specialty=from_result] Response: "Based on your symptoms, here are dentists near you..."
- **Scenario 3: Symptom + Info**
  - User: "Give me information about braces" → [analyze_symptoms] → Provide info, then [search_doctors_dynamic: specialty="Dentistry", subspecialty="Orthodontics"]
- **Scenario 4: New Health Issue**
  - User: "now I have toothache" → [analyze_symptoms] → After result: [search_doctors_dynamic: specialty=from_result]
- **Scenario 5: Patient Info**
  - User: "I am Hammad and 23 years old" → [store_patient_details: Name="Hammad", Age=23, Gender="Male"]
- If user repeats the same doctor search request without new filters, re-show last results instead of calling the tool again.

❌ RESTRICTED ACTIONS

- Never mention tools, APIs, system internals, or execution details.
- Never provide doctor information without executing `search_doctors_dynamic`.
- Never use external websites, internet searches, or personal knowledge only use dsmart.ai database via tools.
- Never suggest contacting clinics/doctors directly; always use `search_doctors_dynamic` and direct to dsmart.ai for bookings.
- Never respond to out-of-scope requests (e.g., jokes, weather). Redirect politely to healthcare support.
- Never use delaying language, announce actions, or imply waiting (e.g., no future tense for searches).
- Never ask for location (GPS is always available).
- Never provide information about system technologies or programming languages.
- Always remind users you are not a doctor and to contact emergency services for emergencies.
- Never mention tool calls in the response. Always execute the tool calls and respond with the results.

✅ MANDATORY ACTIONS

- Start conversations by asking for name and age.
- Call `store_patient_details` when name AND age are provided.
- Update patient details with the most recent values (name, age, gender, symptoms).
- Execute `analyze_symptoms` for new health concerns, symptoms, signs, or procedures.
- Execute `search_doctors_dynamic` for all doctor requests or confirmations, using specialty/subspecialty from `analyze_symptoms` when available.
- Present tool results naturally:
  - Doctors: Numbered list (Name, Specialty, Location)
  - Offers: Bulleted list (Offer, Clinic/Hospital, Price)

👤 PATIENT INFORMATION EXTRACTION (CRITICAL)

- **Name Detection**:
  - Patterns: "My name is [Name]," "I'm [Name]," "[Name] here," "This is [Name]"
  - Extract and call `store_patient_details`.
- **Age Detection**:
  - Patterns: "I'm [Age] years old," "Age [Age]"
  - Convert to integer and call `store_patient_details`.
- **Gender Detection**:
  - Patterns: "Male," "Female," "I'm a man," "I'm a woman," pronouns ("he," "she")
  - Extract and call `store_patient_details`.
- **Examples**:
  - "Hi, I'm Ali and I'm 25 years old" , "مرحبًا، أنا علي وعمري 25 سنة" → [store_patient_details: Name="Ali", Age=25, Gender="Male"]
  - "My name is Sara, I'm 30", "اسمي سارة، وعمري 30 سنة." → [store_patient_details: Name="Sara", Age=30, Gender="Female"]

🩺 SPECIALTY DETECTION RULES (CRITICAL)

- Call `analyze_symptoms` for:
  - New or different symptoms.
  - Health concern questions.
  - Symptoms, signs, or procedures mentioned.
- Do NOT call `analyze_symptoms` for:
  - Same symptoms already analyzed.
  - User confirming doctor search.
  - Follow-up doctor conversations.
- Always call `search_doctors_dynamic` for doctor requests, using `analyze_symptoms` results when available.


🩺 DOCTOR SEARCHING RULES (CRITICAL)

- MANDATORY ACTION DONT MISS: ALWAYS execute the search_doctors_dynamic tool for doctor searches.
-Immediately execute [search_doctors_dynamic: specialty=..., subspecialty=...] for:
-Any doctor request (e.g., "find me dentists," "find Dr. Omar," "male doctors only", "ابحث لي عن أطباء طب الأسنان", "ابحث عن الدكتور عمر", "أطباء ذكور فقط").
-Confirmations after symptom analysis (e.g., "yes," "okay," "please" , "أجل", "أوكي", "يمكن", "نعم") using specialty/subspecialty from [analyze_symptoms].
-Health concern information requests (e.g., "Tell me about braces", "أخبرني عن تقويم الأسنان") after providing information, using [analyze_symptoms] results.
-Execute silently without announcing the search or implying waiting. Respond ONLY with formatted results (e.g., "Here are dentists near you: [list results].") or a clarifying question if input is ambiguous.
-If no results found, execute again with location only and respond: "I couldn’t find your exact request, but here are other doctors near you: [list results]."
- If no results in a list found, dont respond with doctor information instead respond with a question of finding some other options related to the search.
-For bookings/appointments, execute with relevant parameters and include: "Use the 'Book Appointment' button on dsmart.ai to book."
-Never include tool calls or execution details in user-facing responses.


⚠️ FINAL RESPONSE RULES

- Responses must be natural, user-facing, and free of technical details or tool mentions.
- Always assume tool results are available and respond with them.
- Never break role or respond to non-medical queries.
- Ensure all tool calls use [TOOL_NAME: params] format internally before responding.
"""

class ChatHistory:
    def __init__(self):
        self.messages = []
        self.patient_data = None
        self.temp_search_criteria = None
        self.symptom_analysis = None  # Add storage for symptom analysis
        self.tool_execution_history = []  # Track tool executions
        self.last_tool_result = None  # Store the most recent tool result
        self.specialty_data = None  # Explicitly track specialty data
        self.subspecialty_data = None  # Explicitly track subspecialty data
        
        # Ensure all attributes are properly initialized
        self._ensure_attributes_exist()
    
    def _ensure_attributes_exist(self):
        """Ensure all required attributes exist and are properly initialized"""
        required_attrs = {
            'symptom_analysis': None,
            'specialty_data': None,
            'subspecialty_data': None,
            'patient_data': None,
            'temp_search_criteria': None,
            'tool_execution_history': [],
            'last_tool_result': None
        }
        
        for attr, default_value in required_attrs.items():
            if not hasattr(self, attr):
                setattr(self, attr, default_value)
                logger.debug(f"🔧 Initialized missing attribute: {attr} = {default_value}")

    def add_user_message(self, content: str):
        self.messages.append({"type": "human", "content": content})

    def add_ai_message(self, content: str, tool_calls: list = None):
        """Add an AI message to the chat history, optionally with tool calls"""
        message = {"type": "ai", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_tool_message(self, tool_name: str, content: str, tool_call_id: str = None):
        """Add a tool message to the chat history"""
        self.messages.append(
            {
                "type": "tool",
                "content": content,
                "name": tool_name,
                "tool_call_id": tool_call_id,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    def set_patient_data(self, data: dict):
        # Merge with existing data rather than replacing
        if self.patient_data is None:
            self.patient_data = {}

        # Update only non-None values
        for key, value in data.items():
            if value is not None:
                self.patient_data[key] = value

    def get_patient_data(self):
        return self.patient_data

    def set_symptom_analysis(self, analysis: dict):
        """Store symptom analysis results"""
        try:
            if not hasattr(self, 'symptom_analysis'):
                self.symptom_analysis = None
                logger.debug("🔧 Initialized symptom_analysis attribute")
            
            self.symptom_analysis = analysis

            # Record this execution in history
            self.add_tool_execution("analyze_symptoms", analysis)
        except Exception as e:
            logger.error(f"❌ Error setting symptom analysis: {e}")
            # Ensure the attribute exists
            self.symptom_analysis = analysis

    def get_symptom_analysis(self):
        """Get stored symptom analysis"""
        try:
            if hasattr(self, 'symptom_analysis'):
                return self.symptom_analysis
            else:
                logger.debug("ℹ️ symptom_analysis attribute not found, returning None")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Error getting symptom analysis: {e}")
            return None

    def clear_symptom_analysis(self):
        """Clear symptom analysis results"""
        try:
            if hasattr(self, "symptom_analysis"):
                self.symptom_analysis = None
                logger.info("Cleared symptom analysis from history")
            else:
                logger.info("ℹ️ No symptom_analysis attribute to clear")
        except Exception as e:
            logger.warning(f"⚠️ Error clearing symptom analysis: {e}")
            # Ensure the attribute exists and is set to None
            self.symptom_analysis = None

    def add_tool_execution(self, tool_name: str, result: dict):
        """Track a tool execution in history"""
        execution = {
            "tool": tool_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "result": result,
        }
        self.tool_execution_history.append(execution)
        self.last_tool_result = result

    def get_last_tool_result(self):
        """Get the most recent tool result"""
        return self.last_tool_result

    def get_latest_doctor_search(self):
        """Get the most recent doctor search result with doctor data"""
        for execution in reversed(self.tool_execution_history):
            if (
                execution["tool"] == "search_doctors_dynamic"
                and "doctors" in execution["result"]
            ):
                return execution["result"]
        return None

    def clear_symptom_data(self, reason=""):
        """Clear all symptom and specialty related data from this chat history"""
        try:
            if reason:
                logger.info(f"ChatHistory: Clearing symptom data: {reason}")
            else:
                logger.info("ChatHistory: Clearing symptom data")

            # Safely clear attributes
            if hasattr(self, 'symptom_analysis'):
                self.symptom_analysis = None
            if hasattr(self, 'specialty_data'):
                self.specialty_data = None
            if hasattr(self, 'subspecialty_data'):
                self.subspecialty_data = None
                
        except Exception as e:
            logger.warning(f"⚠️ Error clearing symptom data: {e}")
            # Force clear attributes
            self.symptom_analysis = None
            self.specialty_data = None
            self.subspecialty_data = None

        # Also clear from the most recent tool result if it was a symptom analysis
        if (
            self.last_tool_result
            and isinstance(self.last_tool_result, dict)
            and (
                "symptom_analysis" in self.last_tool_result
                or "symptoms" in self.last_tool_result
            )
        ):
            logger.info("ChatHistory: Clearing symptom data from last_tool_result")
            self.last_tool_result = None

        # Also remove analyze_symptoms executions from the history to prevent confusion
        self.tool_execution_history = [
            execution
            for execution in self.tool_execution_history
            if execution["tool"] != "analyze_symptoms"
        ]

    def clear(self):
        try:
            self.messages = []
            self.patient_data = None
            self.temp_search_criteria = None
            self.symptom_analysis = None
            self.tool_execution_history = []
            self.last_tool_result = None
            self.specialty_data = None
            self.subspecialty_data = None
        except Exception as e:
            logger.warning(f"⚠️ Error clearing chat history: {e}")
            # Force clear by reinitializing
            self._ensure_attributes_exist()
            self.messages = []


# Store for chat histories
store = {}

# Initialize OpenAI client directly
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))


def detect_direct_search_criteria(user_message: str) -> dict:
    """
    Detect direct search criteria from user message like doctor names, clinic names, hospital names.
    Returns a dict with detected criteria or empty dict if none found.
    """
    if not user_message:
        return {}
    
    user_message_lower = user_message.lower()
    detected_criteria = {}
    
    # Common patterns for doctor names
    doctor_patterns = [
        r"dr\.?\s+([a-zA-Z\s]+)",  # Dr. Name or Dr Name
        r"doctor\s+([a-zA-Z\s]+)",  # Doctor Name
        r"find\s+me\s+dr\.?\s+([a-zA-Z\s]+)",  # find me Dr. Name
        r"looking\s+for\s+dr\.?\s+([a-zA-Z\s]+)",  # looking for Dr. Name
        r"search\s+for\s+dr\.?\s+([a-zA-Z\s]+)",  # search for Dr. Name
    ]
    
    for pattern in doctor_patterns:
        match = re.search(pattern, user_message_lower)
        if match:
            doctor_name = match.group(1).strip()
            if len(doctor_name) > 2:  # Avoid very short names
                detected_criteria["doctor_name"] = doctor_name
                break
    
    # Common patterns for clinic names
    clinic_patterns = [
        r"from\s+([a-zA-Z\s]+)\s+clinic",  # from X clinic
        r"at\s+([a-zA-Z\s]+)\s+clinic",  # at X clinic
        r"in\s+([a-zA-Z\s]+)\s+clinic",  # in X clinic
        r"([a-zA-Z\s]+)\s+clinic",  # X clinic
        r"clinic\s+([a-zA-Z\s]+)",  # clinic X
    ]
    
    for pattern in clinic_patterns:
        match = re.search(pattern, user_message_lower)
        if match:
            clinic_name = match.group(1).strip()
            if len(clinic_name) > 2:  # Avoid very short names
                detected_criteria["clinic_name"] = clinic_name
                break
    
    # Common patterns for hospital names
    hospital_patterns = [
        r"from\s+([a-zA-Z\s]+)\s+hospital",  # from X hospital
        r"at\s+([a-zA-Z\s]+)\s+hospital",  # at X hospital
        r"in\s+([a-zA-Z\s]+)\s+hospital",  # in X hospital
        r"([a-zA-Z\s]+)\s+hospital",  # X hospital
        r"hospital\s+([a-zA-Z\s]+)",  # hospital X
    ]
    
    for pattern in hospital_patterns:
        match = re.search(pattern, user_message_lower)
        if match:
            hospital_name = match.group(1).strip()
            if len(hospital_name) > 2:  # Avoid very short names
                detected_criteria["hospital_name"] = hospital_name
                break
    
    # Common patterns for medical centers
    center_patterns = [
        r"from\s+([a-zA-Z\s]+)\s+center",  # from X center
        r"at\s+([a-zA-Z\s]+)\s+center",  # at X center
        r"in\s+([a-zA-Z\s]+)\s+center",  # in X center
        r"([a-zA-Z\s]+)\s+center",  # X center
        r"center\s+([a-zA-Z\s]+)",  # center X
    ]
    
    for pattern in center_patterns:
        match = re.search(pattern, user_message_lower)
        if match:
            center_name = match.group(1).strip()
            if len(center_name) > 2:  # Avoid very short names
                detected_criteria["clinic_name"] = center_name  # Treat centers as clinics
                break
    
    return detected_criteria


def get_session_history(session_id: str) -> ChatHistory:
    if session_id not in store:
        logger.info(
            f"New session {session_id} created - Using cached specialty data with {len(SpecialtyDataCache.get_instance())} records"
        )

        # Important: Clear thread_local storage for the new session to prevent data leakage between sessions
        # This ensures specialty/subspecialty from previous session isn't carried over
        clear_symptom_analysis("starting new session", session_id)

        if hasattr(thread_local, "last_search_results"):
            logger.info(
                f"Clearing previous search results from thread_local for new session"
            )
            delattr(thread_local, "last_search_results")

        if hasattr(thread_local, "extracted_criteria"):
            logger.info(
                f"Clearing previous extracted_criteria from thread_local for new session"
            )
            delattr(thread_local, "extracted_criteria")

        # Set the current session_id in thread_local
        thread_local.session_id = session_id

        # Create a new history for this session
        store[session_id] = ChatHistory()
        
        # Ensure the new instance is properly initialized
        try:
            store[session_id]._ensure_attributes_exist()
            logger.debug(f"✅ ChatHistory attributes verified for session {session_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify ChatHistory attributes: {e}")

    history = store[session_id]
    
    # Ensure existing history has all required attributes
    try:
        if hasattr(history, '_ensure_attributes_exist'):
            history._ensure_attributes_exist()
            logger.debug(f"✅ Existing ChatHistory attributes verified for session {session_id}")
    except Exception as e:
        logger.warning(f"⚠️ Could not verify existing ChatHistory attributes: {e}")

    # Debug log to show history state
    if history.messages:
        logger.debug(
            f"Session {session_id} has {len(history.messages)} previous messages"
        )
        if history.get_patient_data():
            logger.info(
                f"Found existing patient data for session {session_id}: {history.get_patient_data()}"
            )
        else:
            logger.debug(f"No patient data found for session {session_id}")
    else:
        logger.debug(f"New conversation for session {session_id}")

    return history

def format_tools_for_openai():
    """Format tools for OpenAI API in the required structure"""
    logger.info("🔧 Setting up tools for OpenAI API")

    # Tool definitions - cleaner approach with consistent descriptions
    tool_definitions = {
        "search_doctors_dynamic": {
            "description": "Search for doctors based on user criteria.) Tool parameters (speciality/subspeciality) detected from analyze_symptoms tool result) If not provided, extracts from user message. CRITICAL: This tool MUST be called for doctor searches. NEVER provide direct responses - ALWAYS use this tool.",
            "params": {
                "user_message": "The user's search request in natural language",
                "latitude": "Latitude coordinate for location-based search (float)",
                "longitude": "Longitude coordinate for location-based search (float)",
                "speciality": "Speciality of the doctor (string, optional) - HIGHEST PRIORITY",
                "subspeciality": "Sub-speciality of the doctor (string, optional) - HIGHEST PRIORITY",
            },
            "required": ["user_message", "latitude", "longitude"],
        },
        "store_patient_details": {
            "description": "Store patient information in the session. CRITICAL: Call this tool IMMEDIATELY whenever any patient details are provided (name, age, gender, location, symptoms). This should typically be the FIRST tool in the flow. You MUST provide at least one of: Name, Age, Gender, Location, or Issue. DO NOT include session_id - it will be handled automatically. VALID FIELDS ONLY: Name, Age, Gender, Location, Issue. DO NOT send any other fields. EXAMPLES: 'i am hammad and 23 years old' → Call with Name='hammad', Age=23, Gender='Male'",
            "params": {
                "Name": "Name of the patient (string, optional but recommended)",
                "Age": "Age of the patient (integer, optional but recommended)",
                "Gender": "Gender of the patient (Male/Female, optional)",
                "Location": "Location/city of the patient (string, optional)",
                "Issue": "The health concerns or symptoms of the patient (string, optional)",
            },
            "required": [],
        },
        "analyze_symptoms": {
            "description": "Analyze patient symptoms to match with appropriate medical specialties. CRITICAL: Call this tool when (1) user describes NEW symptoms that haven't been analyzed yet, OR (2) user describes DIFFERENT symptoms from what was previously analyzed. If the user describes the EXACT SAME symptoms that were already analyzed, DO NOT call this tool again. This tool will automatically clear previous specialty data and analyze the new symptoms to provide updated medical recommendations.",
            "params": {
                "symptom_description": "Description of symptoms or health concerns"
            },
            "required": ["symptom_description"],
        },
    }

    # Convert tool definitions to OpenAI format
    tools = []
    for tool_name, definition in tool_definitions.items():
        properties = {}
        for param_name, description in definition["params"].items():
            if param_name == "Age":
                param_type = "integer"
            elif param_name in ["latitude", "longitude"]:
                param_type = "number"
            else:
                param_type = "string"
            properties[param_name] = {"type": param_type, "description": description}

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": definition["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": definition["required"],
                    },
                },
            }
        )

    logger.info(
        f"🔧 Registered {len(tools)} tools: {[t['function']['name'] for t in tools]}"
    )
    return tools


class SimpleMedicalAgent:
    def __init__(self):
        """Initialize the chat engine with tools and system settings"""
        logger.info("🚀 Initializing SimpleMedicalAgent")
        self.tools = format_tools_for_openai()
        self.messages_by_session = {}  # Track message history by session
        logger.info("✅ SimpleMedicalAgent initialized successfully")

        def get_llm(self):
            # Return the language model to use for standard interactions
            return client

    def sync_session_history(self, session_id):
        """Synchronize OpenAI messages with session history"""
        try:
            logger.info(f"SYNC: Starting history sync for session {session_id}")
            history = get_session_history(session_id)

            # Initialize messages for this session if not exists
            if session_id not in self.messages_by_session:
                self.messages_by_session[session_id] = [
                    {"role": "system", "content": UNIFIED_MEDICAL_ASSISTANT_PROMPT}
                ]

            messages = self.messages_by_session[session_id]

            # Log current state
            logger.info(f"SYNC: Current OpenAI messages: {len(messages)}")
            logger.info(f"SYNC: Current history messages: {len(history.messages)}")
            logger.info(
                "SYNC: OpenAI message types: "
                + ", ".join([f"{m['role']}" for m in messages])
            )
            logger.info(
                "SYNC: History message types: "
                + ", ".join([f"{m['type']}" for m in history.messages])
            )

            # Count messages by type
            openai_counts = {"user": 0, "assistant": 0, "system": 0, "tool": 0}
            history_counts = {"human": 0, "ai": 0, "system": 0, "tool": 0}

            for msg in messages:
                msg_role = msg["role"]
                if isinstance(msg.get("tool_calls"), list):
                    openai_counts["tool"] += 1
                else:
                    openai_counts[msg_role] += 1

            for msg in history.messages:
                history_counts[msg["type"]] += 1

            logger.info(f"SYNC: OpenAI message counts: {openai_counts}")
            logger.info(f"SYNC: History message counts: {history_counts}")

            # Check for mismatch
            if len(messages) != len(history.messages) + 1:  # +1 for system message
                logger.warning(
                    f"SYNC: Message count mismatch - OpenAI: {len(messages)}, History: {len(history.messages)}"
                )

                # Log the actual messages for comparison
                logger.info("SYNC: OpenAI Messages:")
                for i, msg in enumerate(messages):
                    content_preview = ""
                    if msg.get("content"):
                        content_preview = msg["content"][:50] + "..."
                    elif msg.get("tool_calls"):
                        tool_calls = msg["tool_calls"]
                        if isinstance(tool_calls, list):
                            content_preview = f"Tool calls: {[t.function.name if hasattr(t, 'function') else t['function']['name'] for t in tool_calls]}"
                        else:
                            content_preview = (
                                "Tool calls present but not in expected format"
                            )
                    logger.info(f"  {i}: {msg['role']} - {content_preview}")

                logger.info("SYNC: History Messages:")
                for i, msg in enumerate(history.messages):
                    logger.info(
                        f"  {i}: {msg['type']} - {msg.get('content', '')[:50]}..."
                    )

                # Rebuild messages from history
            new_messages = [
                {"role": "system", "content": UNIFIED_MEDICAL_ASSISTANT_PROMPT}
            ]

            for msg in history.messages:
                if msg["type"] == "human":
                    new_messages.append({"role": "user", "content": msg["content"]})
                elif msg["type"] == "ai":
                    if msg.get("tool_calls"):
                        # Handle tool calls in the correct format
                        tool_calls = msg["tool_calls"]
                        formatted_tool_calls = []
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                formatted_tool_calls.append(
                                    {
                                        "id": tool_call.get("id", str(uuid.uuid4())),
                                        "type": "function",
                                        "function": {
                                            "name": tool_call["function"]["name"],
                                            "arguments": tool_call["function"][
                                                "arguments"
                                            ],
                                        },
                                    }
                                )
                        new_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": formatted_tool_calls,
                            }
                        )
                    else:
                        new_messages.append(
                            {"role": "assistant", "content": msg["content"]}
                        )
                elif msg["type"] == "tool":
                    new_messages.append(
                        {
                            "role": "tool",
                            "content": msg.get("content"),
                            "tool_call_id": msg.get("tool_call_id"),
                            "name": msg.get("name"),
                        }
                    )

            # Update the session messages
            self.messages_by_session[session_id] = new_messages
            logger.info(f"SYNC: Rebuilt messages - new count: {len(new_messages)}")
            return self.messages_by_session[session_id]

            return messages
        except Exception as e:
            logger.error(
                f"SYNC ERROR: Failed to sync session history: {str(e)}", exc_info=True
            )
            # Return default messages if sync fails
            return [{"role": "system", "content": UNIFIED_MEDICAL_ASSISTANT_PROMPT}]

    def add_message_to_history(self, session_id: str, message: dict):
        """Helper method to add a message to both OpenAI messages and chat history"""
        try:
            # Get current messages
            if session_id not in self.messages_by_session:
                self.messages_by_session[session_id] = [
                    {"role": "system", "content": UNIFIED_MEDICAL_ASSISTANT_PROMPT}
                ]
            messages = self.messages_by_session[session_id]

            # Get history
            history = get_session_history(session_id)

            # Add to appropriate history based on role/type
            if message["role"] == "user":
                history.add_user_message(message["content"])
                messages.append(message)
            elif message["role"] == "assistant":
                if message.get("tool_calls"):
                    # Store tool calls in history
                    history.add_ai_message(
                        message["content"] or "", tool_calls=message["tool_calls"]
                    )
                else:
                    history.add_ai_message(message["content"])
                messages.append(message)
            elif message["role"] == "tool":
                history.add_tool_message(
                    message["name"], message["content"], message.get("tool_call_id")
                )
                messages.append(message)

            # Update session messages
            self.messages_by_session[session_id] = messages

        except Exception as e:
            logger.error(f"Error adding message to history: {str(e)}", exc_info=True)

    def process_message(
        self,
        message: str,
        session_id: str,
        lat: Optional[float] = None,
        long: Optional[float] = None,
    ) -> dict:
        """Process a user message and return a response - API entry point"""
        logger.info(f"📩 Processing message via API: {message}")
        logger.info(f"🔍 DEBUG: process_message received lat={lat}, long={long}")

        # Convert to the format expected by invoke method
        data = {"input": message, "session_id": session_id, "lat": lat, "long": long}
        logger.info(f"🔍 DEBUG: data dict created: {data}")

        return self.invoke(data)

    def invoke(self, data: dict, session_id: str = None) -> dict:
        """
        Main method to process user messages and generate responses
        """
        try:
            # CRITICAL: Log current history state for debugging
            history = get_session_history(session_id)
            logger.info("=" * 80)
            logger.info("🔄 NEW MESSAGE PROCESSING - CURRENT HISTORY STATE:")
            logger.info("=" * 80)
            
            # Log patient data
            patient_data = history.get_patient_data()
            logger.info(f"👤 PATIENT DATA: {json.dumps(patient_data, indent=2) if patient_data else 'None'}")
            
            # Log tool execution history
            tool_history = history.tool_execution_history
            logger.info(f"🔧 TOOL EXECUTION HISTORY ({len(tool_history)} entries):")
            for i, execution in enumerate(tool_history):
                tool_name = execution.get("tool", "Unknown")
                timestamp = execution.get("timestamp", "No timestamp")
                result_summary = str(execution.get("result", {}))[:200] + "..." if len(str(execution.get("result", {}))) > 200 else str(execution.get("result", {}))
                logger.info(f"  {i+1}. {tool_name} at {timestamp}: {result_summary}")
            
            # Log symptom analysis
            symptom_analysis = history.get_symptom_analysis()
            logger.info(f"🩺 SYMPTOM ANALYSIS: {json.dumps(symptom_analysis, indent=2) if symptom_analysis else 'None'}")
            
            # Log last tool result
            last_tool_result = history.get_last_tool_result()
            logger.info(f"📋 LAST TOOL RESULT: {json.dumps(last_tool_result, indent=2) if last_tool_result else 'None'}")
            
            # Log message history count
            message_count = len(history.messages)
            logger.info(f"💬 MESSAGE HISTORY COUNT: {message_count}")
            
            # Log detected specialties if available
            if patient_data and patient_data.get("detected_specialties"):
                specialties = patient_data["detected_specialties"]
                logger.info(f"🎯 DETECTED SPECIALTIES ({len(specialties)}): {json.dumps(specialties, indent=2)}")
            else:
                logger.info("🎯 DETECTED SPECIALTIES: None")
            
            logger.info("=" * 80)
            logger.info("🔄 END OF HISTORY STATE LOG")
            logger.info("=" * 80)
            
            # Extract input data
            user_message = data.get("input", "")
            session_id = data.get("session_id", "")
            lat = data.get("lat")
            long = data.get("long")

            logger.info(f"Processing message: '{user_message}' for session {session_id}")
            logger.info(f"🔍 DEBUG: lat={lat}, long={long} at start of invoke method")
            logger.info(f"🔍 DEBUG: data.get('lat') = {data.get('lat')}, data.get('long') = {data.get('long')}")
            logger.info(f"🔍 DEBUG: data keys: {list(data.keys())}")
            logger.info(f"🔍 DEBUG: data values: {list(data.values())}")

            try:
                # Main processing logic
                # This try block handles the main message processing
                # Get history for processing
                history = get_session_history(session_id)

                # Get or initialize message history for this session
                messages = self.sync_session_history(session_id)

                # Get patient data
                patient_data = history.get_patient_data()

                # Add user message to history
                self.add_message_to_history(
                    session_id, {"role": "user", "content": user_message}
                )

                # Let AI extract patient information if not already stored
                if not patient_data or not patient_data.get("Name"):
                    logger.info("🔍 AI will extract patient information from user message")
                    # Patient information will be extracted by AI through tools when needed

                # Let the main agent decide which tools to call based on the user message
                # No hardcoded tool calls - the agent will use tools as needed

                # Coordinates will be handled directly in tool call processing
                # The approach: intercept tool calls, correct coordinates if needed, and ensure
                # corrected values are used during actual tool execution

                # Now let the main agent handle everything through tools
                # This is the SINGLE AI call that handles everything
                logger.info("********************************")
                logger.info("CALLING SIMPLE MEDICAL AGENT")
                logger.info("********************************")
                logger.info(f"🔍 DEBUG: lat={lat}, long={long} before AI call")
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                )

                # Process the response
                response_message = response.choices[0].message

                # Check if the model wants to call a tool
                if response_message.tool_calls:
                    # Process tool calls
                    logger.info(
                        f"🔧 Model requested tool call: {response_message.tool_calls}"
                    )
                    logger.info(f"🔍 DEBUG: lat={lat}, long={long} before tool call processing")

                    # First add the assistant message with tool calls
                    messages.append(response_message.model_dump())

                    try:
                        for tool_call in response_message.tool_calls:
                            function_name = tool_call.function.name
                            try:
                                function_args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ Error parsing tool arguments: {str(e)}")
                                function_args = {}

                            # CRITICAL: Ensure coordinates are properly set for search_doctors_dynamic
                            if function_name == "search_doctors_dynamic" and lat is not None and long is not None:
                                logger.info(f"🔧 COORDINATE CORRECTION: Starting for {function_name}")
                                logger.info(f"🔧 COORDINATE CORRECTION: Expected coordinates: lat={lat}, long={long}")
                                
                                # Log original arguments for debugging
                                logger.info(f"🔧 Original search_doctors_dynamic arguments: {function_args}")
                                
                                # Force the correct coordinates if they're missing or incorrect
                                if "latitude" not in function_args or function_args["latitude"] == 0 or function_args["latitude"] == 0.0:
                                    function_args["latitude"] = lat
                                    logger.info(f"🔧 Fixed latitude from {function_args.get('latitude', 'missing')} to {lat}")
                                if "longitude" not in function_args or function_args["longitude"] == 0 or function_args["longitude"] == 0.0:
                                    function_args["longitude"] = long
                                    logger.info(f"🔧 Fixed longitude from {function_args.get('longitude', 'missing')} to {long}")
                                
                                # Log the corrected arguments
                                logger.info(f"🔧 Corrected search_doctors_dynamic arguments: {function_args}")
                                
                                # Final validation
                                if function_args.get("latitude") == lat and function_args.get("longitude") == long:
                                    logger.info(f"✅ Coordinates validated: lat={function_args['latitude']}, long={function_args['longitude']}")
                                else:
                                    logger.error(f"❌ Coordinate validation failed: expected lat={lat}, long={long}, got lat={function_args.get('latitude')}, long={function_args.get('longitude')}")
                                
                                # Update the tool call arguments to ensure the corrected values are used
                                tool_call.function.arguments = json.dumps(function_args)
                                logger.info(f"🔧 Updated tool call arguments with corrected coordinates")
                                
                                # IMPORTANT: Store corrected arguments for use in tool execution
                                # This ensures the corrected coordinates are used when the tool is actually executed
                                if not hasattr(tool_call, '_corrected_args'):
                                    tool_call._corrected_args = function_args.copy()
                                else:
                                    tool_call._corrected_args.update(function_args)
                                
                                logger.info(f"🔧 Stored corrected arguments in tool_call._corrected_args: {tool_call._corrected_args}")
                            else:
                                if function_name == "search_doctors_dynamic":
                                    logger.warning(f"⚠️ COORDINATE CORRECTION: Skipped - lat={lat}, long={long}")
                                else:
                                    logger.info(f"🔧 COORDINATE CORRECTION: Not needed for {function_name}")

                            # Display tool call header
                            tool_header = f"""
================================================================================
============================== TOOL CALL: {function_name.upper()} ==============================
================================================================================
"""
                            logger.info(tool_header)

                            # Process different tool types
                            if function_name == "store_patient_details":
                                logger.info(f"🔍 Processing store_patient_details")
                                logger.info(f"🔍 ORIGINAL function arguments from AI: {function_args}")
                                logger.info(f"🔍 ORIGINAL function arguments keys: {list(function_args.keys())}")
                                
                                # Check if detected_specialties is being sent by AI
                                if "detected_specialties" in function_args:
                                    logger.error(f"❌ CRITICAL: AI is sending 'detected_specialties' in store_patient_details call!")
                                    logger.error(f"❌ This should NOT happen - AI should only send: Name, Age, Gender, Location, Issue")
                                
                                logger.info(f"🔍 Function arguments received: {function_args}")
                                
                                # Validate and clean function arguments - remove ALL invalid keys including detected_specialties
                                valid_keys = ["Name", "Age", "Gender", "Location", "Issue"]
                                invalid_keys = []
                                for key in function_args.keys():
                                    if key not in valid_keys:
                                        invalid_keys.append(key)
                                        logger.warning(f"⚠️ Invalid argument '{key}' in store_patient_details - removing")
                                
                                # Remove invalid arguments
                                for key in invalid_keys:
                                    function_args.pop(key)
                                    logger.info(f"🔄 Removed invalid argument: {key}")
                                
                                logger.info(f"🔄 Cleaned function arguments: {function_args}")
                                
                                # CRITICAL: Ensure no session_id is passed to avoid duplicate argument error
                                if "session_id" in function_args:
                                    function_args.pop("session_id")
                                    logger.info(f"🔄 Removed session_id from function_args to prevent duplicate argument error")
                                
                                try:
                                    # Get existing patient data to preserve valid fields
                                    existing_data = history.get_patient_data() or {}

                                    # Ensure age is properly formatted as integer
                                    if (
                                        "Age" in function_args
                                        and function_args["Age"] is not None
                                    ):
                                        try:
                                            if isinstance(function_args["Age"], str):
                                                age_str = "".join(
                                                    filter(str.isdigit, function_args["Age"])
                                                )
                                                if age_str:
                                                    function_args["Age"] = int(age_str)
                                                else:
                                                    function_args["Age"] = None
                                            elif not isinstance(function_args["Age"], int):
                                                function_args["Age"] = None
                                        except (ValueError, TypeError) as e:
                                            logger.info(
                                                f"Error converting age: {str(e)}, setting to None"
                                            )
                                            function_args["Age"] = None

                                    # Merge with existing data, preserving valid fields
                                    merged_args = {**existing_data, **function_args}

                                    # Remove None values to preserve existing data
                                    for key in list(merged_args.keys()):
                                        if merged_args[key] is None and key in existing_data:
                                            merged_args[key] = existing_data[key]

                                    # Remove session_id from merged_args to avoid duplicate argument error
                                    if "session_id" in merged_args:
                                        merged_args.pop("session_id")
                                        logger.info(f"🔄 Removed session_id from merged_args to avoid duplicate argument error")
                                    
                                    # CRITICAL: Clean merged_args to remove any invalid fields that might have come from existing_data
                                    valid_keys = ["Name", "Age", "Gender", "Location", "Issue"]
                                    invalid_keys_in_merged = []
                                    for key in merged_args.keys():
                                        if key not in valid_keys:
                                            invalid_keys_in_merged.append(key)
                                            logger.warning(f"⚠️ Invalid key '{key}' found in merged_args - removing")
                                    
                                    # Remove invalid keys from merged_args
                                    for key in invalid_keys_in_merged:
                                        merged_args.pop(key)
                                        logger.info(f"🔄 Removed invalid key from merged_args: {key}")
                                    
                                    logger.info(f"🔄 Final cleaned merged_args: {merged_args}")
                                    
                                    # FINAL SAFETY CHECK: Ensure only valid arguments are passed
                                    final_args = {}
                                    for key, value in merged_args.items():
                                        if key in valid_keys:
                                            final_args[key] = value
                                        else:
                                            logger.error(f"❌ CRITICAL: Invalid key '{key}' still present in final_args - removing")
                                    
                                    logger.info(f"🔄 Final validated arguments: {final_args}")
                                    
                                    # Call store_patient_details with final validated arguments
                                    result = store_patient_details(
                                        session_id=session_id, **final_args
                                    )

                                    # Add tool result message immediately after the tool call
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "content": json.dumps(result),
                                            "tool_call_id": tool_call.id,
                                            "name": function_name,
                                        }
                                    )

                                    # Update history with patient data
                                    if result and isinstance(result, dict):
                                        history.set_patient_data(result)
                                        # Update the patient_data variable to reflect the new data
                                        patient_data = history.get_patient_data()
                                        logger.info(f"🔄 Updated patient_data after store_patient_details: {patient_data}")
                                        
                                        # CRITICAL: Add tool execution to history for context tracking
                                        history.add_tool_execution("store_patient_details", result)
                                        logger.info(f"🔄 Added store_patient_details tool execution to history")
                                        
                                        # Log updated history state after tool execution
                                        logger.info("=" * 80)
                                        logger.info("🔄 HISTORY STATE AFTER store_patient_details TOOL EXECUTION:")
                                        logger.info("=" * 80)
                                        updated_patient_data = history.get_patient_data()
                                        logger.info(f"👤 UPDATED PATIENT DATA: {json.dumps(updated_patient_data, indent=2) if updated_patient_data else 'None'}")
                                        updated_tool_history = history.tool_execution_history
                                        logger.info(f"🔧 UPDATED TOOL EXECUTION HISTORY ({len(updated_tool_history)} entries):")
                                        for i, execution in enumerate(updated_tool_history):
                                            tool_name = execution.get("tool", "Unknown")
                                            timestamp = execution.get("timestamp", "No timestamp")
                                            result_summary = str(execution.get("result", {}))[:200] + "..." if len(str(execution.get("result", {}))) > 200 else str(execution.get("result", {}))
                                            logger.info(f"  {i+1}. {tool_name} at {timestamp}: {result_summary}")
                                        logger.info("=" * 80)

                                except Exception as e:
                                    logger.error(
                                        f"❌ Error in store_patient_details: {str(e)}",
                                        exc_info=True,
                                    )
                                    
                                    # Create a fallback result to prevent conversation failure
                                    fallback_result = {
                                        "error": True,
                                        "error_message": str(e),
                                        "Name": function_args.get("Name"),
                                        "Age": function_args.get("Age"),
                                        "Gender": function_args.get("Gender"),
                                        "Location": function_args.get("Location"),
                                        "Issue": function_args.get("Issue"),
                                        "session_id": session_id
                                    }
                                    
                                    # Add fallback tool result message
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "content": json.dumps(fallback_result),
                                            "tool_call_id": tool_call.id,
                                            "name": function_name,
                                        }
                                    )
                                    
                                    logger.warning(f"⚠️ Added fallback result for failed store_patient_details tool call")

                            elif function_name == "analyze_symptoms":
                                logger.info(f"🏥 Analyzing symptoms: {function_args}")
                                
                                # ENHANCED: Check if symptoms are the same to prevent redundant analysis
                                current_symptoms = patient_data.get("symptom_description", "").lower() if patient_data else ""
                                new_symptoms = function_args.get("symptom_description", "").lower()
                                
                                # More intelligent symptom comparison - check if symptoms are substantially similar
                                symptoms_are_similar = False
                                if current_symptoms and new_symptoms:
                                    # Normalize symptoms for comparison
                                    current_normalized = current_symptoms.strip().replace(" ", "")
                                    new_normalized = new_symptoms.strip().replace(" ", "")
                                    
                                    logger.info(f"🔍 Symptom comparison: Current='{current_symptoms}' vs New='{new_symptoms}'")
                                    logger.info(f"🔍 Normalized comparison: Current='{current_normalized}' vs New='{new_normalized}'")
                                    
                                    # Check for exact match
                                    if current_normalized == new_normalized:
                                        symptoms_are_similar = True
                                        logger.info(f"🔍 Exact match detected - symptoms are similar")
                                    # Check if new symptoms are contained within current symptoms
                                    elif current_normalized in new_normalized or new_normalized in current_normalized:
                                        symptoms_are_similar = True
                                        logger.info(f"🔍 Containment detected - symptoms are similar")
                                    # Check for key symptom words - more lenient approach
                                    else:
                                        # Extract key medical terms for comparison
                                        current_words = set(current_normalized.lower().split())
                                        new_words = set(new_normalized.lower().split())
                                        
                                        # Remove common words that don't indicate different symptoms
                                        common_words_to_ignore = {'i', 'have', 'am', 'my', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'since', 'last', 'night', 'morning', 'evening', 'today', 'yesterday'}
                                        current_words = current_words - common_words_to_ignore
                                        new_words = new_words - common_words_to_ignore
                                        
                                        if current_words and new_words:
                                            common_words = current_words.intersection(new_words)
                                            similarity_ratio = len(common_words) / min(len(current_words), len(new_words)) if min(len(current_words), len(new_words)) > 0 else 0
                                            logger.info(f"🔍 Key words comparison: Current key words: {current_words}, New key words: {new_words}")
                                            logger.info(f"🔍 Word similarity: {len(common_words)} common words out of {min(len(current_words), len(new_words))} min words (ratio: {similarity_ratio:.2f})")
                                            if similarity_ratio >= 0.5:  # Lowered threshold to 50% for better detection
                                                symptoms_are_similar = True
                                                logger.info(f"🔍 High similarity detected - symptoms are similar")
                                            else:
                                                logger.info(f"🔍 Low similarity detected - symptoms are different")
                                        else:
                                            logger.info(f"🔍 No meaningful key words to compare - treating as different symptoms")
                                            symptoms_are_similar = False
                                else:
                                    logger.info(f"🔍 No previous symptoms to compare against - proceeding with analysis")
                                
                                # Log the final symptom comparison decision
                                logger.info(f"🔍 FINAL DECISION: Symptoms are {'SIMILAR' if symptoms_are_similar else 'DIFFERENT'}")
                                
                                # Check if this is a truly new/different symptom description
                                if (patient_data and patient_data.get("detected_specialties") and 
                                    symptoms_are_similar):
                                    logger.warning(f"⚠️ Same symptoms already analyzed: '{current_symptoms}'. Skipping redundant symptom analysis.")
                                    # Create a result indicating specialties are already available
                                    result = {
                                        "specialities_already_detected": True,
                                        "existing_specialties": patient_data["detected_specialties"],
                                        "message": "These symptoms have already been analyzed. Use stored specialties for doctor search.",
                                        "symptoms_detected": patient_data.get("symptom_description", "").split(", ") if patient_data.get("symptom_description") else [],
                                        "top_specialties": [spec.get("specialty", "") for spec in patient_data["detected_specialties"]],
                                        "detailed_analysis": {
                                            "status": "skipped_redundant",
                                            "message": "Skipped redundant symptom analysis - same symptoms already analyzed"
                                        }
                                    }
                                    
                                    # Add tool result message
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "content": json.dumps(result),
                                            "tool_call_id": tool_call.id,
                                            "name": function_name,
                                        }
                                    )
                                    logger.info(f"🔄 Skipped redundant symptom analysis - same symptoms already analyzed: {current_symptoms}")
                                    continue  # Skip to next tool call
                                elif patient_data and patient_data.get("detected_specialties"):
                                    logger.info(f"🔄 New/different symptoms detected: '{new_symptoms}' vs previous: '{current_symptoms}'. Proceeding with new analysis.")
                                    logger.info(f"🔄 Previous specialties will be cleared and replaced with new analysis results.")
                                else:
                                    logger.info(f"🔄 No previous specialties detected - proceeding with new symptom analysis")
                                
                                try:
                                    symptom_description = function_args.get("symptom_description", "")

                                    # ENHANCED: Clear previous specialty data when starting new symptom analysis
                                    if patient_data and patient_data.get("detected_specialties"):
                                        logger.info(
                                            f"🔄 Starting new symptom analysis - clearing previous {len(patient_data['detected_specialties'])} specialties"
                                        )
                                        logger.info(f"🔄 Previous specialties: {patient_data['detected_specialties']}")
                                        patient_data.pop("detected_specialties", None)
                                        patient_data.pop("last_symptom_analysis", None)
                                        patient_data.pop("symptom_description", None)
                                        history.set_patient_data(patient_data)
                                        logger.info(f"🔄 Cleared previous specialty data from patient_data")

                                    # Call symptom analysis
                                    symptom_result = analyze_symptoms(symptom_description)

                                    # Store symptom analysis in history
                                    history.set_symptom_analysis(symptom_result)

                                    # Add tool result immediately after the tool call
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "content": json.dumps(symptom_result),
                                            "tool_call_id": tool_call.id,
                                            "name": function_name,
                                        }
                                    )

                                    # Process symptom analysis results immediately
                                    speciality_not_available = False
                                    symptoms_text = ""
                                    specialties = []

                                    # Extract specialty availability status
                                    if (
                                        "speciality_not_available" in symptom_result
                                        and symptom_result["speciality_not_available"]
                                    ):
                                        speciality_not_available = True
                                    elif (
                                        "detailed_analysis" in symptom_result
                                        and "speciality_not_available"
                                        in symptom_result["detailed_analysis"]
                                    ):
                                        speciality_not_available = symptom_result["detailed_analysis"][
                                            "speciality_not_available"
                                        ]
                                    elif (
                                        "detailed_analysis" in symptom_result
                                        and "symptom_analysis" in symptom_result["detailed_analysis"]
                                        and "speciality_not_available"
                                        in symptom_result["detailed_analysis"]["symptom_analysis"]
                                    ):
                                        speciality_not_available = symptom_result[
                                            "detailed_analysis"
                                        ]["symptom_analysis"]["speciality_not_available"]

                                    # Extract symptoms text
                                    if (
                                        "symptoms_detected" in symptom_result
                                        and symptom_result["symptoms_detected"]
                                    ):
                                        symptoms_text = ", ".join(symptom_result["symptoms_detected"])

                                    # Extract specialties
                                    if "specialties" in symptom_result:
                                        specialties = symptom_result["specialties"]
                                    elif "detailed_analysis" in symptom_result:
                                        da = symptom_result["detailed_analysis"]
                                        if "specialties" in da:
                                            specialties = da["specialties"]
                                        elif (
                                            "symptom_analysis" in da
                                            and "recommended_specialties" in da["symptom_analysis"]
                                        ):
                                            specialties = da["symptom_analysis"]["recommended_specialties"]

                                    # Specialties detected - let the main agent decide what to do with this information
                                    if specialties and len(specialties) > 0:
                                        # AI will handle specialty processing dynamically

                                        # ENHANCED: Store specialties in patient data for persistent context
                                        if not patient_data:
                                            patient_data = {"session_id": session_id}

                                        # Store detected specialties in patient data - deduplicate and clean
                                        cleaned_specialties = []
                                        seen_combinations = set()
                                        
                                        for specialty in specialties:
                                            specialty_name = specialty.get("specialty") or specialty.get("name", "")
                                            subspecialty_name = specialty.get("subspecialty") or specialty.get("subspeciality", "")
                                            
                                            # Create unique combination key
                                            combination = f"{specialty_name}:{subspecialty_name}"
                                            
                                            if combination not in seen_combinations:
                                                seen_combinations.add(combination)
                                                cleaned_specialty = {
                                                    "specialty": specialty_name,
                                                    "subspecialty": subspecialty_name,
                                                    "confidence": specialty.get("confidence", 0.0)
                                                }
                                                cleaned_specialties.append(cleaned_specialty)
                                        
                                        patient_data["detected_specialties"] = cleaned_specialties
                                        patient_data["last_symptom_analysis"] = time.time()
                                        patient_data["symptom_description"] = symptom_description
                                        
                                        logger.info(f"🔄 Stored {len(cleaned_specialties)} cleaned specialties: {cleaned_specialties}")
                                        logger.info(f"🔄 Updated symptom description to: {symptom_description}")
                                        logger.info(f"🔄 Updated patient_data with new specialties and cleared old ones")
                                    else:
                                        # Clear any existing specialties if none detected
                                        if patient_data:
                                            patient_data.pop("detected_specialties", None)
                                            patient_data.pop("last_symptom_analysis", None)
                                            patient_data.pop("symptom_description", None)

                                    # Update patient data in history
                                    history.set_patient_data(patient_data)
                                    
                                    # CRITICAL: Add tool execution to history for context tracking
                                    history.add_tool_execution("analyze_symptoms", symptom_result)
                                    logger.info(f"🔄 Added analyze_symptoms tool execution to history")

                                except Exception as e:
                                    logger.error(
                                        f"❌ Error in analyze_symptoms: {str(e)}",
                                        exc_info=True,
                                    )
                                    
                                    # Create a fallback result to prevent conversation failure
                                    fallback_result = {
                                        "error": True,
                                        "error_message": str(e),
                                        "symptom_description": function_args.get("symptom_description", ""),
                                        "speciality_not_available": True,
                                        "symptoms_detected": [],
                                        "top_specialties": []
                                    }
                                    
                                    # Add fallback tool result message
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "content": json.dumps(fallback_result),
                                            "tool_call_id": tool_call.id,
                                            "name": function_name,
                                        }
                                    )
                                    
                                    logger.warning(f"⚠️ Added fallback result for failed analyze_symptoms tool call")

                            elif function_name == "search_doctors_dynamic":
                                logger.info(f"🔍 TOOL EXECUTION: Starting search_doctors_dynamic")
                                logger.info(f"🔍 TOOL EXECUTION: Original function_args: {function_args}")

                                # Use corrected arguments if available (from coordinate correction)
                                if hasattr(tool_call, '_corrected_args'):
                                    corrected_args = tool_call._corrected_args
                                    logger.info(f"🔧 Using corrected arguments: {corrected_args}")
                                    # Use corrected coordinates
                                    latitude = corrected_args.get("latitude")
                                    longitude = corrected_args.get("longitude")
                                    user_message = corrected_args.get("user_message", function_args.get("user_message", ""))
                                    logger.info(f"🔧 TOOL EXECUTION: Using corrected coordinates: lat={latitude}, long={longitude}")
                                else:
                                    # Fallback to original arguments
                                    latitude = function_args.get("latitude")
                                    longitude = function_args.get("longitude")
                                    user_message = function_args.get("user_message", "")
                                    logger.warning(f"⚠️ TOOL EXECUTION: No corrected arguments found, using original: lat={latitude}, long={longitude}")

                                # Log the coordinates being used
                                logger.info(f"🔍 Using coordinates: lat={latitude}, long={longitude}")
                                
                                # Validate coordinates
                                if latitude is None or longitude is None:
                                    logger.error("❌ Missing coordinates for doctor search")
                                    result = {
                                        "error": "Coordinates are required for doctor search"
                                    }
                                elif latitude == 0 or latitude == 0.0 or longitude == 0 or longitude == 0.0:
                                    logger.error(f"❌ Invalid coordinates detected: lat={latitude}, long={longitude}")
                                    result = {
                                        "error": f"Invalid coordinates: lat={latitude}, long={longitude}. Expected non-zero values."
                                    }
                                else:
                                    try:
                                        # Create search criteria
                                        search_criteria = {
                                            "user_message": user_message,
                                            "latitude": float(latitude),
                                            "longitude": float(longitude),
                                        }
                                        
                                        logger.info(f"🔍 Initial search criteria created: {search_criteria}")
                                        logger.info(f"🔍 Original user request: '{user_message}'")

                                        # SIMPLIFIED 2-PRIORITY SYSTEM: Tool params > Message extraction
                                        # PRIORITY 1: Tool parameters (highest priority)
                                        if function_args.get("speciality"):
                                            search_criteria["speciality"] = function_args["speciality"]
                                            logger.info(f"🔍 PRIORITY 1: Using speciality from tool params: {function_args['speciality']}")

                                        if function_args.get("subspeciality"):
                                            search_criteria["subspeciality"] = function_args["subspeciality"]
                                            logger.info(f"🔍 PRIORITY 1: Using subspeciality from tool params: {function_args['subspeciality']}")

                                        # ALWAYS extract from user message to get ALL other criteria
                                        extracted_criteria = extract_search_criteria_from_message(user_message)
                                        logger.info(f"🔍 Extracted ALL criteria from message: {extracted_criteria}")

                                        # Add extracted criteria EXCEPT specialty/subspecialty (we already handled those above)
                                        for key, value in extracted_criteria.items():
                                            if key not in ["speciality", "subspeciality"]:  # Skip specialties, we already have them from tool params
                                                search_criteria[key] = value
                                                logger.info(f"🔍 Added extracted criteria '{key}': '{value}'")

                                        # PRIORITY 2: Only use extracted specialty/subspecialty if tool params didn't provide them
                                        if not search_criteria.get("speciality") and extracted_criteria.get("speciality"):
                                            search_criteria["speciality"] = extracted_criteria["speciality"]
                                            logger.info(f"🔍 PRIORITY 2: Using speciality from message extraction: {extracted_criteria['speciality']}")

                                        if not search_criteria.get("subspeciality") and extracted_criteria.get("subspeciality"):
                                            search_criteria["subspeciality"] = extracted_criteria["subspeciality"]
                                            logger.info(f"🔍 PRIORITY 2: Using subspecialty from message extraction: {extracted_criteria['subspeciality']}")

                                        # Check if user has direct search criteria (doctor name, clinic name, etc.)
                                        direct_search_criteria = detect_direct_search_criteria(user_message)
                                        logger.info(f"🔍 Direct search criteria detected: {direct_search_criteria}")
                                        
                                        # If direct search criteria found, add them to search parameters
                                        if direct_search_criteria:
                                            logger.info(f"🔍 Direct search criteria found - adding to search parameters")
                                            
                                            if direct_search_criteria.get("doctor_name"):
                                                search_criteria["doctor_name"] = direct_search_criteria["doctor_name"]
                                                logger.info(f"🔍 Added doctor name: {direct_search_criteria['doctor_name']}")
                                            
                                            if direct_search_criteria.get("clinic_name"):
                                                search_criteria["clinic_name"] = direct_search_criteria["clinic_name"]
                                                logger.info(f"🔍 Added clinic name: {direct_search_criteria['clinic_name']}")
                                            
                                            if direct_search_criteria.get("hospital_name"):
                                                search_criteria["hospital_name"] = direct_search_criteria["hospital_name"]
                                                logger.info(f"🔍 Added hospital name: {direct_search_criteria['hospital_name']}")

                                        logger.info(f"🔍 Final search criteria: {search_criteria}")
                                        logger.info(f"🔍 Message being sent to search: '{search_criteria['user_message']}'")
                                        logger.info(f"🔍 Direct search parameters: doctor_name={search_criteria.get('doctor_name')}, clinic_name={search_criteria.get('clinic_name')}, hospital_name={search_criteria.get('hospital_name')}")
                                        logger.info(f"🔍 Specialty parameters: speciality={search_criteria.get('speciality')}, subspecialty={search_criteria.get('subspecialty')}")

                                        # Call the doctor search function
                                        search_result = dynamic_doctor_search(
                                            json.dumps(search_criteria)
                                        )

                                        # Record the execution in history
                                        history.add_tool_execution(
                                            "search_doctors_dynamic", search_result
                                        )

                                        result = search_result
                                        logger.info(f"✅ Doctor search completed successfully")

                                    except Exception as e:
                                        logger.error(
                                            f"❌ Error in doctor search: {str(e)}",
                                            exc_info=True,
                                        )
                                        result = {"error": f"Doctor search failed: {str(e)}"}

                                # Add tool result message
                                messages.append(
                                    {
                                        "role": "tool",
                                        "content": json.dumps(result),
                                        "tool_call_id": tool_call.id,
                                        "name": function_name,
                                    }
                                )

                            else:
                                # Handle any other tool calls that weren't specifically handled
                                logger.warning(f"⚠️ Unhandled tool call: {function_name}")
                                try:
                                    # Try to call the tool with the provided arguments
                                    if function_name in globals():
                                        tool_func = globals()[function_name]
                                        result = tool_func(**function_args)
                                    else:
                                        result = {"error": f"Tool {function_name} not found"}
                                except Exception as e:
                                    logger.error(f"❌ Error in unhandled tool {function_name}: {str(e)}")
                                    result = {"error": f"Tool {function_name} failed: {str(e)}"}
                                
                                # Add tool result message
                                messages.append(
                                    {
                                        "role": "tool",
                                        "content": json.dumps(result),
                                        "tool_call_id": tool_call.id,
                                        "name": function_name,
                                    }
                                )

                            # Display tool completion footer
                            logger.info("=" * 80)

                    except Exception as e:
                        logger.error(f"❌ Error processing tool calls: {str(e)}", exc_info=True)
                        # Return error response if tool processing fails
                        return {
                            "response": {
                                "message": "I apologize, but I encountered an error while processing your request. Please try again.",
                                "patient": patient_data or {"session_id": session_id},
                                "data": [],
                                "error": True,
                                "error_details": str(e),
                            },
                            "display_results": False,
                        }

                    # Get updated context with latest doctor search results
                    updated_context_messages = self.sync_session_history(session_id)

                    # Process all tool results to build comprehensive context
                    doctor_search_result = None
                    has_doctor_results = False
                    doctor_data = []
                    offers_data = []

                    # Smart data extraction: Only use doctor data if this message actually performed a doctor search
                    # Check if the current message performed a doctor search by looking at the most recent tool execution
                    current_message_tools = []
                    if history.tool_execution_history:
                        # Get the most recent tool execution to see what was called in this message
                        most_recent_execution = history.tool_execution_history[-1]
                        current_message_tools = [most_recent_execution["tool"]]
                        logger.info(f"🔍 Current message tools: {current_message_tools}")
                    
                    if "search_doctors_dynamic" in current_message_tools:
                        # This message DID search for doctors - extract the results
                        logger.info("🔍 Current message performed doctor search - extracting results")
                        for execution in reversed(history.tool_execution_history):
                            if execution["tool"] == "search_doctors_dynamic":
                                doctor_search_result = execution["result"]
                                # Extract doctor data
                                if isinstance(doctor_search_result, dict):
                                    if "response" in doctor_search_result and isinstance(
                                        doctor_search_result["response"], dict
                                    ):
                                        response_data = doctor_search_result["response"]
                                        if "data" in response_data:
                                            data_field = response_data["data"]
                                            if (
                                                isinstance(data_field, dict)
                                                and "doctors" in data_field
                                            ):
                                                doctor_data = data_field["doctors"]
                                                has_doctor_results = len(doctor_data) > 0
                                            elif isinstance(data_field, list):
                                                doctor_data = data_field
                                                has_doctor_results = len(doctor_data) > 0
                                        elif "data" in doctor_search_result:
                                            if (
                                                isinstance(doctor_search_result["data"], dict)
                                                and "doctors" in doctor_search_result["data"]
                                            ):
                                                doctor_data = doctor_search_result["data"][
                                                    "doctors"
                                                ]
                                                has_doctor_results = len(doctor_data) > 0
                                            elif isinstance(doctor_search_result["data"], list):
                                                doctor_data = doctor_search_result["data"]
                                                has_doctor_results = len(doctor_data) > 0

                                    # Extract offers data
                                    if isinstance(doctor_search_result, dict):
                                        if "offers" in doctor_search_result:
                                            offers_data = doctor_search_result["offers"]
                                        elif "response" in doctor_search_result and isinstance(
                                            doctor_search_result["response"], dict
                                        ):
                                            if "offers" in doctor_search_result["response"]:
                                                offers_data = doctor_search_result["response"][
                                                    "offers"
                                                ]
                                break
                    else:
                        # This message did NOT search for doctors - don't include old results
                        logger.info("🔍 Current message did NOT perform doctor search - clearing old doctor data")
                        doctor_data = []
                        has_doctor_results = False
                        offers_data = []

                    # Get symptom analysis context
                    symptom_result = history.get_symptom_analysis()
                    symptom_context = {}
                    if symptom_result:
                        # Check different possible locations of the flag
                        if (
                            "speciality_not_available" in symptom_result
                            and symptom_result["speciality_not_available"]
                        ):
                            symptom_context["speciality_not_available"] = True
                        elif (
                            "detailed_analysis" in symptom_result
                            and "speciality_not_available"
                            in symptom_result["detailed_analysis"]
                        ):
                            symptom_context["speciality_not_available"] = symptom_result[
                                "detailed_analysis"
                            ]["speciality_not_available"]
                        elif (
                            "detailed_analysis" in symptom_result
                            and "symptom_analysis" in symptom_result["detailed_analysis"]
                            and "speciality_not_available"
                            in symptom_result["detailed_analysis"]["symptom_analysis"]
                        ):
                            symptom_context["speciality_not_available"] = symptom_result[
                                "detailed_analysis"
                            ]["symptom_analysis"]["speciality_not_available"]

                        # Get symptoms text
                        if (
                            "symptoms_detected" in symptom_result
                            and symptom_result["symptoms_detected"]
                        ):
                            symptom_context["symptoms_text"] = ", ".join(
                                symptom_result["symptoms_detected"]
                            )
                        elif (
                            history.get_patient_data()
                            and "Issue" in history.get_patient_data()
                        ):
                            symptom_context["symptoms_text"] = history.get_patient_data()[
                                "Issue"
                            ]

                    # Now we need to make a second AI call to generate the final response based on tool results
                    # This ensures the main agent can process the tool results and generate appropriate responses
                    logger.info(
                        "🔄 Making final AI call to generate response based on tool results"
                    )

                    # Get symptom analysis context to inform the main agent
                    symptom_result = history.get_symptom_analysis()
                    specialty_info = ""
                    specialties_detected = []

                    # Enhanced context extraction with better specialty detection
                    if symptom_result:
                        if "specialties" in symptom_result:
                            specialties_detected = symptom_result["specialties"]
                            if specialties_detected and len(specialties_detected) > 0:
                                specialty_names = []
                                for s in specialties_detected:
                                    specialty_name = s.get("specialty") or s.get(
                                        "name", "Unknown"
                                    )
                                    subspecialty_name = s.get("subspecialty") or s.get(
                                        "subspeciality", ""
                                    )
                                    if subspecialty_name:
                                        specialty_names.append(
                                            f"{specialty_name}/{subspecialty_name}"
                                        )
                                    else:
                                        specialty_names.append(specialty_name)
                                specialty_info = (
                                    f"Specialties detected: {', '.join(specialty_names)}. "
                                )
                        elif "detailed_analysis" in symptom_result:
                            da = symptom_result["detailed_analysis"]
                            if "specialties" in da:
                                specialties_detected = da["specialties"]
                                if specialties_detected and len(specialties_detected) > 0:
                                    specialty_names = []
                                    for s in specialties_detected:
                                        specialty_name = s.get("specialty") or s.get(
                                            "name", "Unknown"
                                        )
                                        subspecialty_name = s.get("subspecialty") or s.get(
                                            "subspeciality", ""
                                        )
                                        if subspecialty_name:
                                            specialty_names.append(
                                                f"{specialty_name}/{subspecialty_name}"
                                            )
                                        else:
                                            specialty_names.append(specialty_name)
                                    specialty_info = f"Specialties detected: {', '.join(specialty_names)}. "

                    # ENHANCED: Check patient data for previously detected specialties
                    if (
                        not specialties_detected
                        and patient_data
                        and patient_data.get("detected_specialties")
                    ):
                        specialties_detected = patient_data["detected_specialties"]
                        if specialties_detected and len(specialties_detected) > 0:
                            specialty_names = []
                            for s in specialties_detected:
                                specialty_name = s.get("specialty") or s.get(
                                    "name", "Unknown"
                                )
                                subspecialty_name = s.get("subspecialty") or s.get(
                                    "subspeciality", ""
                                )
                                if subspecialty_name:
                                    specialty_names.append(
                                        f"{specialty_name}/{subspecialty_name}"
                                    )
                                else:
                                    specialty_names.append(specialty_name)
                            specialty_info = (
                                f"Specialties detected: {', '.join(specialty_names)}. "
                            )
                            logger.info(
                                f"🔍 Retrieved specialties from patient data: {specialty_names}"
                            )

                    # Enhanced context-aware decision making
                    should_search_doctors = False
                    context_aware_search = False

                    if (
                        specialties_detected
                        and len(specialties_detected) > 0
                        and not has_doctor_results
                    ):
                        # Analyze user's last message to determine intent
                        last_user_message = ""
                        for msg in reversed(messages):
                            if msg["role"] == "user":
                                last_user_message = msg["content"].lower()
                                break

                        # AI will determine doctor search intent dynamically

                        # AI will handle specialty detection context dynamically
                        previous_specialty_detection = False
                        if patient_data and patient_data.get("detected_specialties"):
                            previous_specialty_detection = True

                        # AI will make the decision dynamically based on context
                        should_search_doctors = True  # Let AI decide through tools
                        context_aware_search = previous_specialty_detection

                        if context_aware_search:
                            # AI will handle context-aware decisions dynamically

                            # AI will handle specialty extraction dynamically

                            # AI will handle doctor search through tools when needed
                            logger.info("🔍 AI will handle doctor search through tools")
                        
                        # AI will handle context-aware decisions dynamically

                    # Enhanced context message with conversation flow and detected specialties
                    detected_specialties = patient_data.get('detected_specialties', []) if patient_data else []
                    specialty_summary = ""
                    if detected_specialties:
                        specialty_names = []
                        for spec in detected_specialties:
                            specialty_name = spec.get('specialty', '')
                            subspecialty_name = spec.get('subspecialty', '')
                            if specialty_name and subspecialty_name:
                                specialty_names.append(f"{specialty_name} ({subspecialty_name})")
                            elif specialty_name:
                                specialty_names.append(specialty_name)
                        specialty_summary = ", ".join(specialty_names)
                        
                        
                    final_context_message_prompt = f"""Generate a helpful response based on the conversation context and tool results.

🎯 CONVERSATION FLOW CONTEXT:
- User has already provided symptoms: {bool(patient_data and patient_data.get('symptom_description'))}
- Specialties have been analyzed: {bool(detected_specialties)}
- Current request: {data.get('input', 'Unknown')}
- **Current State**: {'Ready for doctor search' if detected_specialties else 'Need symptom analysis'}
- **User Intent**: {'Confirming doctor search' if data.get('input', '').lower() in ['yes', 'okay', 'please', 'sure', 'ok'] else 'Other request'}

🏥 MEDICAL CONTEXT:
- Detected Specialties: {specialty_summary if specialty_summary else 'None detected yet'}
- Symptoms: {patient_data.get('symptom_description', 'None') if patient_data else 'None'}
- Patient: {patient_data.get('Name', 'Not provided') if patient_data else 'Not provided'} ({patient_data.get('Age', 'Unknown') if patient_data else 'Unknown'} years old)

🔒 **ALREADY AVAILABLE - DO NOT RE-ANALYZE:**
- Patient Information: {'Complete' if patient_data and patient_data.get('Name') and patient_data.get('Age') else 'Incomplete'}
- Specialties Analysis: {'Already Completed' if detected_specialties else 'Not Done Yet'}
- Symptom Analysis: {'Already Completed' if patient_data and patient_data.get('symptom_description') else 'Not Done Yet'}

🔍 TOOL EXECUTION RESULTS:
- Doctor Search: {'Performed' if doctor_search_result is not None else 'Not performed'}
- Doctors Found: {len(doctor_data) if doctor_data else 0}
- Symptom Analysis: {'Completed' if symptom_result else 'Not performed'}

📋 RESPONSE STRATEGY:
1. **If user asks for doctor search AND specialties are detected**: 
   - Acknowledge the detected specialties
   - Explain that you can search for doctors in those specialties
   - Ask for confirmation to proceed with search
   - **CRITICAL: DO NOT call analyze_symptoms again - specialties are already available**

2. **If user confirms they want to find doctors (says "yes", "okay", "please", etc.) AND specialties are detected**:
   - Acknowledge their confirmation
   - Use the already detected specialties
   - **CRITICAL: Call `search_doctors_dynamic` tool to find doctors**
   - DO NOT call `analyze_symptoms` again (specialties already detected)

3. **If user asks for doctor search BUT no specialties detected**:
   - Explain that symptoms need to be analyzed first
   - Ask them to describe their symptoms

4. **If user describes symptoms**:
   - Acknowledge their symptoms
   - Explain that you're analyzing for appropriate specialties
   - Provide the analysis results

4. **Always be conversational and helpful**:
   - Use the patient's name naturally
   - Reference detected specialties appropriately
   - Never mention tools, APIs, or system internals
   - Be encouraging and supportive

5. **RESPONSE FORMAT EXAMPLES**:
    ✅ **CORRECT**: "Nice to meet you, Hammad! I've noted your information. How can I help you today?"
    ❌ **WRONG**: "Nice to meet you, Hammad! I've noted your information. [Tool: store_patient_details...]"
    
    ✅ **CORRECT**: "I found 5 dentists in your area specializing in general dentistry."
    ❌ **WRONG**: "I will search for dentists. [Tool: search_doctors_dynamic...]"

🚫 **CRITICAL TOOL USAGE RULES:**
- **NEVER call `analyze_symptoms` if EXACT SAME symptoms already analyzed**
- **NEVER call `store_patient_details` if patient information is already complete**
- **Use existing detected specialties for doctor searches when appropriate**
- **Call `analyze_symptoms` for NEW or DIFFERENT symptoms (even if specialties exist)**

🚫 **CRITICAL RESPONSE RULES:**
- **NEVER mention tools, APIs, or system internals in your response**
- **NEVER show tool call details like `[Tool: store_patient_details...]`**
- **NEVER mention "I will search", "I am looking", or future tense actions**
- **ONLY provide natural, conversational responses**
- **Present information as already available and complete**
- **Act as if you already have all the information you need**

⚠️ **FINAL WARNING**: Your response will be shown directly to the user. Make sure it contains ONLY natural conversation and NO technical details, tool calls, or system information.

## 📋 **QUICK REFERENCE SUMMARY:**

**TOOL SELECTION RULES:**
1. **Personal Info** → `store_patient_details`
2. **Direct Doctor Search** → `search_doctors_dynamic`
3. **New Symptoms** → `analyze_symptoms`
4. **Confirm After Analysis** → `search_doctors_dynamic`
5. **New Health Issue** → `analyze_symptoms` (replaces previous)

**NEVER DO:**
- ❌ Call `analyze_symptoms` when EXACT SAME symptoms already analyzed
- ❌ Call `store_patient_details` when patient info complete
- ❌ Call `search_doctors_dynamic` without specialties (unless direct search)

**ALWAYS DO:**
- ✅ Use detected specialties for doctor search
- ✅ Handle flow switching gracefully
- ✅ Provide natural, helpful responses

Generate a natural, helpful response that follows this strategy and incorporates the context appropriately."""
                        
                    final_context_message = {
                        "role": "system",
                        "content": final_context_message_prompt
                    }

                    # AI will handle patient context dynamically
                    # Refresh patient data to ensure we have the latest information
                    
                    # logger.info(f"FINAL MESSAGE PROMPT: \n {final_context_message_prompt} \n")
                    patient_data = history.get_patient_data()
                    logger.info(f"🔄 Final patient_data before response: {patient_data}")
                    
                    # Ensure we have the most up-to-date patient data
                    if patient_data is None:
                        patient_data = {}
                    logger.info(f"🔄 Final patient_data after None check: {patient_data}")

                    # AI will handle context generation dynamically

                    # Make the final AI call to generate the response
                    final_response = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=messages + [final_context_message],
                        tools=[],  # No tools needed for final response
                        tool_choice="none",
                    )

                    final_response_content = final_response.choices[0].message.content

                    # CRITICAL: Add the final context message and AI response to conversation history
                    # This ensures the main agent maintains full context between messages
                    logger.info(f"🔄 Adding final context message and AI response to conversation history")
                    
                    # Add the final context message to history (for debugging and context tracking)
                    history.add_ai_message(f"[SYSTEM CONTEXT: {final_context_message_prompt[:200]}...]")
                    
                    # Add the final AI response to history
                    history.add_ai_message(final_response_content)
                    
                    # Also add to the OpenAI messages for next turn
                    messages.append({"role": "assistant", "content": final_response_content})

                    # Build the response object using the final AI response
                    logger.info(f"🔄 Building response object with patient_data: {patient_data}")
                    logger.info(f"🔄 Building response object with session_id: {session_id}")
                    
                    # Detect if this is a consent message asking for permission to search for doctors
                    # Use current message tools instead of entire history for more accurate detection
                    is_consent_message = (
                        "analyze_symptoms" in current_message_tools and
                        "search_doctors_dynamic" not in current_message_tools and
                        patient_data and patient_data.get("detected_specialties")
                    )
                    
                    # Log consent message detection for debugging
                    if is_consent_message:
                        logger.info(f"🔒 CONSENT MESSAGE DETECTED: Excluding old doctor data from response")
                    else:
                        logger.info(f"🔍 NOT CONSENT MESSAGE: Including doctor data as normal")
                    
                    response_object = {
                        "response": {
                            "message": final_response_content,
                            "patient": patient_data if patient_data else {"session_id": session_id},
                            "data": [] if is_consent_message else (doctor_data if has_doctor_results else []),
                        },
                        "display_results": not is_consent_message and has_doctor_results,
                    }
                    
                    # Debug logging for response object
                    logger.info(f"🔄 Final response_object.patient: {response_object['response']['patient']}")
                    logger.info(f"🔄 Final patient_data variable: {patient_data}")
                    logger.info(f"🔄 Final history.get_patient_data(): {history.get_patient_data()}")

                    # AI will handle offers data dynamically
                    if offers_data:
                        response_object["response"]["offers"] = offers_data
                    else:
                        response_object["response"]["offers"] = []

                    # Add doctor-specific fields based on results
                    if has_doctor_results:
                        response_object["response"]["is_doctor_search"] = True
                        response_object["doctor_count"] = len(doctor_data)

                        # Safely clear symptom_analysis from history after doctor search
                        try:
                            if hasattr(history, 'symptom_analysis'):
                                history.clear_symptom_analysis()
                            else:
                                logger.info("ℹ️ No symptom_analysis to clear - direct doctor search scenario")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not clear symptom analysis: {e}")
                            
                    elif doctor_search_result is not None:
                        # This handles both the zero results case and the
                        # case where we explicitly processed no doctors found above
                        response_object["response"]["is_doctor_search"] = True
                        response_object["doctor_count"] = 0

                    return response_object

                else:
                    # No tool calls were made - handle direct response
                    logger.info("ℹ️ No tool calls requested - processing direct response")
                    
                    # Add the assistant message to history
                    content = response_message.content
                    if content:
                        messages.append({"role": "assistant", "content": content})
                        history.add_ai_message(content)
                        
                        # Build response object for direct response
                        response_object = {
                            "response": {
                                "message": content,
                                "patient": patient_data or {"session_id": session_id},
                                "data": [],
                            },
                            "display_results": False,
                        }
                        
                        # If symptom analysis was performed, add to the response
                        symptom_result = history.get_symptom_analysis()
                        if symptom_result:
                            response_object["symptom_analysis"] = symptom_result
                        
                        return response_object
                    else:
                        logger.warning("⚠️ Response message content is null")
                        # AI will handle error responses dynamically
                        return {
                            "response": {
                                "message": "I apologize, but I couldn't generate a proper response. Please try rephrasing your question.",
                                "patient": patient_data or {"session_id": session_id},
                                "data": [],
                                "error": True,
                            },
                            "display_results": False,
                        }

            except Exception as e:
                logger.error(f"❌ Error processing message: {str(e)}", exc_info=True)

                # Main agent handles error responses - no additional AI calls
                # Return a structured error response that the main agent can handle
                return {
                    "response": {
                        "message": "I apologize, but I encountered an error while processing your request. Please try rephrasing your question or contact support if the issue persists.",
                        "patient": patient_data or {"session_id": session_id},
                        "data": [],
                        "error": True,
                        "error_details": str(e),
                    },
                    "display_results": False,
                }
        except Exception as e:
            logger.error(f"❌ Error in main processing: {str(e)}", exc_info=True)
            # Return error response if main processing fails
            return {
                "response": {
                    "message": "I apologize, but I encountered an error while processing your request. Please try again.",
                    "patient": patient_data or {"session_id": session_id},
                    "data": [],
                    "error": True,
                    "error_details": str(e),
                },
                "display_results": False,
            }