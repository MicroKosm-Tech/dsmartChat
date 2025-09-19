import os
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel
from sqlalchemy import text
from decimal import Decimal
from .specialty_matcher import SpecialtyDataCache
from .db import DB
import threading
import time
from openai import OpenAI
import uuid
import json
from jellyfish import metaphone


# Initialize thread_local storage
thread_local = threading.local()

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Initialize the database connection
db = DB()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))

class SearchCriteria(BaseModel):
    """Model representing search criteria extracted from user query"""
    speciality: Optional[str] = None
    subspeciality: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    min_rating: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_experience: Optional[float] = None
    gender: Optional[str] = None
    hospital_name: Optional[str] = None
    branch_name: Optional[str] = None
    doctor_name: Optional[str] = None
    original_message: Optional[str] = None
    
    def dict(self, exclude_none: bool = False, **kwargs):
        """Return dictionary representation with option to exclude None values"""
        result = super().dict(**kwargs)
        if exclude_none:
            return {k: v for k, v in result.items() if v is not None}
        return result

def clean_subspeciality(subspeciality: str) -> str:
    """
    Extract only the first subspeciality before any comma.
    This ensures we pass only one subspeciality to the stored procedure.
    
    Args:
        subspeciality: The subspeciality string that may contain commas
        
    Returns:
        Cleaned subspeciality string with only the first part before comma
    """
    if not subspeciality:
        return subspeciality
    
    # Validate input type
    if not isinstance(subspeciality, str):
        logger.error(f"Invalid subspeciality type: {type(subspeciality)}, expected str")
        return str(subspeciality) if subspeciality else ""
    
    # Split by comma and take only the first part
    first_part = subspeciality.split(',')[0]
    
    # Clean up whitespace
    cleaned = first_part.strip()
    
    # Validate cleaned result
    if not cleaned:
        logger.warning(f"Subspeciality became empty after cleaning: '{subspeciality}'")
        return ""
    
    # Log the cleaning process for debugging
    if ',' in subspeciality:
        logger.info(f"🧹 Cleaned subspeciality: '{subspeciality}' -> '{cleaned}' (removed comma-separated parts)")
    else:
        logger.info(f"🧹 Subspeciality already clean: '{subspeciality}'")
    
    return cleaned

def generate_sound_name(name: str) -> str:
    """Generate sound name using metaphone phonetic encoding."""
    if not name or not name.strip():
        return ""
    
    words = name.split()
    phonetic_codes = [metaphone(word) for word in words]
    return ' '.join(phonetic_codes)

def clean_and_generate_sound_name(user_input: str) -> str:
    """Clean user input and generate sound name for database matching."""
    if not user_input:
        return ""
    
    # Remove doctor prefixes
    prefixes_to_remove = ['dr.', 'dr', 'doctor', 'prof.', 'prof', 'professor']
    name_lower = user_input.lower().strip()
    
    for prefix in prefixes_to_remove:
        if name_lower.startswith(prefix + ' '):
            user_input = user_input[len(prefix):].strip()
            break
        elif name_lower.startswith(prefix):
            user_input = user_input[len(prefix):].strip()
            break
    
    cleaned_name = ' '.join(user_input.split())
    return generate_sound_name(cleaned_name)


def build_query_params(criteria: SearchCriteria) -> Dict[str, Any]:
    """
    Build parameters for Sp_IntelligentSearch stored procedure without executing it
    
    Args:
        criteria: SearchCriteria object containing search parameters
        
    Returns:
        Dictionary containing the parameters for the stored procedure
    """
    try:
        # Log the criteria we're using
        logger.info(f"Building query parameters with criteria: {criteria.dict(exclude_none=True)}")
        
        # Debug log for specialty/subspecialty tracking
        if criteria.speciality:
            logger.info(f"DEBUG: Using specialty '{criteria.speciality}' in WHERE clause")
            if criteria.subspeciality:
                cleaned_subspeciality = clean_subspeciality(criteria.subspeciality)
                logger.info(f"DEBUG: Using subspecialty '{criteria.subspeciality}' -> cleaned to '{cleaned_subspeciality}' in WHERE clause")
        else:
            logger.warning("DEBUG: No specialty found in search criteria!")
        
        # Start building dynamic WHERE clause
        where_conditions = []
        
        # Doctor name search (high priority) - using sound name for phonetic matching
        if criteria.doctor_name:
            where_conditions.append(build_doctor_sound_name_clause(criteria.doctor_name))

        # Hospital/branch name search
        if criteria.branch_name:
            where_conditions.append(build_multi_word_like_clause("bg.BranchName", criteria.branch_name))
        
        # Hospital name (if different from branch name)
        if criteria.hospital_name and criteria.hospital_name != criteria.branch_name:
            hospital_name = criteria.hospital_name.replace("'", "''")
            # where_conditions.append(f"AND (bg.BranchName_en LIKE N'%{hospital_name}%' OR bg.BranchName_ar LIKE N'%{hospital_name}%')")
            where_conditions.append(f"""AND (SELECT STRING_AGG(N'(bg.BranchName_en LIKE N''%'+ value + N'%'' OR bg.BranchName_ar LIKE N''%'+ value + N'%'' )',N' AND ') FROM STRING_SPLIT(N'{hospital_name}', N' ')) = (N'(bg.BranchName_en LIKE N''%{hospital_name}%'' OR bg.BranchName_ar LIKE N''%{hospital_name}%'')')""")
        
        # Specialty and subspecialty search
        if criteria.speciality:
            specialty_value = criteria.speciality.replace("'", "''")
            
            if criteria.subspeciality:
                # Clean subspeciality to get only first part before comma
                cleaned_subspeciality = clean_subspeciality(criteria.subspeciality)
                subspecialty_value = cleaned_subspeciality.replace("'", "''")
                
                # Use a combined search that looks for the specialty in the Specialty field
                # and the cleaned subspecialty in the Subspecialities field
                where_conditions.append(f"AND (le.Specialty LIKE N'%{specialty_value}%' AND s.SubSpeciality LIKE N'%{subspecialty_value}%')")
                logger.info(f"Added combined specialty+subspecialty filter: (le.Specialty LIKE N'%{specialty_value}%' AND s.SubSpeciality LIKE N'%{subspecialty_value}%')")
            else:
                # If no subspecialty, search for the specialty in both fields
                where_conditions.append(f"AND (le.Specialty LIKE N'%{specialty_value}%' OR s.SubSpeciality LIKE N'%{specialty_value}%')")
                logger.info(f"Added specialty-only filter: (le.Specialty LIKE N'%{specialty_value}%' OR s.SubSpeciality LIKE N'%{specialty_value}%')")
        
        # Handle subspecialty without specialty (rare case)
        elif criteria.subspeciality:
            # Clean subspeciality to get only first part before comma
            cleaned_subspeciality = clean_subspeciality(criteria.subspeciality)
            subspecialty_value = cleaned_subspeciality.replace("'", "''")
            
            # If it's likely a dental subspecialty, assume dentist as the specialty
            dental_subspecialties = ["Orthodontics", "Endodontics", "Periodontics", "Dental Implants", 
                                     "Prosthodontics", "Oral Surgery", "Oral and Maxillofacial Surgery",
                                     "Pediatric Dentistry", "GP", "Dental Hygienist"]
            
            if cleaned_subspeciality in dental_subspecialties:
                # For dental subspecialties, search for dentistry in Specialty and the cleaned subspecialty in Subspecialities
                where_conditions.append(f"AND (le.Specialty LIKE N'%dentist%' AND s.SubSpeciality LIKE N'%{subspecialty_value}%')")
                logger.info(f"Added dental subspecialty filter: (le.Specialty LIKE N'%dentist%' AND s.SubSpeciality LIKE N'%{subspecialty_value}%')")
            else:
                # For non-dental subspecialties, just search for the cleaned subspecialty in the Subspecialities field
                where_conditions.append(f"AND (s.SubSpeciality LIKE N'%{subspecialty_value}%')")
                logger.info(f"Added subspecialty-only filter: (s.SubSpeciality LIKE N'%{subspecialty_value}%')")
        
        # Rating filter
        if criteria.min_rating is not None:
            where_conditions.append(f"AND le.Rating >= {float(criteria.min_rating)}")
        
        # Price/fee filters
        if criteria.min_price is not None and criteria.max_price is not None:
            where_conditions.append(f"AND le.Fee BETWEEN {float(criteria.min_price)} AND {float(criteria.max_price)}")
        elif criteria.min_price is not None:
            where_conditions.append(f"AND le.Fee >= {float(criteria.min_price)}")
        elif criteria.max_price is not None:
            where_conditions.append(f"AND le.Fee <= {float(criteria.max_price)}")
        
        # Experience filter
        if criteria.min_experience is not None:
            where_conditions.append(f"AND le.Experience >= {float(criteria.min_experience)}")
        
        # Gender filter
        if criteria.gender:
            gender_value = criteria.gender.replace("'", "''")
            where_conditions.append(f"AND le.Gender = N'{gender_value}'")
        
        # Combine conditions into a single where clause
        # The stored procedure expects the WHERE clause with AND at the beginning of each condition
        where_clause = " ".join(where_conditions)
            
        logger.info(f"Built WHERE clause: {where_clause}")
        
        # Generate sound name for doctor search if doctor name is provided
        name_sound = ""
        if criteria.doctor_name:
            name_sound = clean_and_generate_sound_name(criteria.doctor_name)
            logger.info(f"🔊 Generated sound name: '{criteria.doctor_name}' -> '{name_sound}'")
        
        # Create parameters dictionary with latitude, longitude, name sound, and WHERE clause
        params = {
            "@Latitude": criteria.latitude if criteria.latitude is not None else 0.0,
            "@Longitude": criteria.longitude if criteria.longitude is not None else 0.0,
            "@NameSound": name_sound,
            "@DynamicWhereClause": where_clause,
            "@BoostedOnly": 0
        }
        
        logger.info(f"Using coordinates: Lat={params['@Latitude']}, Long={params['@Longitude']}")
        logger.info(f"Generated parameters: {params}")
        
        return params
        
    except Exception as e:
        logger.error(f"Error building query parameters: {str(e)}")
        raise

def build_query(criteria: SearchCriteria) -> Dict[str, Any]:
    """
    Build parameters for Sp_IntelligentSearch stored procedure and execute it
    
    Args:
        criteria: SearchCriteria object containing search parameters
        
    Returns:
        Dictionary containing the search results from the stored procedure
    """
    try:
        # Build parameters using the helper function
        params = build_query_params(criteria)
        
        # Execute the stored procedure and log the results
        sp_name = "[dbo].[Sp_IntelligentSearch]"
        logger.info(f"🔍 DOCTOR SEARCH: Executing stored procedure: {sp_name}")
        logger.info(f"🔍 DOCTOR SEARCH: With parameters: {params}")
        
        result = db.execute_stored_procedure(sp_name, params)
        logger.info(f"🔍 DOCTOR SEARCH: Database returned result: {json.dumps(result, indent=2)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error building query: {str(e)}")
        raise
def build_multi_word_like_clause(field_prefix: str, phrase: str) -> str:
    """
    Build a dynamic WHERE clause part for multi-word AND search on English and Arabic fields.

    :param field_prefix: The table alias and field prefix, e.g. 'bg.BranchName'
    :param phrase: The user input phrase
    :return: A string like:
        AND ((bg.BranchName_en LIKE N'%word1%' OR bg.BranchName_ar LIKE N'%word1%') AND (bg.BranchName_en LIKE N'%word2%' OR bg.BranchName_ar LIKE N'%word2%'))
    """
    # Escape single quotes in phrase for T-SQL
    safe_phrase = phrase.replace("'", "''").strip()
    if not safe_phrase:
        return ""

    words = safe_phrase.split()
    conditions = [
        f"({field_prefix}_en LIKE N'%{word}%' OR {field_prefix}_ar LIKE N'%{word}%')" for word in words
    ]
    return "AND (" + " AND ".join(conditions) + ")"

def build_doctor_sound_name_clause(doctor_name: str) -> str:
    """
    Build a WHERE clause for doctor name search.
    Since we're now using @NameSound parameter in the stored procedure,
    this function returns an empty string as the sound matching is handled
    by the stored procedure itself.

    :param doctor_name: The doctor name to search for
    :return: Empty string since sound matching is handled by @NameSound parameter
    """
    if not doctor_name or not doctor_name.strip():
        return ""
    
    # Generate sound name for logging purposes
    sound_name = clean_and_generate_sound_name(doctor_name)
    
    logger.info(f"🔊 DOCTOR SOUND SEARCH: Original='{doctor_name}' -> Sound='{sound_name}' (handled by @NameSound parameter)")
    
    # Return empty string since sound matching is now handled by the stored procedure
    # via the @NameSound parameter
    return ""

def normalize_specialty(specialty_name: str) -> dict:
    """
    Normalize specialty names to match the database terminology.
    This is needed because user queries and LLM outputs may use different terms
    than what's used in the database.
    
    Args:
        specialty_name: The original specialty name
        
    Returns:
        Dictionary with normalized specialty and subspecialty if applicable
    """
    if not specialty_name:
        return {"specialty": specialty_name}
        
    # Convert to lowercase for comparison
    specialty_lower = specialty_name.lower()
    
    # Special case: GP is a subspecialty of DENTISTRY
    gp_terms = ["gp", "general practice", "general practitioner", "family medicine", 
                "general physician", "primary care"]
    
    for term in gp_terms:
        if term in specialty_lower:
            logger.info(f"SPECIALTY: Detected GP term in '{specialty_name}', setting as subspecialty of Dentistry")
            return {
                "specialty": "Dentistry",
                "subspecialty": "GP"
            }
    
    # Common specialty name mappings - use proper case as expected by the stored procedure
    mappings = {
        "dentist": "Dentistry",
        "dental": "Dentistry",
        "dentistry": "Dentistry",
        "cardiology": "Cardiology",
        "cardiologist": "Cardiology",
        "heart": "Cardiology",
        "dermatology": "Dermatology",
        "dermatologist": "Dermatology",
        "skin": "Dermatology",
        "pediatric": "Pediatrics",
        "pediatrics": "Pediatrics",
        "pediatrician": "Pediatrics",
        "children": "Pediatrics",
        "orthopedic": "Orthopedics",
        "orthopedics": "Orthopedics",
        "bones": "Orthopedics",
        "gynecology": "Obstetrics & Gynecology",
        "obgyn": "Obstetrics & Gynecology",
        "obstetrics": "Obstetrics & Gynecology"
    }
    
    # Check for exact matches first
    if specialty_lower in mappings:
        normalized = mappings[specialty_lower]
        logger.info(f"SPECIALTY: Normalized '{specialty_name}' to '{normalized}'")
        return {"specialty": normalized}
    
    # Check for partial matches
    for key, value in mappings.items():
        if key in specialty_lower:
            logger.info(f"SPECIALTY: Partial match normalized '{specialty_name}' to '{value}'")
            return {"specialty": value}
    
    # If no mapping found, return the original with first letter of each word capitalized
    logger.info(f"SPECIALTY: No mapping found for '{specialty_name}', keeping as is")
    return {"specialty": specialty_name}

def extract_processed_criteria(search_criteria: Union[dict, str]) -> Dict[str, Any]:
    """
    Extract processed search criteria from the unified search function.
    This function processes the search criteria the same way as unified_doctor_search
    but returns the processed criteria instead of executing the search.
    
    Args:
        search_criteria: The search criteria to process
        
    Returns:
        Dictionary containing the processed search criteria
    """
    logger.info(f"EXTRACTING CRITERIA: Processing search criteria: {search_criteria}")
    
    # Initialize search parameters
    search_params = {}
    
    # Handle different input types
    if isinstance(search_criteria, str):
        try:
            # Try to parse as JSON first
            json_data = json.loads(search_criteria)
            if isinstance(json_data, dict):
                search_params = json_data
                logger.info(f"EXTRACTING CRITERIA: Parsed JSON search criteria: {search_params}")
        except json.JSONDecodeError:
            # If not JSON, treat as natural language query
            logger.info(f"EXTRACTING CRITERIA: Processing natural language query: '{search_criteria}'")
            search_params = extract_search_criteria_from_message(search_criteria)
            logger.info(f"EXTRACTING CRITERIA: Extracted search criteria: {search_params}")
            
    elif isinstance(search_criteria, dict):
        # If it's already a dict, check if it contains a user_message
        if "user_message" in search_criteria:
            user_message = search_criteria["user_message"]
            logger.info(f"EXTRACTING CRITERIA: Found user_message in criteria: '{user_message}'")
            
            # ALWAYS extract from user message first to get ALL filters (rating, price, experience, doctor name, etc.)
            logger.info(f"EXTRACTING CRITERIA: Always extracting from user message to get all filters")
            extracted_criteria = extract_search_criteria_from_message(user_message)
            logger.info(f"EXTRACTING CRITERIA: Extracted search criteria: {extracted_criteria}")
            
            # Apply specialty priority logic AFTER extraction
            if "speciality" in search_criteria and search_criteria["speciality"] and "subspeciality" in search_criteria and search_criteria["subspeciality"]:
                # Priority 1: Use provided specialties from tool parameters (override extracted ones)
                search_params["speciality"] = search_criteria["speciality"]
                search_params["subspeciality"] = search_criteria["subspeciality"]
                logger.info(f"EXTRACTING CRITERIA: Using provided specialties - speciality: '{search_criteria['speciality']}', subspeciality: '{search_criteria['subspeciality']}' (overriding extracted)")
                
            elif "speciality" in search_criteria and search_criteria["speciality"]:
                # Priority 2: Use provided specialty, use extracted subspecialty if available
                search_params["speciality"] = search_criteria["speciality"]
                logger.info(f"EXTRACTING CRITERIA: Using provided specialty '{search_criteria['speciality']}'")
                
                if "subspeciality" in extracted_criteria and extracted_criteria["subspeciality"]:
                    search_params["subspeciality"] = extracted_criteria["subspeciality"]
                    logger.info(f"EXTRACTING CRITERIA: Using extracted subspecialty '{extracted_criteria['subspeciality']}' from message")
                
            elif "subspeciality" in search_criteria and search_criteria["subspeciality"]:
                # Priority 3: Use provided subspecialty, use extracted specialty if available
                search_params["subspeciality"] = search_criteria["subspeciality"]
                logger.info(f"EXTRACTING CRITERIA: Using provided subspecialty '{search_criteria['subspeciality']}'")
                
                if "speciality" in extracted_criteria and extracted_criteria["speciality"]:
                    search_params["speciality"] = extracted_criteria["speciality"]
                    logger.info(f"EXTRACTING CRITERIA: Using extracted specialty '{extracted_criteria['speciality']}' from message")
                
            else:
                # Priority 4: No specialties provided in params, use extracted ones
                if "speciality" in extracted_criteria and extracted_criteria["speciality"]:
                    search_params["speciality"] = extracted_criteria["speciality"]
                    logger.info(f"EXTRACTING CRITERIA: Using extracted specialty '{extracted_criteria['speciality']}' from message")
                
                if "subspeciality" in extracted_criteria and extracted_criteria["subspeciality"]:
                    search_params["subspeciality"] = extracted_criteria["subspeciality"]
                    logger.info(f"EXTRACTING CRITERIA: Using extracted subspecialty '{extracted_criteria['subspeciality']}' from message")
                
                # Handle dental subspecialty without specialty case
                elif "subspeciality" in extracted_criteria and "speciality" not in extracted_criteria:
                    dental_subspecialties = ["Orthodontics", "Endodontics", "Periodontics", "Dental Implants", 
                                          "Prosthodontics", "Oral Surgery", "Oral and Maxillofacial Surgery",
                                          "Pediatric Dentistry", "GP", "Dental Hygienist"]
                    if extracted_criteria["subspeciality"] in dental_subspecialties:
                        search_params["speciality"] = "dentist"
                        logger.info(f"EXTRACTING CRITERIA: Added default specialty 'dentist' for dental subspecialty '{extracted_criteria['subspeciality']}'")
            
            # ALWAYS copy over ALL other extracted criteria (rating, price, experience, doctor name, clinic name, etc.)
            for key, value in extracted_criteria.items():
                if key not in ["speciality", "subspeciality"]:  # We already handled specialties above
                    search_params[key] = value
                    logger.info(f"EXTRACTING CRITERIA: Added extracted criteria '{key}': '{value}'")
            
            # Copy over any additional parameters from original criteria (coordinates, etc.)
            for key, value in search_criteria.items():
                if key != "user_message" and key not in search_params:
                    search_params[key] = value
                    logger.info(f"EXTRACTING CRITERIA: Added original criteria '{key}': '{value}'")
                    
            # Check if we have symptom analysis data that should supplement (NOT override) search criteria
            # Only use symptom analysis if we don't already have specialties from tool parameters
            if hasattr(thread_local, 'symptom_analysis') and ("speciality" not in search_params or "subspecialty" not in search_params):
                symptom_data = thread_local.symptom_analysis
                logger.info(f"EXTRACTING CRITERIA: Found symptom analysis data to supplement search criteria")
                
                # Extract specialties from symptom analysis
                specialties = []
                
                # First check the symptom_analysis structure
                if 'symptom_analysis' in symptom_data:
                    sa = symptom_data.get('symptom_analysis', {})
                    
                    # Try all possible locations for specialty data
                    if 'specialties' in sa and sa['specialties']:
                        specialties = sa['specialties']
                        logger.info(f"EXTRACTING CRITERIA: Found {len(specialties)} specialties in symptom_analysis.specialties")
                    elif 'recommended_specialties' in sa and sa['recommended_specialties']:
                        specialties = sa['recommended_specialties']
                        logger.info(f"EXTRACTING CRITERIA: Found {len(specialties)} specialties in symptom_analysis.recommended_specialties")
                
                # Then check the top-level specialties
                elif 'specialties' in symptom_data and symptom_data['specialties']:
                    specialties = symptom_data['specialties']
                    logger.info(f"EXTRACTING CRITERIA: Found {len(specialties)} specialties at top level of symptom data")
                
                # Use the symptom analysis specialty/subspecialty if we found any
                if specialties and len(specialties) > 0:
                    top_specialty = specialties[0]
                    logger.info(f"EXTRACTING CRITERIA: Using top specialty from symptom analysis: {top_specialty}")
                    
                    # Add specialty ONLY if we don't already have one (no overriding)
                    if "speciality" not in search_params and ("specialty" in top_specialty or "name" in top_specialty):
                        specialty_name = top_specialty.get("specialty") or top_specialty.get("name")
                        if specialty_name:
                            search_params["speciality"] = specialty_name
                            logger.info(f"EXTRACTING CRITERIA: Added missing specialty '{specialty_name}' from symptom analysis")
                    
                    # Add subspecialty ONLY if we don't already have one (no overriding)
                    if "subspecialty" not in search_params and "subspecialty" in top_specialty:
                        subspecialty_name = top_specialty.get("subspecialty")
                        if subspecialty_name:
                            search_params["subspecialty"] = subspecialty_name
                            logger.info(f"EXTRACTING CRITERIA: Added missing subspecialty '{subspecialty_name}' from symptom analysis")
        else:
            # Use the dictionary as-is
            search_params = search_criteria
            logger.info(f"EXTRACTING CRITERIA: Using provided dictionary criteria: {search_params}")
    
    # Add coordinates from search params if available
    lat = search_params.get("latitude")
    long = search_params.get("longitude")
    
    if lat is not None and long is not None:
        logger.info(f"EXTRACTING CRITERIA: Coordinates - lat: {lat}, long: {long}")
    else:
        logger.warning("EXTRACTING CRITERIA: No coordinates provided in search criteria")
    
    logger.info(f"EXTRACTING CRITERIA: Final processed criteria: {search_params}")
    return search_params

def unified_doctor_search(search_criteria: Union[dict, str]) -> dict:
    """Unified doctor search function that handles both structured and natural language queries"""
    logger.info(f"TOOL CALL: unified_doctor_search with input: {search_criteria}")
    
    # Initialize search parameters
    search_params = {}
    
    # Check if this is purely an information request (without doctor search needed)
    is_info_request = False
    if isinstance(search_criteria, dict) and "user_message" in search_criteria:
        # Try to detect if this is just asking for information about a procedure
        msg = search_criteria["user_message"].lower()
        info_patterns = ["what is", "tell me about", "explain", "information about", "how does", "what are"]
        
        # ENHANCED: Check if this is asking about a specific doctor (should trigger doctor search)
        doctor_patterns = ["dr ", "doctor ", "دكتور ", "specialty of", "speciality of"]
        is_doctor_query = any(pattern in msg for pattern in doctor_patterns)
        
        if any(pattern in msg for pattern in info_patterns):
            action_words = ["find", "book", "need", "looking for", "search for", "want", "recommend", "suggest"]
            # If it contains info words but no action words, it's likely just an info request
            # BUT if it's asking about a specific doctor, we should still search
            if not any(action in msg for action in action_words) and not is_doctor_query:
                logger.info(f"Detected information request without doctor search: '{msg}'")
                is_info_request = True
                
                return {
                    "response": {
                        "message": "This appears to be an information request. I can help with that.",
                        "raw_message": "",
                        "patient": {"session_id": getattr(thread_local, 'session_id', '')},
                        "data": [],
                        "is_doctor_search": False,
                        "is_info_request": True,
                        "needs_llm_processing": True
                    },
                    "display_results": False,
                    "doctor_count": 0
                }
            elif is_doctor_query:
                logger.info(f"Detected doctor-specific query: '{msg}' - Will proceed with doctor search")
    
    # Handle different input types
    if isinstance(search_criteria, str):
        try:
            # Try to parse as JSON first
            json_data = json.loads(search_criteria)
            if isinstance(json_data, dict):
                search_params = json_data
                logger.info(f"Parsed JSON search criteria: {search_params}")
        except json.JSONDecodeError:
            # If not JSON, treat as natural language query
            logger.info(f"Processing natural language query: '{search_criteria}'")
            # Extract search criteria from natural language
            print("\n" + "="*50)
            print("DEBUG: About to call extract_search_criteria_from_message")
            print("DEBUG: Input message:", search_criteria)
            print("="*50 + "\n")
            
            search_params = extract_search_criteria_from_message(search_criteria)
            logger.info(f"Extracted search criteria: {search_params}")
            
    elif isinstance(search_criteria, dict):
        # If it's already a dict, check if it contains a user_message
        if "user_message" in search_criteria:
            user_message = search_criteria["user_message"]
            logger.info(f"Found user_message in criteria: '{user_message}'")
            
            # ALWAYS extract from user message first to get ALL filters (rating, price, experience, doctor name, etc.)
            logger.info(f"Always extracting from user message to get all filters")
            extracted_criteria = extract_search_criteria_from_message(user_message)
            logger.info(f"Extracted search criteria: {extracted_criteria}")
            
            # Apply specialty priority logic AFTER extraction
            if "speciality" in search_criteria and search_criteria["speciality"] and "subspeciality" in search_criteria and search_criteria["subspeciality"]:
                # Priority 1: Use provided specialties from tool parameters (override extracted ones)
                search_params["speciality"] = search_criteria["speciality"]
                search_params["subspeciality"] = search_criteria["subspeciality"]
                logger.info(f"Using provided specialties - speciality: '{search_criteria['speciality']}', subspeciality: '{search_criteria['subspeciality']}' (overriding extracted)")
                        
            elif "speciality" in search_criteria and search_criteria["speciality"]:
                # Priority 2: Use provided specialty, use extracted subspecialty if available
                search_params["speciality"] = search_criteria["speciality"]
                logger.info(f"Using provided specialty '{search_criteria['speciality']}'")
                
                if "subspeciality" in extracted_criteria and extracted_criteria["subspeciality"]:
                    search_params["subspeciality"] = extracted_criteria["subspeciality"]
                    logger.info(f"Using extracted subspecialty '{extracted_criteria['subspeciality']}' from message")
                        
            elif "subspeciality" in search_criteria and search_criteria["subspeciality"]:
                # Priority 3: Use provided subspecialty, use extracted specialty if available
                search_params["subspeciality"] = search_criteria["subspeciality"]
                logger.info(f"Using provided subspecialty '{search_criteria['subspeciality']}'")
                
                if "speciality" in extracted_criteria and extracted_criteria["speciality"]:
                    search_params["speciality"] = extracted_criteria["speciality"]
                    logger.info(f"Using extracted specialty '{extracted_criteria['speciality']}' from message")
                        
            else:
                # Priority 4: No specialties provided in params, use extracted ones
                
                if "speciality" in extracted_criteria and extracted_criteria["speciality"]:
                    search_params["speciality"] = extracted_criteria["speciality"]
                    logger.info(f"Using extracted specialty '{extracted_criteria['speciality']}' from message")
                
                if "subspeciality" in extracted_criteria and extracted_criteria["subspeciality"]:
                    search_params["subspeciality"] = extracted_criteria["subspeciality"]
                    logger.info(f"Using extracted subspecialty '{extracted_criteria['subspeciality']}' from message")
                
                # Handle dental subspecialty without specialty case
                elif "subspeciality" in extracted_criteria and "speciality" not in extracted_criteria:
                    dental_subspecialties = ["Orthodontics", "Endodontics", "Periodontics", "Dental Implants", 
                                          "Prosthodontics", "Oral Surgery", "Oral and Maxillofacial Surgery",
                                          "Pediatric Dentistry", "GP", "Dental Hygienist"]
                    if extracted_criteria["subspeciality"] in dental_subspecialties:
                        search_params["speciality"] = "dentist"
                        logger.info(f"Added default specialty 'dentist' for dental subspecialty '{extracted_criteria['subspeciality']}'")
            
            # ALWAYS copy over ALL other extracted criteria (rating, price, experience, doctor name, clinic name, etc.)
            for key, value in extracted_criteria.items():
                if key not in ["speciality", "subspeciality"]:  # We already handled specialties above
                    search_params[key] = value
                    logger.info(f"Added extracted criteria '{key}': '{value}'")
            
            # Copy over any additional parameters from original criteria (coordinates, etc.)
            for key, value in search_criteria.items():
                if key != "user_message" and key not in search_params:
                    search_params[key] = value
                    logger.info(f"Added original criteria '{key}': '{value}'")
                
                # Copy over all other extracted criteria
                for key, value in extracted_criteria.items():
                    if key not in ["speciality", "subspecialty"]:  # We already handled these
                        search_params[key] = value
                
                # Preserve any additional parameters from original criteria
                for key, value in search_criteria.items():
                    if key != "user_message" and key not in search_params:
                        search_params[key] = value
                    
            # Check if we have symptom analysis data that should supplement (NOT override) search criteria
            # Only use symptom analysis if we don't already have specialties from tool parameters
            if hasattr(thread_local, 'symptom_analysis') and ("speciality" not in search_params or "subspecialty" not in search_params):
                symptom_data = thread_local.symptom_analysis
                logger.info(f"Found symptom analysis data to supplement missing specialties")
                
                # Extract specialties from symptom analysis
                specialties = []
                
                # First check the symptom_analysis structure
                if 'symptom_analysis' in symptom_data:
                    sa = symptom_data.get('symptom_analysis', {})
                    
                    # Try all possible locations for specialty data
                    if 'specialties' in sa and sa['specialties']:
                        specialties = sa['specialties']
                        logger.info(f"Found {len(specialties)} specialties in symptom_analysis.specialties")
                    elif 'recommended_specialties' in sa and sa['recommended_specialties']:
                        specialties = sa['recommended_specialties']
                        logger.info(f"Found {len(specialties)} specialties in symptom_analysis.recommended_specialties")
                
                # Then check the top-level specialties
                elif 'specialties' in symptom_data and symptom_data['specialties']:
                    specialties = symptom_data['specialties']
                    logger.info(f"Found {len(specialties)} specialties at top level of symptom data")
                
                # Use the symptom analysis specialty/subspecialty if we found any
                if specialties and len(specialties) > 0:
                    top_specialty = specialties[0]
                    logger.info(f"Using top specialty from symptom analysis: {top_specialty}")
                    
                    # Add specialty ONLY if we don't already have one (no overriding)
                    if "speciality" not in search_params and ("specialty" in top_specialty or "name" in top_specialty):
                        specialty_name = top_specialty.get("specialty") or top_specialty.get("name")
                        if specialty_name:
                            search_params["speciality"] = specialty_name
                            logger.info(f"Added missing specialty '{specialty_name}' from symptom analysis")
                    
                    # Add subspecialty ONLY if we don't already have one (no overriding)
                    if "subspecialty" not in search_params and "subspecialty" in top_specialty:
                        subspecialty_name = top_specialty.get("subspecialty")
                        if subspecialty_name:
                            search_params["subspecialty"] = subspecialty_name
                            logger.info(f"Added missing subspecialty '{subspecialty_name}' from symptom analysis")
        else:
            # Use the dictionary as-is
            search_params = search_criteria
            logger.info(f"Using provided dictionary criteria: {search_params}")
    
    # Add coordinates from search params if available
    lat = search_params.get("latitude")
    long = search_params.get("longitude")
    
    if lat is not None and long is not None:
        logger.info(f"UNIFIED_SEARCH: Coordinates - lat: {lat}, long: {long}")
    else:
        logger.warning("UNIFIED_SEARCH: No coordinates provided in search criteria")
    
    # Convert to SearchCriteria object
    logger.info(f"UNIFIED_SEARCH: Final search criteria: {search_params}")
    try:
        criteria = SearchCriteria(**search_params)
        logger.info(f"UNIFIED_SEARCH: Converted to SearchCriteria: {criteria.dict()}")
    except Exception as e:
        logger.error(f"UNIFIED_SEARCH: Error converting to SearchCriteria: {str(e)}")
        # If conversion fails, create empty criteria
        criteria = SearchCriteria()
        logger.warning("UNIFIED_SEARCH: Using empty SearchCriteria as fallback")
    
    # Build and execute query
    logger.info(f"🔍 DOCTOR SEARCH: Building query with criteria: {criteria.dict()}")
    try:
        logger.info(f"🔍 DOCTOR SEARCH: About to call build_query function")
        result = build_query(criteria)
        logger.info(f"🔍 DOCTOR SEARCH: build_query completed successfully")
        
        # Process results
        if result and "data" in result and "doctors" in result["data"]:
            doctors = result["data"]["doctors"]
            logger.info(f"🔍 DOCTOR SEARCH: Found {len(doctors)} doctors")
            logger.info(f"🔍 DOCTOR SEARCH: Returning result with {len(doctors)} doctors")
        else:
            logger.warning("🔍 DOCTOR SEARCH: No doctors found in result")
            result = {"data": {"doctors": []}}
        
        return {
            "response": {
                "message": f"I found {len(result['data']['doctors'])} doctors matching your criteria.",
                "raw_message": "",
                "patient": {"session_id": getattr(thread_local, 'session_id', '')},
                "data": result["data"]["doctors"],
                "is_doctor_search": True,
                "needs_llm_processing": True
            },
            "display_results": len(result["data"]["doctors"]) > 0,
            "doctor_count": len(result["data"]["doctors"])
        }
    except Exception as e:
        logger.error(f"UNIFIED_SEARCH: Error building or executing query: {str(e)}")
        return {
            "response": {
                "message": "I couldn't find any doctors matching your criteria.",
                "raw_message": "",
                "patient": {"session_id": getattr(thread_local, 'session_id', '')},
                "data": [],
                "is_doctor_search": True,
                "needs_llm_processing": True
            },
            "display_results": False,
            "doctor_count": 0
        }

def extract_search_criteria_from_message(message: str) -> Dict[str, Any]:
    """
    Uses GPT to extract search criteria from a message.
    This ensures consistent understanding between symptom analysis and search criteria.
    
    Args:
        message: User's message to extract criteria from
        
    Returns:
        Dictionary containing extracted search criteria
    """
    # Add very obvious debug statement
    print("\n" + "!"*50)
    print("DEBUG: EXTRACTION FUNCTION CALLED!")
    print("DEBUG: Input message:", message)
    print("!"*50 + "\n")
    
    try:
        # Log to both root logger and module logger for visibility
        root_logger.info("\n" + "="*50)
        root_logger.info("🔍 Starting search criteria extraction for message: '%s'", message)
        root_logger.info("="*50)
        
        # Prepare prompt for criteria extraction
        system_prompt = """Extract search criteria from the user's message into a structured format. Focus on:
        - Specialty/type of doctor (e.g., dentistry, cardiology, pediatrics) - ALWAYS in English
        - Price range (min and max in SAR) in western numbers
        - Rating requirements (minimum rating out of 5) in western numbers
        - Experience requirements (minimum years) in western numbers
        - Doctor name if mentioned (with title Dr/Doctor removed) - ALWAYS convert to English for phonetic matching
        - Clinic/branch name if mentioned - KEEP in original language (Arabic/English)
        - Gender preference ('male' or 'female' doctor) - ALWAYS in English
        
        IMPORTANT RULES:
        1. Doctor names and branch names:
           - Doctor names: ALWAYS convert to English for phonetic matching (even if mentioned in Arabic)
           - Branch names: Keep in original language (Arabic or English)
           - Remove titles like "Dr.", "Doctor", "الدكتور", "دكتور", "د.", "دكتر", "دكتوره", "دكتورة"
           - For Arabic names, pay special attention to these patterns:
             * "الدكتور [name]" -> extract "[name]"
             * "دكتور [name]" -> extract "[name]"
             * "ابحث عن الدكتور [name]" -> extract "[name]"
             * "اريد الدكتور [name]" -> extract "[name]"
             * "عايز الدكتور [name]" -> extract "[name]"
             * "عند الدكتور [name]" -> extract "[name]"
             * "مع الدكتور [name]" -> extract "[name]"
             * "عند دكتور [name]" -> extract "[name]"
             * "مع دكتور [name]" -> extract "[name]"
             * "عايز دكتور [name]" -> extract "[name]"
             * "ابحث عن دكتور [name]" -> extract "[name]"
             * "اريد دكتور [name]" -> extract "[name]"
             * "ابحث عن [name]" -> extract "[name]" (if context suggests it's a doctor)
             * "اريد [name]" -> extract "[name]" (if context suggests it's a doctor)
             * "عايز [name]" -> extract "[name]" (if context suggests it's a doctor)
             * "عند [name]" -> extract "[name]" (if context suggests it's a doctor)
             * "مع [name]" -> extract "[name]" (if context suggests it's a doctor)
           - Examples for doctor names (convert Arabic to English):
             * "Dr. Smith" -> "Smith"
             * "الدكتور أحمد" -> "Ahmed" (convert Arabic to English)
             * "دكتور محمد" -> "Mohammed" (convert Arabic to English)
             * "ابحث عن الدكتور الغريب" -> "Alghareeb" (convert Arabic to English)
             * "اريد الدكتور يوسف" -> "Youssef" (convert Arabic to English)
             * "عايز الدكتور علي" -> "Ali" (convert Arabic to English)
             * "عند الدكتور خالد" -> "Khalid" (convert Arabic to English)
             * "مع الدكتور سعيد" -> "Saeed" (convert Arabic to English)
             * "ابحث عن الغريب" -> "Alghareeb" (convert Arabic to English)
             * "اريد يوسف" -> "Youssef" (convert Arabic to English)
             * "عايز علي" -> "Ali" (convert Arabic to English)
             * "عند خالد" -> "Khalid" (convert Arabic to English)
             * "مع سعيد" -> "Saeed" (convert Arabic to English)
           - Examples for branch names (keep original language):
             * "مستشفى الملك فهد" -> "مستشفى الملك فهد"
             * "King Fahd Hospital" -> "King Fahd Hospital"
        
        2. Branch/Clinic name extraction for offers:
           - Pay special attention to offers-related queries
           - Look for patterns like "offers from [clinic]", "offers at [clinic]", "offers in [clinic]"
           - Extract the clinic/branch name even if no specialty is mentioned
           - Examples:
             * "find me offers from glam clinic" -> branch_name: "glam clinic"
             * "show offers at King Fahd Hospital" -> branch_name: "King Fahd Hospital"
             * "offers from dental clinic" -> branch_name: "dental clinic"
             * "عروض من عيادة الأسنان" -> branch_name: "عيادة الأسنان"
             * "عروض في مستشفى الملك فهد" -> branch_name: "مستشفى الملك فهد"
             * "عروض من عيادة التجميل" -> branch_name: "عيادة التجميل"
             * "عروض خاصة من عيادة الأسنان" -> branch_name: "عيادة الأسنان"
             * "خصومات في عيادة التجميل" -> branch_name: "عيادة التجميل"
        
        3. All other fields must be in English:
           - Specialty: "طبيب أسنان" -> "dentistry"
           - Gender: "طبيبة" -> "female", "طبيب" -> "male"
           - Numbers: Use western numerals (1, 2, 3) not Arabic numerals (١، ٢، ٣)
        
        4. For gender, look for any indication that the user wants only male or female doctors:
           - "only females" or "only males"
           - "female doctors" or "male doctors"
           - "women doctors" or "men doctors"
           - "show me females" or "show me males"
           - "طبيب" -> "male"
           - "طبيبة" -> "female"
           - "دكتور" -> "male"
           - "دكتورة" -> "female"
        
        5. For specialty and subspecialty detection:
           - ALWAYS detect both specialty and subspecialty when possible
           - For dental-related:
              * Main specialty is ALWAYS "Dentistry"
              * ALWAYS use the specialty "Dentistry" when dental related terms are mentioned.
              * Look for specific dental subspecialties:
                - "braces" or "orthodontist" or "teeth alignment" -> subspeciality: "Orthodontics"
                - "root canal" or "toothache" or "tooth pain" -> subspeciality: "Endodontics"
                - "gum" or "periodontal" -> subspeciality: "Periodontics"
                - "implant" or "dental implant" -> subspeciality: "Dental Implants"
                - "crown" or "bridge" or "denture" -> subspeciality: "Prosthodontics"
                - "wisdom tooth" or "oral surgery" or "tooth extraction" -> subspeciality: "Oral Surgery"
                - "children's teeth" or "kids dentist" or "pediatric dentist" -> subspeciality: "Pediatric Dentistry"
                - "general dentist" or "routine dental checkup" -> subspeciality: "GP"
           - Examples:
              * "I need a braces doctor" -> {"speciality": "dentist", "subspeciality": "Orthodontics"}
              * "Find me a male dentist for root canal" -> {"speciality": "dentist", "subspeciality": "Endodontics", "gender": "male"}
              * "Looking for implant specialists" -> {"speciality": "dentist", "subspeciality": "Dental Implants"}
              * "I have toothache" -> {"speciality": "dentist", "subspeciality": "Endodontics"}
              * "Looking for a dentist who does braces" -> {"speciality": "dentist", "subspeciality": "Orthodontics"}
           
        6. Format subspecialty names EXACTLY according to this list for dentistry:
           - "Dental Hygienist"
           - "Dental Implants" (not "implants" or "implantology")
           - "Endodontics" (not "endodontist" or "root canal")
           - "Forensic Dentistry"
           - "GP" (for general practice)
           - "Oral Medicine Specialist"
           - "Oral Surgery"
           - "Oral and Maxillofacial Surgery"
           - "Orthodontics" (not "orthodontist" or "braces")
           - "Pediatric Dentistry" (not "pedodontist")
           - "Periodontics" (not "periodontist")
           - "Prosthodontics" (not "prosthodontist")
           - "Temporomandibular Joint TMJ Disorders"
        
        7. CRITICALLY IMPORTANT - You MUST include both speciality and subspeciality fields when a specific dental procedure is mentioned:
           * "I need braces" -> {"speciality": "Dentistry", "subspeciality": "Orthodontics"}
           * "I have tooth pain" -> {"speciality": "Dentistry", "subspeciality": "Endodontics"}
           * "Where can I get implants" -> {"speciality": "Dentistry", "subspeciality": "Dental Implants"}
        
        8- Make sure you are taking care and giving priority to user intent about what is being searched. for Examples
		   - Make sure common nouns for hospitals and clinics are stripped both in English and Arabic eg Clinic, Clinics, عيادة, عيادات or any other similar kind of words or pronouns.
		   - Try to understand of user has intention to find a certain named doctor or using some adjectives to explain it. For example, Who is best doctor in Loran clinic, مين افضل دكتور عياده ديب كير. So always focus on nouns, pro-nouns when creating quires search criteria. 

        Return ONLY a JSON object with these fields (include only if mentioned):
        {
            "speciality": "string (in English)",
            "subspeciality": "string (in English, formatted exactly as in the list above)",
            "min_price": number,
            "max_price": number,
            "min_rating": number,
            "min_experience": number,
            "doctor_name": "name (in English for phonetic matching)",
            "branch_name": "name (in original language)",
            "gender": "male" or "female"
        }
        """
        
        # Add message for extraction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        root_logger.info("********************************")
        root_logger.info("CALLING QUERY BUILDER AGENT")
        root_logger.info("********************************")
        root_logger.info("🤖 Calling GPT for criteria extraction...")
        
        # Call GPT to extract criteria with improved configuration
        response = openai_client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for better Arabic understanding
            messages=messages,
            temperature=0.3,  # Slightly higher temperature for better handling of variations
            max_tokens=500,  # Increased token limit for better extraction
            presence_penalty=0.1,  # Encourage more diverse token usage
            frequency_penalty=0.1  # Discourage repetition
        )
        
        # Log the raw response
        raw_response = response.choices[0].message.content
        root_logger.info("\n📝 Raw GPT Response:")
        root_logger.info("-"*30)
        root_logger.info(raw_response)
        root_logger.info("-"*30)
        
        # Parse the response
        try:
            extracted = json.loads(raw_response)
            root_logger.info("\n✅ Successfully parsed JSON response")
            root_logger.info("\n🔍 Extracted search criteria:")
            for key, value in extracted.items():
                root_logger.info(f"   - {key}: {value}")
            
            # Additional validation for doctor name extraction
            if "doctor_name" in extracted:
                original_name = extracted["doctor_name"]
                # Remove any remaining titles that might have been missed
                doctor_name = original_name
                titles_to_remove = ["الدكتور", "دكتور", "د.", "دكتر", "دكتوره", "دكتورة", "Dr.", "Doctor"]
                for title in titles_to_remove:
                    if doctor_name.startswith(title + " "):
                        doctor_name = doctor_name[len(title) + 1:]
                extracted["doctor_name"] = doctor_name.strip()
                
                if original_name != extracted["doctor_name"]:
                    root_logger.info(f"\n🔄 Cleaned doctor name: '{original_name}' -> '{extracted['doctor_name']}'")
                
                # Generate and print sound name for debugging
                sound_name = clean_and_generate_sound_name(extracted["doctor_name"])
                print(f"\n🔊 SOUND NAME GENERATION:")
                print(f"   Original name: '{extracted['doctor_name']}'")
                print(f"   Sound name: '{sound_name}'")
                print(f"🔊 END SOUND NAME GENERATION\n")
                
                # Log the sound name generation
                root_logger.info(f"🔊 SOUND NAME: Original='{extracted['doctor_name']}' -> Sound='{sound_name}'")
            
            # Check if we found subspecialty without specialty
            if "subspeciality" in extracted and "speciality" not in extracted:
                dental_subspecialties = ["Orthodontics", "Endodontics", "Periodontics", "Dental Implants", 
                                         "Prosthodontics", "Oral Surgery", "Oral and Maxillofacial Surgery",
                                         "Pediatric Dentistry", "GP", "Dental Hygienist"]
                
                if extracted["subspeciality"] in dental_subspecialties:
                    extracted["speciality"] = "dentist"
                    root_logger.info(f"\n🦷 Adding default specialty 'dentist' for dental subspecialty '{extracted['subspeciality']}'")
            
            root_logger.info("\n📊 Final extracted criteria:")
            root_logger.info(json.dumps(extracted, indent=2, ensure_ascii=False))
            root_logger.info("\n" + "="*50 + "\n")  # Visual separator
            
            # Also log to module logger
            logger.info(f"Extracted criteria: {json.dumps(extracted, indent=2, ensure_ascii=False)}")
            
            return extracted
            
        except json.JSONDecodeError as e:
            root_logger.error(f"\n❌ Failed to parse GPT response as JSON: {str(e)}")
            root_logger.error(f"\n❌ Raw response that failed to parse:")
            root_logger.error("-"*30)
            root_logger.error(raw_response)
            root_logger.error("-"*30)
            return {}
            
    except Exception as e:
        root_logger.error(f"\n❌ Error extracting search criteria: {str(e)}")
        return {}

def unified_doctor_search_tool(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tool wrapper for unified_doctor_search to be used by the agent.
    Searches for doctors based on criteria.
    """
    try:
        logger.info(f"TOOL CALL: unified_doctor_search with input: {input_data}")
        
        # Make sure the session_id is available in thread_local
        if hasattr(thread_local, 'session_id'):
            logger.info(f"Using existing session_id in thread_local: {thread_local.session_id}")
        else:
            # Generate a session ID if not present
            thread_local.session_id = str(uuid.uuid4())
            logger.info(f"Created new session_id in thread_local: {thread_local.session_id}")
            
        # If we're passed symptom analysis in the input, save it to thread_local
        if isinstance(input_data, dict) and 'symptom_analysis' in input_data:
            thread_local.symptom_analysis = input_data.pop('symptom_analysis')
            logger.info(f"Stored symptom analysis in thread_local")
            
        # If we're passed patient info in the input, save it to thread_local
        if isinstance(input_data, dict) and 'patient_info' in input_data:
            thread_local.patient_info = input_data.pop('patient_info')
            logger.info(f"Stored patient info in thread_local")
        
        # Call the unified search function once
        search_results = unified_doctor_search(input_data)
        # formatted_result = ensure_proper_doctor_search_format(search_results)
        logger.info("TOOL CALL: Got search results")
        
        # Extract the doctor count from the results for logging
        doctor_count = 0
        doctor_list = []
        
        # Check response structure
        logger.info(f"TOOL DEBUG: Search results keys: {list(search_results.keys()) if isinstance(search_results, dict) else 'Not a dict'}")
        
        if isinstance(search_results, dict):
            # Extract doctor data from the typical structure
            if "data" in search_results and "doctors" in search_results["data"]:
                doctor_list = search_results["data"]["doctors"]
                doctor_count = len(doctor_list)
                logger.info(f"TOOL DEBUG: Found {doctor_count} doctors in response.data")
        
        logger.info(f"TOOL RESULT: Found {doctor_count} doctors")
        
        # Store the search results in thread_local for future reference
        thread_local.last_search_results = search_results
        
        # Return results with explicit doctor count
        if isinstance(search_results, dict) and "data" in search_results:
            result = search_results["data"]
            # Add count for easier access
            if not "count" in result and "doctors" in result:
                result["count"] = len(result["doctors"])
                logger.info(f"TOOL DEBUG: Added count field with value {result['count']}")
        
        # formatted_result = ensure_proper_doctor_search_format(search_results)
        return search_results
        
    except Exception as e:
        logger.error(f"Error in unified_doctor_search_tool: {str(e)}", exc_info=True)
        return {
            "response": {
                "message": f"We are currently certifying doctors in our network. Please check back soon.",
                "patient": {
                    "Name": "",
                    "Gender": "",
                    "Location": "",
                    "Issue": "",
                    "session_id": getattr(thread_local, 'session_id', '')
                },
                "data": []
            },
            "processing_time": 0,
            "performance": {
                "total_time": 0,
                "processing_time": 0
            }
        }

def extract_search_criteria_tool(user_query: str) -> Dict[str, Any]:
    """
    Tool wrapper for criteria extraction to be used by the agent.
    Uses GPT to extract search criteria and combines with symptom analysis.
    
    Args:
        user_query: Natural language query from the user
        
    Returns:
        Dictionary with extracted search criteria
    """
    try:
        logger.info(f"TOOL CALL: extract_search_criteria with query: '{user_query[:50]}...'")
        
        # Extract criteria using GPT
        extracted = extract_search_criteria_from_message(user_query)
        
        # Get symptom analysis from thread_local if it exists
        symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
        if symptom_analysis:
            logger.info(f"Found symptom analysis in thread_local for criteria extraction")
            try:
                # Extract specialty info from symptom analysis - with safety checks
                specialties = []
                
                # Check multiple possible locations for specialty data
                if 'symptom_analysis' in symptom_analysis:
                    sa = symptom_analysis.get('symptom_analysis', {})
                    if 'matched_specialties' in sa and sa['matched_specialties']:
                        specialties = sa['matched_specialties']
                    elif 'recommended_specialties' in sa and sa['recommended_specialties']:
                        specialties = sa['recommended_specialties']
                    elif 'specialties' in sa and sa['specialties']:
                        specialties = sa['specialties']
                
                # Process the first specialty if available
                if specialties and len(specialties) > 0:
                    top_specialty = specialties[0]
                    
                    # Only add specialty if not already present in extracted criteria
                    if 'speciality' not in extracted:
                        specialty = top_specialty.get('specialty') or top_specialty.get('name')
                        if specialty:
                            extracted['speciality'] = specialty
                            logger.info(f"Added specialty '{specialty}' from symptom analysis")
                    
                    # Only add subspecialty if not already present in extracted criteria
                    if 'subspeciality' not in extracted:
                        subspecialty = top_specialty.get('subspecialty')
                        if subspecialty:
                            extracted['subspeciality'] = subspecialty
                            logger.info(f"Added subspecialty '{subspecialty}' from symptom analysis")
            except Exception as e:
                logger.error(f"Error extracting specialty info from symptom analysis: {str(e)}")
        
        # Determine what information is missing
        missing_info = []
        if not extracted.get('speciality') and not extracted.get('doctor_name') and not extracted.get('branch_name'):
            missing_info.append('specialty')
            
        # Store the extracted criteria in thread_local for later use
        thread_local.extracted_criteria = extracted
            
        return {
            "criteria": extracted,
            "needs_more_info": len(missing_info) > 0,
            "missing_info": missing_info,
            "extracted_items": list(extracted.keys()),
            "symptom_analysis_used": symptom_analysis is not None
        }
        
    except Exception as e:
        logger.error(f"Error in extract_search_criteria_tool: {str(e)}")
        return {
            "error": str(e),
            "criteria": {},
            "needs_more_info": True,
            "missing_info": ["error occurred"],
            "extracted_items": []
        }

# Export all necessary functions
__all__ = ['SearchCriteria', 'clean_subspeciality', 'generate_sound_name', 'clean_and_generate_sound_name', 'build_doctor_sound_name_clause', 'unified_doctor_search', 'unified_doctor_search_tool', 'extract_search_criteria_tool'] 