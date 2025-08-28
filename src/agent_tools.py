# Removed LangChain dependencies - using direct function definitions
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import logging
import uuid
import json
import threading

from .db import DB
import re
from decimal import Decimal
import time
from functools import wraps

# Import the patient utilities from the new package structure
from .utils import store_patient_details as store_patient_details_util, thread_local, clear_symptom_analysis_data

# Import the specialty matcher functionality
from .specialty_matcher import detect_symptoms_and_specialties

# Import the query builder functionality
from .query_builder_agent import (
    unified_doctor_search_tool, 
    extract_search_criteria_tool, 
    normalize_specialty,
    extract_search_criteria_from_message,
    unified_doctor_search,
    extract_processed_criteria
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the database
db = DB()

def execute_offers_search(search_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the offers stored procedure in parallel with doctor search
    
    Args:
        search_criteria: Dictionary containing search criteria
        
    Returns:
        Dictionary containing offers search results
    """
    try:
        logger.info("🎁 Starting offers search with criteria: %s", search_criteria)
        
        # Debug: Print the structure of search_criteria to identify any nested dictionaries
        logger.info("🎁 DEBUG: search_criteria structure:")
        for key, value in search_criteria.items():
            logger.info(f"🎁   {key}: {value} (type: {type(value)})")
            if isinstance(value, dict):
                logger.warning(f"🎁   WARNING: {key} is a dictionary: {value}")
        
        # Extract parameters for the offers stored procedure
        # Handle both direct criteria and criteria from natural language processing
        speciality = search_criteria.get('speciality') or search_criteria.get('specialty')
        subspeciality = search_criteria.get('subspeciality') or search_criteria.get('subspecialty')
        latitude = search_criteria.get('latitude')
        longitude = search_criteria.get('longitude')
        price = search_criteria.get('max_price') or search_criteria.get('min_price') or search_criteria.get('price')
        branch_name = search_criteria.get('branch_name') or search_criteria.get('hospital_name')
        
        # Log the extracted branch name for debugging
        if branch_name:
            logger.info(f"🎁 Extracted branch name: '{branch_name}'")
        else:
            logger.info(f"🎁 No branch name found in search criteria")
        
        # Convert "dentist" to "Dentistry" for the offers stored procedure
        if speciality and speciality.lower() == "dentist":
            speciality = "Dentistry"
            logger.info(f"🎁 Converted specialty from 'dentist' to 'Dentistry' for offers stored procedure")
        
        # Ensure all values are properly converted to strings or None
        # Handle cases where values might be dictionaries or other complex objects
        def safe_convert(value):
            if value is None:
                return None
            elif isinstance(value, (str, int, float)):
                return str(value)
            elif isinstance(value, dict):
                # If it's a dictionary, try to extract a meaningful string representation
                logger.warning(f"🎁 Converting dictionary to string: {value}")
                return str(value)
            else:
                # For any other type, convert to string
                logger.warning(f"🎁 Converting {type(value)} to string: {value}")
                return str(value)
        
        # Build parameters for the stored procedure
        params = {}
        
        if speciality:
            params['@speciality'] = safe_convert(speciality)
        else:
            params['@speciality'] = None
            
        if subspeciality:
            params['@subspeciality'] = safe_convert(subspeciality)
        else:
            params['@subspeciality'] = None
            
        if latitude and longitude:
            params['@lat'] = safe_convert(latitude)
            params['@long'] = safe_convert(longitude)
        else:
            params['@lat'] = None
            params['@long'] = None
            
        if price:
            params['@price'] = safe_convert(price)
        else:
            params['@price'] = None
            
        if branch_name:
            params['@branchName'] = safe_convert(branch_name)
        else:
            params['@branchName'] = None
        
        logger.info("🎁 Executing offers stored procedure with params: %s", params)
        
        # Log each parameter individually for debugging
        logger.info("🎁 DEBUG: Individual parameters being passed to offers stored procedure:")
        for param_name, param_value in params.items():
            logger.info(f"🎁   {param_name}: {param_value} (type: {type(param_value)})")
        
        # Execute the stored procedure
        offers_result = db.execute_stored_procedure(
            "[dbo].[Sp_GetOffersBySpecialityAndLocation]", 
            params
        )
        
        logger.info("🎁 Offers search completed successfully")
        
        # Store the raw offers_result for potential fallback extraction
        raw_offers_result = offers_result
        
        # Print offers results to terminal as requested
        print("\n" + "="*80)
        print("🎁 OFFERS SEARCH RESULTS")
        print("="*80)
        print(f"Speciality: {speciality}")
        print(f"Subspeciality: {subspeciality}")
        print(f"Location: lat={latitude}, long={longitude}")
        print(f"Price: {price}")
        print(f"Branch Name: {branch_name}")
        print("-"*80)
        
        offers_data = []
        if isinstance(offers_result, dict) and 'data' in offers_result:
            offers_data_dict = offers_result.get('data', {})
            
            if isinstance(offers_data_dict, dict) and 'doctors' in offers_data_dict:
                # Check if the doctors list contains offer data (has OfferId, OfferName, etc.)
                doctors_list = offers_data_dict['doctors']
                if doctors_list and isinstance(doctors_list, list) and len(doctors_list) > 0:
                    first_item = doctors_list[0]
                    if isinstance(first_item, dict) and 'OfferId' in first_item:
                        # This is offers data, not doctors data
                        offers_data = doctors_list
                        logger.info(f"🎁 [execute_offers_search] Found {len(offers_data)} offers in data.doctors")
                    else:
                        logger.info(f"🎁 [execute_offers_search] Found {len(doctors_list)} doctors in data.doctors (not offers)")
                else:
                    logger.info(f"🎁 [execute_offers_search] No doctors found in data.doctors")
            elif isinstance(offers_data_dict, list):
                # If data is directly a list, use it as offers
                offers_data = offers_data_dict
                logger.info(f"🎁 [execute_offers_search] Found {len(offers_data)} offers in data (direct list)")
            elif isinstance(offers_data_dict, dict):
                # Try to find any list in the data that might be offers
                for key, value in offers_data_dict.items():
                    if isinstance(value, list):
                        offers_data = value
                        logger.info(f"🎁 [execute_offers_search] Found offers in key '{key}': {len(offers_data)} items")
                        break
        elif isinstance(offers_result, dict):
            # Look for any list in the result that might be offers
            for key, value in offers_result.items():
                if isinstance(value, list):
                    offers_data = value
                    logger.info(f"🎁 [execute_offers_search] Found offers in key '{key}': {len(offers_data)} items")
                    break
        else:
            logger.warning("🎁 [execute_offers_search] Offers result is not a dict or doesn't contain 'data'")
        
        # If still no offers_data, try to extract from the raw offers_result
        if not offers_data and isinstance(offers_result, dict):
            # Look for any list in the entire offers_result structure
            def find_offers_in_dict(data, path=""):
                if isinstance(data, dict):
                    for key, value in data.items():
                        current_path = f"{path}.{key}" if path else key
                        if isinstance(value, list) and value:
                            # Check if this looks like offers data
                            if isinstance(value[0], dict) and any(field in value[0] for field in ['OfferId', 'OfferName', 'BeforePrice', 'AfterPrice']):
                                logger.info(f"🎁 [execute_offers_search] Found offers in {current_path}: {len(value)} items")
                                return value
                        elif isinstance(value, dict):
                            result = find_offers_in_dict(value, current_path)
                            if result:
                                return result
                return None
            
            offers_data = find_offers_in_dict(offers_result)
            if offers_data:
                logger.info(f"🎁 DEBUG: Successfully extracted {len(offers_data)} offers from raw offers_result")
        
        if offers_data:
            print(f"Found {len(offers_data)} offers:")
            for i, offer in enumerate(offers_data, 1):
                print(f"  {i}. {offer}")
            print(f"✅ These offers will be included in the API response")
            logger.info(f"🎁 DEBUG: offers_data length: {len(offers_data)}")
            logger.info(f"🎁 DEBUG: offers_data type: {type(offers_data)}")
            if offers_data:
                logger.info(f"🎁 DEBUG: First offer sample: {offers_data[0]}")
        else:
            print("No offers data found in expected format")
            logger.warning("🎁 DEBUG: offers_data is empty or None")
            logger.warning(f"🎁 DEBUG: offers_data value: {offers_data}")
            
            # Final fallback: if we have raw_offers_result but no offers_data,
            # try to extract offers from the raw result
            if raw_offers_result and isinstance(raw_offers_result, dict):
                logger.info(f"🎁 DEBUG: Attempting final fallback extraction from raw_offers_result")
                # Look for any list that might contain offers
                for key, value in raw_offers_result.items():
                    if isinstance(value, list) and value:
                        # Check if this looks like offers data
                        if isinstance(value[0], dict) and any(field in value[0] for field in ['OfferId', 'OfferName', 'BeforePrice', 'AfterPrice']):
                            offers_data = value
                            logger.info(f"🎁 DEBUG: Fallback found {len(offers_data)} offers in key '{key}'")
                            print(f"🎁 Fallback: Found {len(offers_data)} offers in raw result")
                            break
        
        print("="*80 + "\n")
        
        # Return both doctors and offers in a unified structure
        # This ensures both doctors and offers are at the same level
        if offers_data:
            logger.info(f"🎁 [execute_offers_search] RETURNING: {len(offers_data)} offers")
            result_data = {
                "data": {
                    "doctors": [],
                    "offers": offers_data
                }
            }
            return result_data
        else:
            # Final fallback: if we still don't have offers_data but we have raw_offers_result,
            # and it contains offers (as evidenced by the terminal print), let's try to extract them
            logger.info("🎁 [execute_offers_search] No offers_data found, trying fallback")
            
            # If raw_offers_result is a dict, look for any list that might be offers
            if isinstance(raw_offers_result, dict):
                for key, value in raw_offers_result.items():
                    if isinstance(value, list) and value:
                        # Check if this looks like offers data
                        if isinstance(value[0], dict) and any(field in value[0] for field in ['OfferId', 'OfferName', 'BeforePrice', 'AfterPrice']):
                            logger.info(f"🎁 [execute_offers_search] Fallback: Found {len(value)} offers in key '{key}'")
                            result_data = {
                                "data": {
                                    "doctors": [],
                                    "offers": value
                                }
                            }
                            return result_data
            
            logger.info("🎁 [execute_offers_search] No offers data found, returning empty structure")
            result_data = {
                "data": {
                    "doctors": [],
                    "offers": []
                }
            }
            return result_data
        
    except Exception as e:
        logger.error("🎁 Error in offers search: %s", str(e), exc_info=True)
        return {"error": str(e), "offers": []}

class StorePatientDetails(BaseModel):
    """Schema for storing patient details"""
    Name: Optional[str] = Field(default=None, description="Name of the patient")
    Age: Optional[int] = Field(default=None, description="Age of the patient")
    Gender: Optional[str] = Field(default=None, description="Gender of the patient")
    Location: Optional[str] = Field(default=None, description="Location of the patient")
    Issue: Optional[str] = Field(default=None, description="The Health Concerns or Symptoms of a patient")

class AnalyzeSymptomInput(BaseModel):
    """Schema for symptom analysis input"""
    symptom_description: str = Field(description="Description of symptoms or health concerns")

def store_patient_details(
    Name: Optional[str] = None,
    Age: Optional[int] = None,
    Gender: Optional[str] = None,
    Location: Optional[str] = None,
    Issue: Optional[str] = None,
    session_id: Optional[str] = None
) -> dict:
    """
    Store patient details in the session state for later use
    
    Args:
        Name: Patient's name
        Age: Patient's age
        Gender: Patient's gender
        Location: Patient's location
        Issue: Patient's health issue or concern
        session_id: Optional session identifier
        
    Returns:
        Dictionary with stored patient details
    """
    try:
        # Call the utility function with all parameters
        return store_patient_details_util(Name=Name, Age=Age, Gender=Gender, Location=Location, Issue=Issue, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Error storing patient details: {str(e)}")
        # Return whatever was passed in, as a fallback
        return {
            "Name": Name,
            "Age": Age,
            "Gender": Gender,
            "Location": Location,
            "Issue": Issue,
            "session_id": session_id,
            "error": str(e)
        }

# Function is now directly callable (removed LangChain wrapper)

def profile(func):
    """Simple profiling decorator to track function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run")
        return result
    return wrapper

# Regular function (removed LangChain decorator)
@profile
def dynamic_doctor_search(search_query: Union[str, dict]) -> dict:
    """
    Search for doctors based on various criteria.
    
    Args:
        search_query: Either a JSON string with search criteria or a natural language query
            
    Returns:
        Dictionary with search results
    """
    logger.info(f"DOCTOR SEARCH: Using session ID: {getattr(thread_local, 'session_id', 'unknown')}")
    
    # Clear previous symptom analysis data to avoid contamination
    logger.info(f"DOCTOR SEARCH: Clearing previous symptom analysis data")
    clear_symptom_analysis_data("starting new direct search")
    
    try:
        # Check if this is an offers-only query
        is_offers_only = False
        user_message = ""
        
        # Extract user message from search query
        if isinstance(search_query, str):
            try:
                criteria_dict = json.loads(search_query)
                user_message = criteria_dict.get("user_message", "").lower()
            except json.JSONDecodeError:
                user_message = search_query.lower()
        elif isinstance(search_query, dict):
            user_message = search_query.get("user_message", "").lower()
        
        # Check for offers-specific keywords in English and Arabic
        offers_keywords = [
            # English keywords
            "offers", "deals", "promotions", "discounts", "special offers", "medical offers",
            # Arabic keywords
            "عروض", "عرض", "عروض خاصة", "خصومات", "تخفيضات", "عروض طبية", "عروض صحية"
        ]
        
        # Check for doctors-only keywords in English and Arabic
        doctors_only_keywords = [
            # English keywords
            "doctors only", "specialists only", "find doctor", "need doctor", "looking for doctor",
            # Arabic keywords
            "أطباء فقط", "دكتور فقط", "أحتاج طبيب", "أبحث عن طبيب", "مختص فقط"
        ]
        
        is_offers_only = any(keyword in user_message for keyword in offers_keywords)
        is_doctors_only = any(keyword in user_message for keyword in doctors_only_keywords)
        
        if is_offers_only:
            logger.info(f"🎁 OFFERS-ONLY QUERY DETECTED: '{user_message}'")
            logger.info(f"🎁 Skipping doctor search, proceeding with offers-only search")
            
            # Extract search criteria for offers search only
            search_criteria = None
            processed_criteria = None
        elif is_doctors_only:
            logger.info(f"🔍 DOCTORS-ONLY QUERY DETECTED: '{user_message}'")
            logger.info(f"🔍 Skipping offers search, proceeding with doctors-only search")
            
            # Extract search criteria for doctors search only
        search_criteria = None
        processed_criteria = None
        
        # If this is an offers-only query, skip doctor search and only do offers search
        if is_offers_only:
            logger.info(f"🎁 OFFERS-ONLY MODE: Skipping doctor search")
            
            # Extract search criteria for offers search
            if isinstance(search_query, str):
                try:
                    criteria_dict = json.loads(search_query)
                    search_criteria = criteria_dict
                    processed_criteria = criteria_dict
                except json.JSONDecodeError:
                    # If not valid JSON, treat as natural language query
                    coordinates = {}
                    if hasattr(thread_local, 'latitude') and hasattr(thread_local, 'longitude'):
                        coordinates = {
                            "latitude": thread_local.latitude,
                            "longitude": thread_local.longitude
                        }
                    
                    search_params = {
                        "user_message": search_query,
                        **coordinates
                    }
                    search_criteria = search_params
                    processed_criteria = search_params
            else:
                search_criteria = search_query
                processed_criteria = search_query
            
            # Execute offers search only
            logger.info(f"🎁 OFFERS-ONLY: Executing offers search with criteria: {processed_criteria}")
            offers_result = execute_offers_search(processed_criteria)
            
            # Create result with only offers data
            offers_data = offers_result.get("data", {}).get("offers", []) if isinstance(offers_result, dict) else []
            offers_count = len(offers_data)
            
            result = {
                "response": {
                    "message": f"I found {offers_count} offers for you." if offers_count > 0 else "No offers found matching your criteria.",
                    "data": [],
                    "is_offers_search": True,
                    "offers": offers_data
                },
                "display_results": offers_count > 0,
                "offers_count": offers_count
            }
            
            logger.info(f"🎁 OFFERS-ONLY: Returning offers-only result with {len(result['response']['offers'])} offers")
            return result
        
        # If this is a doctors-only query, skip offers search and only do doctor search
        if is_doctors_only:
            logger.info(f"🔍 DOCTORS-ONLY MODE: Skipping offers search")
            
            # Extract search criteria for doctors search
            if isinstance(search_query, str):
                try:
                    criteria_dict = json.loads(search_query)
                    search_criteria = criteria_dict
                    processed_criteria = criteria_dict
                except json.JSONDecodeError:
                    # If not valid JSON, treat as natural language query
                    coordinates = {}
                    if hasattr(thread_local, 'latitude') and hasattr(thread_local, 'longitude'):
                        coordinates = {
                            "latitude": thread_local.latitude,
                            "longitude": thread_local.longitude
                        }
                    
                    search_params = {
                        "user_message": search_query,
                        **coordinates
                    }
                    search_criteria = search_params
                    processed_criteria = search_params
            else:
                search_criteria = search_query
                processed_criteria = search_query
            
            # Execute doctor search only
            logger.info(f"🔍 DOCTORS-ONLY: Executing doctor search with criteria: {processed_criteria}")
            doctor_result = unified_doctor_search(processed_criteria)
            
            # FIX: Correctly extract doctors from the nested response structure
            doctors_data = []
            if isinstance(doctor_result, dict):
                if "response" in doctor_result and "data" in doctor_result["response"]:
                    # Handle the actual structure: {"response": {"data": [doctors]}}
                    doctors_data = doctor_result["response"]["data"]
                    logger.info(f"🔍 DOCTORS-ONLY: Extracted {len(doctors_data)} doctors from response.data")
                elif "data" in doctor_result and "doctors" in doctor_result["data"]:
                    # Handle alternative structure: {"data": {"doctors": [doctors]}}
                    doctors_data = doctor_result["data"]["doctors"]
                    logger.info(f"🔍 DOCTORS-ONLY: Extracted {len(doctors_data)} doctors from data.doctors")
                else:
                    # Fallback: try to find doctors anywhere in the result
                    doctors_data = doctor_result.get("doctors", [])
                    if not doctors_data and "data" in doctor_result:
                        doctors_data = doctor_result["data"]
                    logger.info(f"🔍 DOCTORS-ONLY: Fallback extraction found {len(doctors_data)} doctors")
            else:
                logger.warning(f"🔍 DOCTORS-ONLY: doctor_result is not a dict: {type(doctor_result)}")
            
            doctors_count = len(doctors_data)
            logger.info(f"🔍 DOCTORS-ONLY: Final doctors count: {doctors_count}")
            
            result = {
                "response": {
                    "message": f"I found {doctors_count} doctors for you." if doctors_count > 0 else "No doctors found matching your criteria.",
                    "data": doctors_data,
                    "is_doctor_search": True,
                    "doctors_only": True,
                    "offers": []  # No offers data
                },
                "display_results": doctors_count > 0,
                "doctor_count": doctors_count
            }
            
            logger.info(f"🔍 DOCTORS-ONLY: Returning doctors-only result with {len(result['response']['data'])} doctors")
            return result
        
        # If the search_query is a string, try to parse as JSON first
        if isinstance(search_query, str):
            try:
                criteria_dict = json.loads(search_query)
                logger.info(f"Detected structured criteria in JSON format: {criteria_dict}")
                search_criteria = criteria_dict
                
                # Extract specialty information if available
                if "speciality" in criteria_dict:
                    logger.info(f"Using specialty: {criteria_dict['speciality']}")
                
                # Extract coordinates
                if "latitude" in criteria_dict and "longitude" in criteria_dict:
                    thread_local.latitude = criteria_dict["latitude"]
                    thread_local.longitude = criteria_dict["longitude"]
                    logger.info(f"Using coordinates from search params: lat={criteria_dict['latitude']}, long={criteria_dict['longitude']}")
                
                logger.info(f"Calling unified_doctor_search with criteria: {criteria_dict}")
                doctor_result = unified_doctor_search(criteria_dict)
                
                # Extract processed criteria from the result or use original criteria
                processed_criteria = criteria_dict
                
            except json.JSONDecodeError:
                # If not valid JSON, treat as natural language query
                logger.info(f"Processing as natural language query: {search_query}")
                
                # Extract coordinates if available in thread_local
                coordinates = {}
                if hasattr(thread_local, 'latitude') and hasattr(thread_local, 'longitude'):
                    coordinates = {
                        "latitude": thread_local.latitude,
                        "longitude": thread_local.longitude
                    }
                    logger.info(f"Using coordinates from thread_local: lat={thread_local.latitude}, long={thread_local.longitude}")
                
                # Create parameters for unified search
                search_params = {
                    "user_message": search_query,
                    **coordinates
                }
                search_criteria = search_params
                
                logger.info(f"Calling unified_doctor_search with text and coordinates: {search_params}")
                doctor_result = unified_doctor_search(search_params)
                
                # For natural language queries, we need to extract the processed criteria
                # The unified_doctor_search processes the criteria but doesn't return it
                # So we'll extract it from the thread_local or use the original
                processed_criteria = search_params
        else:
            # Handle direct dictionary input
            logger.info(f"Using provided dictionary: {search_query}")
            search_criteria = search_query
            doctor_result = unified_doctor_search(search_query)
            processed_criteria = search_query
        
        # Execute offers search in parallel if we have search criteria (skip for offers-only and doctors-only queries)
        offers_result = None
        if processed_criteria and not is_offers_only and not is_doctors_only:
            logger.info("🎁 Starting parallel offers search with processed criteria")
            logger.info(f"🎁 Processed criteria: {processed_criteria}")
            
            # Extract the properly processed criteria using the helper function
            # This ensures we get the same processed criteria that unified_doctor_search uses
            final_processed_criteria = extract_processed_criteria(processed_criteria)
            logger.info(f"🎁 Final processed criteria for offers: {final_processed_criteria}")
            
            # Create a thread for offers search with better error handling
            # Use a shared variable instead of thread_local for better communication
            shared_offers_result = [None]  # Use list to make it mutable
            
            def execute_offers_in_thread():
                try:
                    result = execute_offers_search(final_processed_criteria)
                    shared_offers_result[0] = result  # Store in shared variable
                    logger.info(f"🎁 [Thread] Offers search completed successfully")
                except Exception as e:
                    logger.error(f"🎁 [Thread] Error in offers search: {str(e)}")
                    shared_offers_result[0] = {"data": {"doctors": [], "offers": []}}
            
            offers_thread = threading.Thread(target=execute_offers_in_thread)
            offers_thread.start()
            offers_thread.join()  # Wait for completion
            
            # Get the offers result from shared variable
            offers_result = shared_offers_result[0]
            if offers_result is None:
                logger.error("🎁 [dynamic_doctor_search] ERROR: offers_result is None!")
            elif isinstance(offers_result, dict):
                logger.info(f"🎁 [dynamic_doctor_search] Retrieved offers result successfully")
            else:
                logger.warning(f"🎁 [dynamic_doctor_search] offers_result is not a dict: {offers_result}")
            
            logger.info("🎁 [dynamic_doctor_search] Parallel offers search completed")
        
        # Now combine doctor_result with offers_result
        if 'doctor_result' in locals() and doctor_result:
            # Use the doctor_result as the base result
            result = doctor_result
            logger.info(f"🎁 [dynamic_doctor_search] Using doctor_result as base result")
            
            # Add offers to the result if we have them
            if offers_result and isinstance(offers_result, dict):
                offers_data = offers_result.get("data", {}).get("offers", []) if isinstance(offers_result, dict) else []
                if offers_data:
                    result['offers'] = offers_data
                    logger.info(f"🎁 [dynamic_doctor_search] Added {len(offers_data)} offers to result")
                else:
                    logger.info(f"🎁 [dynamic_doctor_search] No offers data to add")
            else:
                logger.info(f"🎁 [dynamic_doctor_search] No offers_result to add")
        elif is_offers_only:
            logger.info("🎁 OFFERS-ONLY MODE: Skipping parallel offers search (already done)")
        
        # Ensure proper formatting of result (now with offers included)
        result = ensure_proper_doctor_search_format(result, str(search_query))
        
        # Check if offers are properly placed in the response object after formatting
        if isinstance(result, dict) and 'response' in result and 'offers' in result['response']:
            logger.info(f"🎁 [dynamic_doctor_search] FINAL: {len(result['response']['offers'])} offers in response")
        else:
            logger.info(f"🎁 [dynamic_doctor_search] FINAL: No offers in response")
        
        if is_offers_only:
            logger.info(f"🎁 [dynamic_doctor_search] RETURNING: Offers-only result")
        elif is_doctors_only:
            logger.info(f"🔍 [dynamic_doctor_search] RETURNING: Doctors-only result")
        else:
            logger.info(f"🎁 [dynamic_doctor_search] RETURNING: Final result (both doctors and offers)")
        
        # Ensure the function has a name attribute that matches its actual name
        dynamic_doctor_search.__name__ = "dynamic_doctor_search"
        
        return result
    except Exception as e:
        logger.error(f"Error in doctor search: {str(e)}", exc_info=True)
        return {"error": str(e), "doctors": []}
    finally:
        # Add a timestamp to the last search
        thread_local.last_search_time = time.time()

def ensure_proper_doctor_search_format(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Ensures doctor search results are returned in the correct format with separate
    message and data sections.
    
    Args:
        result: The search result to format
        query: The original query for context
        
    Returns:
        A properly formatted response dictionary
    """
    if isinstance(result, dict) and 'offers' in result:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] RECEIVED: {len(result['offers'])} offers")
    else:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] RECEIVED: No offers")
    
    # Check for nested response objects and unwrap them
    if isinstance(result, dict) and "response" in result and isinstance(result["response"], dict):
        if "response" in result["response"]:
            # Unwrap doubly-nested response
            logger.info(f"DEBUG FORMAT: Unwrapping doubly-nested response")
            result = {"response": result["response"]["response"]}
    
    # Extract data for standardized format
    doctors = []
    message = None
    patient_info = {"session_id": getattr(thread_local, 'session_id', '')}
    
    # Handle the specific format returned by unified_doctor_search_tool
    if isinstance(result, dict) and "response" in result:
        logger.info(f"DEBUG FORMAT: Found 'response' key in result")
        response_data = result.get("response", {})
        
        # Check if response_data itself contains a response key (nested responses)
        if "response" in response_data:
            logger.info(f"DEBUG FORMAT: Found nested response, unwrapping")
            response_data = response_data["response"]
            
        # Get doctors list from response
        if isinstance(response_data, dict):
            if "data" in response_data and isinstance(response_data["data"], list):
                doctors = response_data["data"]
                logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in response.data")
            
            # Get message
            if "message" in response_data:
                message = response_data["message"]
            
            # Get patient info if available
            if "patient" in response_data:
                patient_info = response_data["patient"]
                
            # Preserve offers data if it exists in the original result
            offers_data = []
            if isinstance(result, dict) and 'offers' in result:
                offers_data = result['offers']
                logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: Preserved {len(offers_data)} offers")
            else:
                logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: No offers to preserve")
            
            # Return properly formatted result with offers
            formatted_result = {
                "response": {
                    "message": message,
                    "patient": patient_info,
                    "data": doctors,
                    "doctor_count": len(doctors),
                    "is_doctor_search": True
                },
                "display_results": True,
                "doctor_count": len(doctors)
            }
            
            # Add offers to the response object
            if offers_data:
                formatted_result['response']['offers'] = offers_data
                logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: Added {len(offers_data)} offers")
            else:
                formatted_result['response']['offers'] = []
                logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: No offers added")
            
            return formatted_result
    
    # Extract doctor data from various possible locations
    if isinstance(result, dict):
        # Get doctors list
        if "data" in result and "doctors" in result["data"]:
            doctors = result["data"]["doctors"]
            logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in data.doctors")
        elif "doctors" in result:
            doctors = result["doctors"]
            logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in top level")
        elif "data" in result and isinstance(result["data"], list):
            doctors = result["data"]
            logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in data array")
        
        # Get message
        if "message" in result:
            message = result["message"]
        elif "status" in result and result["status"] == "not_found":
            message = None
        else:
            message = None
        
        # Get patient info if available
        if "patient" in result:
            patient_info = result["patient"]
        elif "data" in result and "patient" in result["data"]:
            patient_info = result["data"]["patient"]
    
    # Create the standardized response structure
    standardized_result = {
        "response": {
            "message": message,
            "patient": patient_info,
            "data": doctors,
            "doctor_count": len(doctors),
            "is_doctor_search": True
        },
        "display_results": True,
        "doctor_count": len(doctors)
    }
    
    # Preserve offers data if it exists in the original result
    offers_data = []
    if isinstance(result, dict) and 'offers' in result:
        offers_data = result['offers']
        logger.info(f"🎁 [ensure_proper_doctor_search_format] Preserved {len(offers_data)} offers")
    else:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] No offers to preserve")
    
    # Add offers to the response object at the same level as patient, message, and data
    if offers_data:
        standardized_result['response']['offers'] = offers_data
        logger.info(f"🎁 [ensure_proper_doctor_search_format] Added {len(offers_data)} offers")
    else:
        standardized_result['response']['offers'] = []
        logger.info(f"🎁 [ensure_proper_doctor_search_format] No offers added")
    
    # Ensure the structure matches the expected format with both data and offers
    if 'data' not in standardized_result:
        standardized_result['data'] = {"doctors": doctors}
    elif isinstance(standardized_result['data'], dict) and 'doctors' not in standardized_result['data']:
        standardized_result['data']['doctors'] = doctors
    
    logger.info(f"🎁 [ensure_proper_doctor_search_format] RETURNING: {len(doctors)} doctors, {len(standardized_result['response'].get('offers', []))} offers")
    return standardized_result

# Regular function (removed LangChain decorator)
@profile
def analyze_symptoms(symptom_description: str) -> Dict[str, Any]:
    """
    Analyze the patient's symptom description to determine likely medical specialties required.
    Uses the MedicalSpecialty model to determine the right specialty based on symptoms mentioned.
    
    Args:
        symptom_description: Description of symptoms from the patient
        
    Returns:
        Dictionary with list of detected symptoms and matched specialties
    """
    function_start = time.time()
    logger.info(f"DETAILED DEBUG: Starting symptom analysis for: '{symptom_description[:50]}...'")
    
    try:
        # Detect symptoms and specialties
        result = detect_symptoms_and_specialties(symptom_description)
        
        # Store in thread_local for unified search access
        thread_local.symptom_analysis = result
        logger.info(f"DETAILED DEBUG: Stored symptom analysis in thread_local for session {getattr(thread_local, 'session_id', 'unknown')}")
        
        # Create a simpler output format
        simplified_result = {}
        
        # Check if specialty is not available in database
        speciality_not_available = False
        if "symptom_analysis" in result and "speciality_not_available" in result["symptom_analysis"]:
            speciality_not_available = result["symptom_analysis"]["speciality_not_available"]
        elif "speciality_not_available" in result:
            speciality_not_available = result["speciality_not_available"]
        elif "status" in result and result["status"] == "no_matching_specialty":
            speciality_not_available = True
            
        # Add the flag to the simplified result
        simplified_result["speciality_not_available"] = speciality_not_available
        
        if speciality_not_available:
            logger.info(f"DETAILED DEBUG: Specialty not available in database for the described symptoms")
            
            # Return early with only detected symptoms but no specialties
            simplified_result["symptoms_detected"] = result.get("symptom_analysis", {}).get("detected_symptoms", [])
            if not simplified_result["symptoms_detected"] and "detected_symptoms" in result:
                simplified_result["symptoms_detected"] = result["detected_symptoms"]
                
            simplified_result["top_specialties"] = []
            simplified_result["detailed_analysis"] = result
            
            logger.info(f"DETAILED DEBUG: Final simplified result for unmatched symptoms: {simplified_result}")
            return simplified_result
        
        # Extract detected symptoms
        symptoms_detected = []
        if "symptom_analysis" in result and "detected_symptoms" in result["symptom_analysis"]:
            symptoms_detected = result["symptom_analysis"]["detected_symptoms"]
            logger.info(f"DETAILED DEBUG: Extracted symptoms from symptom_analysis.detected_symptoms: {symptoms_detected}")
        simplified_result["symptoms_detected"] = symptoms_detected
        
        # Extract specialties
        top_specialties = []
        specialties_data = []
        
        if "specialties" in result:
            specialties_data = result["specialties"]
            logger.info(f"DETAILED DEBUG: Found specialties from result.specialties")
        elif "symptom_analysis" in result and "recommended_specialties" in result["symptom_analysis"]:
            specialties_data = result["symptom_analysis"]["recommended_specialties"]
            logger.info(f"DETAILED DEBUG: Found specialties from symptom_analysis.recommended_specialties")
        
        # Convert subspecialty to subspeciality format for compatibility
        if specialties_data:
            for specialty in specialties_data:
                if "subspecialty" in specialty and "subspeciality" not in specialty:
                    specialty["subspeciality"] = specialty["subspecialty"]
            
            logger.info(f"DETAILED DEBUG: Extracted {len(specialties_data)} specialties: {specialties_data}")
            
            # Extract top specialty names
            for specialty in specialties_data:
                if "name" in specialty:
                    top_specialties.append(specialty["name"])
                elif "specialty" in specialty:
                    top_specialties.append(specialty["specialty"])
        
        simplified_result["top_specialties"] = top_specialties
        logger.info(f"DETAILED DEBUG: Top specialties: {top_specialties}")
        
        # Include the full detailed analysis
        simplified_result["detailed_analysis"] = result
        
        logger.info(f"DETAILED DEBUG: Final simplified result: {simplified_result}")
        
        # Ensure the function has a name attribute that matches its actual name
        analyze_symptoms.__name__ = "analyze_symptoms"
        
        return simplified_result
    except Exception as e:
        logger.error(f"Error in symptom analysis: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "symptoms_detected": [],
            "top_specialties": [],
            "detailed_analysis": {"error": str(e)}
        }

# Function is now directly callable (removed LangChain wrapper)
