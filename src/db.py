import os
from pprint import pprint
import logging
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import urllib
import uuid
import datetime 
from decimal import Decimal
import functools
import time
from sqlalchemy.engine import URL
from sqlalchemy.engine.base import Engine


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple cache implementation
class Cache:
    def __init__(self, max_size=100, ttl=3600):  # 1-hour TTL by default
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                # Expired entry
                del self.cache[key]
        return None
    
    def set(self, key, value):
        # Implement LRU eviction if needed
        if len(self.cache) >= self.max_size:
            # Remove oldest item (simple approach)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())

class DB:
    def __init__(self):
        """Initialize database connection and cache"""
        try:
            # Build connection string from individual parameters
            params = urllib.parse.quote_plus(
                f"DRIVER={os.getenv('DB_DRIVER')};"
                f"SERVER={os.getenv('DB_HOST')};"
                f"DATABASE={os.getenv('DB_DATABASE')};"
                f"UID={os.getenv('DB_USER')};"
                f"PWD={os.getenv('DB_PASSWORD')};"
                f"TrustServerCertificate=yes"
            )
            
            # Create SQLAlchemy connection string
            connection_string = f"mssql+pyodbc:///?odbc_connect={params}"

            test_db_url = "mssql+pyodbc://@(localdb)\\MSSQLLocalDB/DSmart_Dev?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

            
            # Configure connection pooling for better performance
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,  # Validates connections before using them
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800,  # Recycle connections after 30 minutes
                echo=False  # Disable SQL query logging in production
            )
            
            # Initialize cache with 1-hour TTL
            self.cache = Cache(max_size=100, ttl=3600)
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            logger.error("Please ensure the following environment variables are set:")
            logger.error("- DB_HOST: The database server hostname")
            logger.error("- DB_DATABASE: The database name")
            logger.error("- DB_USER: The database username")
            logger.error("- DB_PASSWORD: The database password")
            raise

    def __del__(self):
        print("Connection to the MySQL database closed.")

    def get_available_specialties(self) -> List[str]:
        """
        Get a list of all available medical specialties from the database.
        
        Returns:
            List of specialty names as strings
        """
        cache_key = "available_specialties"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for {cache_key}")
            return cached_result
        
        try:
            cursor = self.engine.connect()
            # Get distinct specialty names from the Speciality table
            query = "SELECT DISTINCT SpecialityName FROM Speciality WHERE SpecialityName IS NOT NULL"
            result = cursor.execute(text(query))
            
            # Extract the specialty names from the result
            specialties = [row['SpecialityName'] for row in result.mappings()]
            cursor.close()
            
            # Store in cache
            self.cache.set(cache_key, specialties)
            
            logger.info(f"Retrieved {len(specialties)} specialties from database")
            return specialties
            
        except Exception as e:
            logger.error(f"Error retrieving specialties: {e}")
            return []
    
    def get_specialty_subspecialty_mapping(self) -> Dict[str, str]:
        """
        Get a mapping of subspecialties to their parent specialties.
        
        Returns:
            Dictionary mapping subspecialty names to their parent specialty names
        """
        cache_key = "specialty_subspecialty_mapping"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for {cache_key}")
            return cached_result
        
        try:
            cursor = self.engine.connect()
            # Get subspecialties and their parent specialties
            query = "SELECT SpecialityName, SubSpeciality FROM Speciality WHERE SubSpeciality IS NOT NULL"
            result = cursor.execute(text(query))
            
            # Build the mapping
            mapping = {}
            for row in result.mappings():
                specialty = row['SpecialityName']
                subspecialty = row['SubSpeciality']
                if specialty and subspecialty:
                    mapping[subspecialty] = specialty
            
            cursor.close()
            
            # Store in cache
            self.cache.set(cache_key, mapping)
            
            logger.info(f"Retrieved mapping for {len(mapping)} subspecialties")
            return mapping
            
        except Exception as e:
            logger.error(f"Error retrieving specialty-subspecialty mapping: {e}")
            return {}
            
    def get_doctor_name_by_speciality(
        self, 
        speciality: str, 
        location: str,
        min_rating: float = None,
        max_price: float = None,
        min_price: float = None,
        min_experience: float = None
    ) -> list[dict[str, str | int | float | bool]]:
        # Create a cache key including the filter parameters
        cache_key = f"{speciality}:{location}:{min_rating}:{min_price}:{max_price}:{min_experience}"
        
        # Check if result is in cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for {cache_key}")
            return cached_result
        
        try:
            start_time = time.time()
            
            # Build the base query
            base_query = """
                SELECT TOP 5 
                le.Id AS DoctorId,
                le.DocName_en AS DoctorName_en,
                le.DocName_ar AS DoctorName_ar,
                le.Specialty AS Speciality,
                le.Fee AS Fee,
                CAST(NULLIF(le.Rating, '') AS FLOAT) AS Rating,
                le.Experience AS Experience,
                le.HasDiscount AS HasDiscount,
                b.BranchName_en AS Branch_en,
                b.BranchName_ar AS Branch_ar,
                b.Address_en AS Address_en,
                b.Address_ar AS Address_ar,
                d.DiscountType AS DiscountType,
                d.DiscountValue AS DiscountValue
                FROM [dbo].[LowerEntity] le
                JOIN [dbo].[Branch] b ON le.BranchId = b.Id
                LEFT JOIN [dbo].[Discount] d ON le.discount_id = d.discount_id
                WHERE le.Specialty LIKE :speciality
                AND (b.Address_en LIKE :location OR b.Address_ar LIKE :location)
                AND le.isActive = 1
            """
            
            # Build parameters dictionary
            params = {
                'speciality': f"%{speciality}%",
                'location': f"%{location}%"
            }
            
            # Add optional filters
            if min_rating is not None:
                base_query += " AND CAST(NULLIF(le.Rating, '') AS FLOAT) >= :min_rating"
                params['min_rating'] = min_rating
            
            if min_price is not None:
                base_query += " AND CAST(NULLIF(le.Fee, '') AS FLOAT) >= :min_price"
                params['min_price'] = min_price
                
            if max_price is not None:
                base_query += " AND CAST(NULLIF(le.Fee, '') AS FLOAT) <= :max_price"
                params['max_price'] = max_price
                
            if min_experience is not None:
                base_query += " AND le.Experience >= :min_experience"
                params['min_experience'] = min_experience
            
            # Add ordering
            base_query += """
                ORDER BY 
                CASE WHEN le.Rating IS NULL THEN 1 ELSE 0 END,
                CAST(NULLIF(le.Rating, '') AS FLOAT) DESC,
                CAST(NULLIF(le.Fee, '') AS FLOAT) ASC
            """
            
            cursor = self.engine.connect()
            result = cursor.execute(text(base_query), params)
            records = [dict(row) for row in result.mappings()]
            cursor.close()
            
            # Store in cache
            self.cache.set(cache_key, records)
            
            query_time = time.time() - start_time
            logger.info(f"Database query took {query_time:.2f} seconds")

            return records
        except Exception as e:
            logger.error(f"Error retrieving doctors for specialty: {e}")
            return []

    def search_doctors_dynamic(self, query: str, params: dict) -> List[dict]:
        """
        Execute dynamic doctor search query based on a fully formed SQL query
        
        Args:
            query: The complete SQL query string
            params: Dictionary of query parameters
            
        Returns:
            List of doctor dictionaries
        """
        try:
            logger.info("Executing direct SQL query from query builder")
            
            if not query or not query.strip():
                logger.info("Empty query received, returning no results")
                return []
            
            start_time = time.time()
            
            # Execute the prepared query directly
            with self.engine.connect() as connection:
                try:
                    # Execute the query with parameters
                    logger.info(f"Executing query: {query}")
                    logger.info(f"With parameters: {params}")
                    
                    result = connection.execute(text(query), params)
                    
                    # Convert the results to dictionaries
                    rows = [dict(row) for row in result.mappings()]
                    
                    end_time = time.time()
                    logger.info(f"Query executed successfully in {end_time - start_time:.2f} seconds")
                    logger.info(f"Found {len(rows)} matching doctors")
                    
                    return rows
                    
                except Exception as query_error:
                    logger.error(f"SQL execution error: {str(query_error)}")
                    logger.error(f"Failed query: {query}")
                    logger.error(f"Failed parameters: {params}")
                    # Try to continue with empty results rather than failing completely
                    return []
            
        except Exception as e:
            logger.error(f"General error in search_doctors_dynamic: {str(e)}")
            logger.error(f"Failed query: {query}")
            logger.error(f"Failed parameters: {params}")
            return []

    def execute_stored_procedure(self, proc_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a stored procedure with parameters
        
        Args:
            proc_name: Name of the stored procedure
            params: Dictionary of parameter names and values
            
        Returns:
            List of dictionaries containing the result rows
        """
        try:
            logger.info(f"Executing stored procedure {proc_name} with params: {params}")
            
            # Debug: Log each parameter value and its type
            logger.info("DEBUG: Parameter types:")
            for k, v in params.items():
                logger.info(f"  {k}: {v} (type: {type(v)})")
                if isinstance(v, dict):
                    logger.error(f"  ERROR: {k} is a dictionary: {v}")
            
            # Build the EXEC statement with proper quoting for string parameters
            param_parts = []
            for k, v in params.items():
                if v is None:
                    # Handle None values by passing NULL
                    param_parts.append(f"{k}=NULL")
                elif isinstance(v, str):
                    # For string parameters, escape any single quotes by doubling them
                    # and wrap the entire value in single quotes
                    escaped_value = v.replace("'", "''")
                    param_parts.append(f"{k}=N'{escaped_value}'")
                elif isinstance(v, dict):
                    # Handle dictionary values by converting to string representation
                    logger.warning(f"Converting dictionary parameter {k} to string: {v}")
                    dict_str = str(v)
                    escaped_value = dict_str.replace("'", "''")
                    param_parts.append(f"{k}=N'{escaped_value}'")
                else:
                    param_parts.append(f"{k}={v}")
            
            param_str = ", ".join(param_parts)
            exec_statement = f"EXEC {proc_name} {param_str}"
            
            logger.info(f"Executing SQL: {exec_statement}")
            
            # Execute the stored procedure
            with self.engine.connect() as connection:
                result = connection.execute(text(exec_statement))
                
                # Convert result to list of dictionaries using mappings()
                rows = [dict(row) for row in result.mappings()]
                
                # Convert special types
                for row in rows:
                    for key, value in row.items():
                        if isinstance(value, (datetime.datetime, datetime.date)):
                            row[key] = value.isoformat()
                        elif isinstance(value, Decimal):
                            row[key] = float(value)
                
                logger.info(f"Got {len(rows)} results from stored procedure")
                
                # Add detailed logging to inspect the data structure
                if rows:
                    logger.info(f"Sample first row keys: {list(rows[0].keys())}")
                    
                    # Check if this is offers data or doctors data based on the keys
                    first_row = rows[0]
                    is_offers_data = 'OfferId' in first_row
                    is_doctors_data = 'DoctorId' in first_row
                    
                    logger.info(f"Data type detection: is_offers_data={is_offers_data}, is_doctors_data={is_doctors_data}")
                    
                    if is_offers_data:
                        # This is offers data, return in data.offers structure
                        result_dict = {
                            "data": {
                                "doctors": [],
                                "offers": rows
                            }
                        }
                        logger.info(f"Formatted result with data.offers structure containing {len(rows)} offers")
                        return result_dict
                    elif is_doctors_data:
                        # This is doctors data, return in data.doctors structure
                        result_dict = {
                            "data": {
                                "doctors": rows,
                                "offers": []
                            }
                        }
                        logger.info(f"Formatted result with data.doctors structure containing {len(rows)} doctors")
                        return result_dict
                    else:
                        # Unknown data type, default to doctors structure
                        logger.warning(f"Unknown data type, defaulting to doctors structure. Keys: {list(first_row.keys())}")
                        result_dict = {
                            "data": {
                                "doctors": rows,
                                "offers": []
                            }
                        }
                        logger.info(f"Formatted result with default data.doctors structure containing {len(rows)} items")
                        return result_dict
                else:
                    logger.info("No rows returned, returning empty result dict")
                    return {"data": {"doctors": [], "offers": []}}
                
        except Exception as e:
            logger.error(f"Error executing stored procedure: {str(e)}")
            return {"data": {"doctors": [], "offers": []}}  # Return structured empty result

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results
        This is a wrapper around execute_stored_procedure for backward compatibility
        """
        if query.strip().upper().startswith("[DBO].[SPDYAMICQUERYBUILDER]"):
            # Extract stored procedure name and parameters
            return self.execute_stored_procedure(query, params or {})
        else:
            logger.warning("Direct SQL queries are deprecated. Please use stored procedures.")
            try:
                with self.engine.connect() as connection:
                    result = connection.execute(text(query), params or {})
                    return [dict(row) for row in result]
            except Exception as e:
                logger.error(f"Error executing query: {str(e)}")
                raise

if __name__ == "__main__":
    db = DB()
    specialties = db.get_available_specialties()
    print("Available Specialities:")
    pprint(specialties)

    mapping = db.get_specialty_subspecialty_mapping()
    print("Subspecialty to Specialty Mapping:")
    pprint(mapping)

    print("Doctors for speciality: Dentistry in Jeddah")
    docs = db.get_doctor_name_by_speciality("DENTISTRY", 'Jeddah')
    pprint(docs)
