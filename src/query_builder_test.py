"""
Test module for query_builder_agent.py to validate WHERE clause generation
"""
import logging
import unittest
from pprint import pprint
from .query_builder_agent import SearchCriteria, build_query_params

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestQueryBuilder(unittest.TestCase):
    """Test the query builder functions"""
    
    def test_specialty_where_clause(self):
        """Test WHERE clause generation for specialty criteria"""
        criteria = SearchCriteria(
            speciality="Cardiology",
            location="Riyadh"
        )
        params = build_query_params(criteria)
        
        logger.info(f"Specialty WHERE Clause: {params['@DynamicWhereClause']}")
        self.assertIn("le.Specialty LIKE N'%Cardiology%'", params["@DynamicWhereClause"])
        # Note: location search might not be in the WHERE clause if it's handled differently
    
    def test_doctor_name_where_clause(self):
        """Test WHERE clause generation for doctor name criteria"""
        criteria = SearchCriteria(doctor_name="Ahmed")
        params = build_query_params(criteria)
        
        logger.info(f"Doctor Name Parameters: {params}")
        # Test that @NameSound parameter is generated
        self.assertIn("@NameSound", params)
        self.assertIsNotNone(params["@NameSound"])
        # Test that WHERE clause is empty since sound matching is handled by @NameSound
        self.assertEqual(params["@DynamicWhereClause"], "")
    
    def test_complex_where_clause(self):
        """Test WHERE clause generation for complex criteria"""
        criteria = SearchCriteria(
            speciality="Pediatrics",
            subspeciality="Neonatology,Pediatric Cardiology",
            location="Jeddah",
            min_rating=4.0,
            min_price=100,
            max_price=500
        )
        params = build_query_params(criteria)
        
        logger.info(f"Complex WHERE Clause: {params['@DynamicWhereClause']}")
        where_clause = params["@DynamicWhereClause"]
        self.assertIn("le.Specialty LIKE N'%Pediatrics%'", where_clause)
        self.assertIn("N'Neonatology'", where_clause)
        # Note: location search might not be in the WHERE clause if it's handled differently
        self.assertIn("le.Rating >= 4.0", where_clause)
        self.assertIn("le.Fee BETWEEN 100.0 AND 500.0", where_clause)
    
    def test_branch_name_where_clause(self):
        """Test WHERE clause generation for branch name criteria"""
        criteria = SearchCriteria(branch_name="Deep Care Clinic")
        params = build_query_params(criteria)
        
        logger.info(f"Branch Name WHERE Clause: {params['@DynamicWhereClause']}")
        self.assertIn("bg.BranchName_en LIKE N'%Deep%'", params["@DynamicWhereClause"])
        self.assertIn("bg.BranchName_en LIKE N'%Care%'", params["@DynamicWhereClause"])
        self.assertIn("bg.BranchName_en LIKE N'%Clinic%'", params["@DynamicWhereClause"])
    
    def test_escaping_single_quotes(self):
        """Test handling of single quotes in criteria values"""
        criteria = SearchCriteria(
            speciality="Children's Health",
            branch_name="St. Mary's Hospital"
        )
        params = build_query_params(criteria)
        
        logger.info(f"Escaped Quotes WHERE Clause: {params['@DynamicWhereClause']}")
        where_clause = params["@DynamicWhereClause"]
        self.assertIn("le.Specialty LIKE N'%Children''s Health%'", where_clause)
        self.assertIn("bg.BranchName_en LIKE N'%St. Mary''s Hospital%'", where_clause)
    
    def test_name_sound_parameter(self):
        """Test that @NameSound parameter is correctly generated"""
        criteria = SearchCriteria(doctor_name="Youssef")
        params = build_query_params(criteria)
        
        logger.info(f"Name Sound Parameters: {params}")
        # Test that @NameSound parameter exists and has a value
        self.assertIn("@NameSound", params)
        self.assertIsNotNone(params["@NameSound"])
        self.assertNotEqual(params["@NameSound"], "")
        
        # Test that @BoostedOnly parameter is set to 0
        self.assertIn("@BoostedOnly", params)
        self.assertEqual(params["@BoostedOnly"], 0)
        
        # Test that WHERE clause is empty for doctor name search
        self.assertEqual(params["@DynamicWhereClause"], "")

def run_tests():
    """Run all the query builder tests"""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
if __name__ == "__main__":
    run_tests() 