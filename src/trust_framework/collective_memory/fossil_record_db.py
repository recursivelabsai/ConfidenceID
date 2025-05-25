"""
Collective Trust Memory: Fossil Record Database

This module implements the Fossil Record Database for the Collective Trust Memory
component of ConfidenceID. It provides a persistent storage for verification "fossils" -
records of verification events with rich metadata that can be analyzed to identify
patterns across time and context.

The implementation is based on the collective memory theory described in
claude.metalayer.txt (Layer 8.2).
"""

import sqlite3
import json
import uuid
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import numpy as np
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FossilRecordDB:
    """
    A database for storing, retrieving, and querying verification "fossils".
    
    The Fossil Record DB maintains a distributed ledger of verification events
    with rich metadata, enabling pattern recognition across verification history
    through the "Trust Archaeology" system.
    """
    
    def __init__(self, db_path: str = "fossil_record.db"):
        """
        Initialize the Fossil Record Database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create the main fossils table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_fossils (
                fossil_id TEXT PRIMARY KEY,
                content_fingerprint TEXT,
                verification_score REAL,
                timestamp TEXT,
                content_type TEXT,
                verification_method TEXT,
                verifier_id TEXT,
                context_vector BLOB,
                metadata TEXT,
                created_at TEXT
            )
            ''')
            
            # Create indices for common query patterns
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_fingerprint ON verification_fossils(content_fingerprint)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON verification_fossils(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON verification_fossils(content_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_verification_method ON verification_fossils(verification_method)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_verifier_id ON verification_fossils(verifier_id)')
            
            # Create the patterns table for storing identified patterns
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_description TEXT,
                confidence_score REAL,
                related_fossils TEXT,
                metadata TEXT,
                detected_at TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Fossil Record DB initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Fossil Record DB: {e}")
            raise
    
    def store_fossil(self, 
                    content_fingerprint: str,
                    verification_score: float,
                    timestamp: Union[str, datetime],
                    content_type: str,
                    verification_method: str,
                    verifier_id: Optional[str] = None,
                    context_vector: Optional[np.ndarray] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a verification fossil in the database.
        
        Args:
            content_fingerprint: Privacy-preserving hash of the content
            verification_score: Score from the verification (0-1)
            timestamp: When the verification occurred
            content_type: Type of content (e.g., 'text', 'image', 'audio', 'video')
            verification_method: Method used for verification (e.g., 'watermark', 'perplexity')
            verifier_id: Optional ID of the verifier (e.g., node ID in decentralized network)
            context_vector: Optional vector representing the context of verification
            metadata: Optional additional metadata about the verification
            
        Returns:
            The unique fossil ID
        """
        try:
            # Generate a unique fossil ID
            fossil_id = str(uuid.uuid4())
            
            # Process timestamp
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = timestamp
            
            # Process context vector
            if context_vector is not None:
                context_blob = context_vector.tobytes()
            else:
                context_blob = None
            
            # Process metadata
            if metadata is not None:
                metadata_json = json.dumps(metadata)
            else:
                metadata_json = json.dumps({})
            
            # Store the fossil
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO verification_fossils
            (fossil_id, content_fingerprint, verification_score, timestamp, content_type,
             verification_method, verifier_id, context_vector, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fossil_id,
                content_fingerprint,
                verification_score,
                timestamp_str,
                content_type,
                verification_method,
                verifier_id,
                context_blob,
                metadata_json,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored verification fossil with ID: {fossil_id}")
            return fossil_id
        
        except Exception as e:
            logger.error(f"Error storing verification fossil: {e}")
            raise
    
    def get_fossil(self, fossil_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a verification fossil by ID.
        
        Args:
            fossil_id: The unique ID of the fossil
            
        Returns:
            The fossil data as a dictionary, or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT fossil_id, content_fingerprint, verification_score, timestamp, content_type,
                   verification_method, verifier_id, context_vector, metadata, created_at
            FROM verification_fossils
            WHERE fossil_id = ?
            ''', (fossil_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            # Process context vector
            context_vector = None
            if row[7] is not None:  # context_vector
                context_vector = np.frombuffer(row[7])
            
            # Process metadata
            metadata = json.loads(row[8])
            
            # Create the fossil dictionary
            fossil = {
                'fossil_id': row[0],
                'content_fingerprint': row[1],
                'verification_score': row[2],
                'timestamp': row[3],
                'content_type': row[4],
                'verification_method': row[5],
                'verifier_id': row[6],
                'context_vector': context_vector,
                'metadata': metadata,
                'created_at': row[9]
            }
            
            return fossil
        
        except Exception as e:
            logger.error(f"Error retrieving fossil {fossil_id}: {e}")
            return None
    
    def query_fossils(self, 
                     content_fingerprint: Optional[str] = None,
                     content_type: Optional[str] = None,
                     verification_method: Optional[str] = None,
                     verifier_id: Optional[str] = None,
                     min_score: Optional[float] = None,
                     max_score: Optional[float] = None,
                     start_time: Optional[Union[str, datetime]] = None,
                     end_time: Optional[Union[str, datetime]] = None,
                     limit: int = 100,
                     offset: int = 0) -> List[Dict[str, Any]]:
        """
        Query verification fossils based on various criteria.
        
        Args:
            content_fingerprint: Optional content fingerprint to filter by
            content_type: Optional content type to filter by
            verification_method: Optional verification method to filter by
            verifier_id: Optional verifier ID to filter by
            min_score: Optional minimum verification score
            max_score: Optional maximum verification score
            start_time: Optional start time for time range
            end_time: Optional end time for time range
            limit: Maximum number of fossils to return
            offset: Offset for pagination
            
        Returns:
            List of matching fossils as dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build the query
            query = '''
            SELECT fossil_id, content_fingerprint, verification_score, timestamp, content_type,
                   verification_method, verifier_id, context_vector, metadata, created_at
            FROM verification_fossils
            WHERE 1=1
            '''
            params = []
            
            # Add filters
            if content_fingerprint is not None:
                query += " AND content_fingerprint = ?"
                params.append(content_fingerprint)
            
            if content_type is not None:
                query += " AND content_type = ?"
                params.append(content_type)
            
            if verification_method is not None:
                query += " AND verification_method = ?"
                params.append(verification_method)
            
            if verifier_id is not None:
                query += " AND verifier_id = ?"
                params.append(verifier_id)
            
            if min_score is not None:
                query += " AND verification_score >= ?"
                params.append(min_score)
            
            if max_score is not None:
                query += " AND verification_score <= ?"
                params.append(max_score)
            
            if start_time is not None:
                if isinstance(start_time, datetime):
                    start_time = start_time.isoformat()
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                if isinstance(end_time, datetime):
                    end_time = end_time.isoformat()
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            # Add order by, limit, and offset
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute the query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Process the results
            fossils = []
            for row in rows:
                # Process context vector
                context_vector = None
                if row[7] is not None:  # context_vector
                    context_vector = np.frombuffer(row[7])
                
                # Process metadata
                metadata = json.loads(row[8])
                
                # Create the fossil dictionary
                fossil = {
                    'fossil_id': row[0],
                    'content_fingerprint': row[1],
                    'verification_score': row[2],
                    'timestamp': row[3],
                    'content_type': row[4],
                    'verification_method': row[5],
                    'verifier_id': row[6],
                    'context_vector': context_vector,
                    'metadata': metadata,
                    'created_at': row[9]
                }
                
                fossils.append(fossil)
            
            return fossils
        
        except Exception as e:
            logger.error(f"Error querying fossils: {e}")
            return []
    
    def store_pattern(self, 
                     pattern_type: str,
                     pattern_description: str,
                     confidence_score: float,
                     related_fossils: List[str],
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a verification pattern in the database.
        
        Patterns are higher-level structures identified by analyzing multiple
        verification fossils, such as trends, anomalies, or recurring verification patterns.
        
        Args:
            pattern_type: Type of pattern (e.g., 'trend', 'anomaly', 'recurrence')
            pattern_description: Human-readable description of the pattern
            confidence_score: Confidence in the pattern (0-1)
            related_fossils: List of fossil IDs related to this pattern
            metadata: Optional additional metadata about the pattern
            
        Returns:
            The unique pattern ID
        """
        try:
            # Generate a unique pattern ID
            pattern_id = str(uuid.uuid4())
            
            # Process related fossils
            related_fossils_json = json.dumps(related_fossils)
            
            # Process metadata
            if metadata is not None:
                metadata_json = json.dumps(metadata)
            else:
                metadata_json = json.dumps({})
            
            # Store the pattern
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO verification_patterns
            (pattern_id, pattern_type, pattern_description, confidence_score, related_fossils, metadata, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                pattern_type,
                pattern_description,
                confidence_score,
                related_fossils_json,
                metadata_json,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored verification pattern with ID: {pattern_id}")
            return pattern_id
        
        except Exception as e:
            logger.error(f"Error storing verification pattern: {e}")
            raise
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a verification pattern by ID.
        
        Args:
            pattern_id: The unique ID of the pattern
            
        Returns:
            The pattern data as a dictionary, or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT pattern_id, pattern_type, pattern_description, confidence_score,
                   related_fossils, metadata, detected_at
            FROM verification_patterns
            WHERE pattern_id = ?
            ''', (pattern_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            # Process related fossils
            related_fossils = json.loads(row[4])
            
            # Process metadata
            metadata = json.loads(row[5])
            
            # Create the pattern dictionary
            pattern = {
                'pattern_id': row[0],
                'pattern_type': row[1],
                'pattern_description': row[2],
                'confidence_score': row[3],
                'related_fossils': related_fossils,
                'metadata': metadata,
                'detected_at': row[6]
            }
            
            return pattern
        
        except Exception as e:
            logger.error(f"Error retrieving pattern {pattern_id}: {e}")
            return None
    
    def query_patterns(self, 
                      pattern_type: Optional[str] = None,
                      min_confidence: Optional[float] = None,
                      max_confidence: Optional[float] = None,
                      related_fossil_id: Optional[str] = None,
                      limit: int = 100,
                      offset: int = 0) -> List[Dict[str, Any]]:
        """
        Query verification patterns based on various criteria.
        
        Args:
            pattern_type: Optional pattern type to filter by
            min_confidence: Optional minimum confidence score
            max_confidence: Optional maximum confidence score
            related_fossil_id: Optional filter to patterns related to a specific fossil
            limit: Maximum number of patterns to return
            offset: Offset for pagination
            
        Returns:
            List of matching patterns as dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build the query
            query = '''
            SELECT pattern_id, pattern_type, pattern_description, confidence_score,
                   related_fossils, metadata, detected_at
            FROM verification_patterns
            WHERE 1=1
            '''
            params = []
            
            # Add filters
            if pattern_type is not None:
                query += " AND pattern_type = ?"
                params.append(pattern_type)
            
            if min_confidence is not None:
                query += " AND confidence_score >= ?"
                params.append(min_confidence)
            
            if max_confidence is not None:
                query += " AND confidence_score <= ?"
                params.append(max_confidence)
            
            # Note: this is a more complex filter since related_fossils is stored as JSON
            if related_fossil_id is not None:
                # This is a simple approach - for production, you might want to use a more efficient method
                query += " AND related_fossils LIKE ?"
                params.append(f'%"{related_fossil_id}"%')
            
            # Add order by, limit, and offset
            query += " ORDER BY detected_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute the query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Process the results
            patterns = []
            for row in rows:
                # Process related fossils
                related_fossils = json.loads(row[4])
                
                # Process metadata
                metadata = json.loads(row[5])
                
                # Create the pattern dictionary
                pattern = {
                    'pattern_id': row[0],
                    'pattern_type': row[1],
                    'pattern_description': row[2],
                    'confidence_score': row[3],
                    'related_fossils': related_fossils,
                    'metadata': metadata,
                    'detected_at': row[6]
                }
                
                patterns.append(pattern)
            
            return patterns
        
        except Exception as e:
            logger.error(f"Error querying patterns: {e}")
            return []
    
    def get_fossils_by_fingerprint(self, content_fingerprint: str) -> List[Dict[str, Any]]:
        """
        Get all verification fossils for a specific content fingerprint.
        
        This is useful for tracing the verification history of a specific piece of content.
        
        Args:
            content_fingerprint: The content fingerprint to look up
            
        Returns:
            List of fossils related to the content fingerprint
        """
        return self.query_fossils(content_fingerprint=content_fingerprint)
    
    def get_verification_history_stats(self, 
                                      start_time: Optional[Union[str, datetime]] = None,
                                      end_time: Optional[Union[str, datetime]] = None,
                                      group_by: str = 'content_type') -> Dict[str, Any]:
        """
        Get statistical summaries of verification history.
        
        Args:
            start_time: Optional start time for time range
            end_time: Optional end time for time range
            group_by: Field to group by ('content_type', 'verification_method', 'verifier_id')
            
        Returns:
            Dictionary with statistical summaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Validate group_by
            valid_group_by = ['content_type', 'verification_method', 'verifier_id']
            if group_by not in valid_group_by:
                group_by = 'content_type'
            
            # Build the query
            query = f'''
            SELECT {group_by}, 
                   COUNT(*) as count, 
                   AVG(verification_score) as avg_score,
                   MIN(verification_score) as min_score,
                   MAX(verification_score) as max_score
            FROM verification_fossils
            WHERE 1=1
            '''
            params = []
            
            # Add time filters
            if start_time is not None:
                if isinstance(start_time, datetime):
                    start_time = start_time.isoformat()
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                if isinstance(end_time, datetime):
                    end_time = end_time.isoformat()
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            # Add group by
            query += f" GROUP BY {group_by}"
            
            # Execute the query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Process the results
            stats = {
                'total_fossils': 0,
                'groups': {}
            }
            
            for row in rows:
                group_value = row[0]
                count = row[1]
                avg_score = row[2]
                min_score = row[3]
                max_score = row[4]
                
                stats['total_fossils'] += count
                stats['groups'][group_value] = {
                    'count': count,
                    'avg_score': avg_score,
                    'min_score': min_score,
                    'max_score': max_score
                }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting verification history stats: {e}")
            return {'total_fossils': 0, 'groups': {}}
    
    def generate_content_fingerprint(self, content: Union[str, bytes]) -> str:
        """
        Generate a privacy-preserving fingerprint for content.
        
        This is a utility method to create consistent fingerprints for content
        that can be used to track verification history without exposing the original content.
        
        Args:
            content: The content to fingerprint (string or bytes)
            
        Returns:
            A privacy-preserving hash of the content
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        # Use SHA-256 for the fingerprint
        return hashlib.sha256(content).hexdigest()
    
    def close(self) -> None:
        """Close any open database connections."""
        # SQLite doesn't keep persistent connections in this implementation
        pass
