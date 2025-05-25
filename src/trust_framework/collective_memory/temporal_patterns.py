"""
Collective Trust Memory: Temporal Patterns

This module implements temporal pattern analysis for the Collective Trust Memory component
of ConfidenceID. It provides specialized algorithms for identifying patterns in verification
events over time, such as recurring verification behaviors, temporal trends, and cyclical patterns.

The implementation is based on the collective memory theory described in
claude.metalayer.txt (Layer 8.2).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import logging
from collections import Counter, defaultdict
import json
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import math

from .fossil_record_db import FossilRecordDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalPatternAnalyzer:
    """
    Analyzes temporal patterns in verification events from the fossil record.
    
    This class specializes in identifying patterns that emerge over time, including
    trends, cycles, seasonality, and temporal relationships between different
    verification methods, content types, and verifiers.
    """
    
    def __init__(self, fossil_db: FossilRecordDB):
        """
        Initialize the Temporal Pattern Analyzer.
        
        Args:
            fossil_db: The Fossil Record Database to analyze
        """
        self.fossil_db = fossil_db
    
    def analyze_temporal_evolution(self, 
                                  content_fingerprint: Optional[str] = None,
                                  content_type: Optional[str] = None,
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None,
                                  min_fossils: int = 20) -> Dict[str, Any]:
        """
        Analyze the temporal evolution of verification patterns.
        
        This method performs a comprehensive analysis of how verification patterns
        have evolved over time, identifying trends, cycles, and other temporal patterns.
        
        Args:
            content_fingerprint: Optional content fingerprint to focus analysis on
            content_type: Optional content type to focus analysis on
            start_time: Optional start time for analysis
            end_time: Optional end time for analysis
            min_fossils: Minimum number of fossils required for analysis
            
        Returns:
            Dictionary containing temporal evolution analysis
        """
        # Retrieve fossils for analysis
        query_kwargs = {}
        if content_fingerprint:
            query_kwargs['content_fingerprint'] = content_fingerprint
        if content_type:
            query_kwargs['content_type'] = content_type
        if start_time:
            query_kwargs['start_time'] = start_time
        if end_time:
            query_kwargs['end_time'] = end_time
        
        # Set a reasonable limit to avoid memory issues
        query_kwargs['limit'] = 10000
        
        fossils = self.fossil_db.query_fossils(**query_kwargs)
        
        if len(fossils) < min_fossils:
            logger.warning(f"Insufficient fossils for temporal analysis: {len(fossils)} < {min_fossils}")
            return {
                'status': 'insufficient_data',
                'fossils_count': len(fossils),
                'min_required': min_fossils
            }
        
        # Convert to DataFrame
        df = self._fossils_to_dataframe(fossils)
        
        # Perform temporal analyses
        trend_analysis = self._analyze_trends(df)
        seasonal_analysis = self._analyze_seasonality(df)
        event_density_analysis = self._analyze_event_density(df)
        periodicity_analysis = self._analyze_periodicity(df)
        method_evolution_analysis = self._analyze_method_evolution(df)
        verifier_evolution_analysis = self._analyze_verifier_evolution(df)
        
        # Combine results
        return {
            'status': 'success',
            'fossils_analyzed': len(fossils),
            'time_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'trend_analysis': trend_analysis,
            'seasonal_analysis': seasonal_analysis,
            'event_density_analysis': event_density_analysis,
            'periodicity_analysis': periodicity_analysis,
            'method_evolution': method_evolution_analysis,
            'verifier_evolution': verifier_evolution_analysis
        }
    
    def detect_content_lifecycle_patterns(self,
                                         content_type: Optional[str] = None,
                                         min_content_items: int = 10,
                                         min_fossils_per_item: int = 5) -> Dict[str, Any]:
        """
        Detect patterns in the lifecycle of content verification.
        
        This method analyzes how verification patterns evolve over the lifecycle
        of content items, from initial verification to long-term trust establishment.
        
        Args:
            content_type: Optional content type to focus analysis on
            min_content_items: Minimum number of distinct content items required
            min_fossils_per_item: Minimum fossils per content item
            
        Returns:
            Dictionary containing content lifecycle pattern analysis
        """
        # Retrieve fossils for analysis
        query_kwargs = {}
        if content_type:
            query_kwargs['content_type'] = content_type
        
        # Set a reasonable limit to avoid memory issues
        query_kwargs['limit'] = 10000
        
        fossils = self.fossil_db.query_fossils(**query_kwargs)
        
        if len(fossils) < min_content_items * min_fossils_per_item:
            logger.warning(f"Insufficient fossils for lifecycle analysis: {len(fossils)}")
            return {
                'status': 'insufficient_data',
                'fossils_count': len(fossils),
                'min_required': min_content_items * min_fossils_per_item
            }
        
        # Convert to DataFrame
        df = self._fossils_to_dataframe(fossils)
        
        # Group by content fingerprint
        content_groups = df.groupby('content_fingerprint')
        
        # Filter to content items with sufficient fossils
        valid_content = [group for name, group in content_groups if len(group) >= min_fossils_per_item]
        
        if len(valid_content) < min_content_items:
            logger.warning(f"Insufficient content items with enough fossils: {len(valid_content)} < {min_content_items}")
            return {
                'status': 'insufficient_data',
                'content_items_with_sufficient_fossils': len(valid_content),
                'min_required': min_content_items
            }
        
        # Analyze lifecycle patterns
        initial_verification_patterns = self._analyze_initial_verification(valid_content)
        evolution_patterns = self._analyze_verification_evolution(valid_content)
        convergence_patterns = self._analyze_verification_convergence(valid_content)
        method_transition_patterns = self._analyze_method_transitions(valid_content)
        
        # Combine results
        return {
            'status': 'success',
            'content_items_analyzed': len(valid_content),
            'total_fossils': sum(len(content) for content in valid_content),
            'initial_verification_patterns': initial_verification_patterns,
            'evolution_patterns': evolution_patterns,
            'convergence_patterns': convergence_patterns,
            'method_transition_patterns': method_transition_patterns
        }
    
    def identify_verification_cascades(self,
                                      max_time_gap: timedelta = timedelta(hours=24),
                                      min_cascade_size: int = 5,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Identify cascades of verification events that propagate across content items.
        
        This method detects patterns where verification of one content item triggers
        verification of related items within a short time window, forming a cascade.
        
        Args:
            max_time_gap: Maximum time gap between events in a cascade
            min_cascade_size: Minimum number of events to constitute a cascade
            start_time: Optional start time for analysis
            end_time: Optional end time for analysis
            
        Returns:
            Dictionary containing verification cascade analysis
        """
        # Retrieve fossils for analysis
        query_kwargs = {}
        if start_time:
            query_kwargs['start_time'] = start_time
        if end_time:
            query_kwargs['end_time'] = end_time
        
        # Set a reasonable limit to avoid memory issues
        query_kwargs['limit'] = 10000
        
        fossils = self.fossil_db.query_fossils(**query_kwargs)
        
        if len(fossils) < min_cascade_size * 2:  # Need at least 2 potential cascades
            logger.warning(f"Insufficient fossils for cascade analysis: {len(fossils)}")
            return {
                'status': 'insufficient_data',
                'fossils_count': len(fossils),
                'min_required': min_cascade_size * 2
            }
        
        # Convert to DataFrame
        df = self._fossils_to_dataframe(fossils)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Identify cascades
        cascades = self._identify_cascades(df, max_time_gap, min_cascade_size)
        
        if len(cascades) == 0:
            logger.info("No verification cascades identified")
            return {
                'status': 'no_cascades_found',
                'fossils_analyzed': len(fossils),
                'cascade_parameters': {
                    'max_time_gap_hours': max_time_gap.total_seconds() / 3600,
                    'min_cascade_size': min_cascade_size
                }
            }
        
        # Analyze cascade characteristics
        cascade_stats = self._analyze_cascade_statistics(cascades)
        cascade_triggers = self._analyze_cascade_triggers(cascades, df)
        cascade_propagation = self._analyze_cascade_propagation(cascades, df)
        
        # Generate cascade visualization data
        cascade_visualizations = self._generate_cascade_visualizations(cascades, df)
        
        # Combine results
        return {
            'status': 'success',
            'fossils_analyzed': len(fossils),
            'cascades_identified': len(cascades),
            'cascade_parameters': {
                'max_time_gap_hours': max_time_gap.total_seconds() / 3600,
                'min_cascade_size': min_cascade_size
            },
            'cascade_statistics': cascade_stats,
            'cascade_triggers': cascade_triggers,
            'cascade_propagation': cascade_propagation,
            'cascade_visualizations': cascade_visualizations
        }
    
    def analyze_cross_method_temporal_relationships(self,
                                                  content_type: Optional[str] = None,
                                                  min_methods: int = 2,
                                                  min_fossils_per_method: int = 10) -> Dict[str, Any]:
        """
        Analyze temporal relationships between different verification methods.
        
        This method examines how different verification methods interact over time,
        identifying leads, lags, and other temporal relationships.
        
        Args:
            content_type: Optional content type to focus analysis on
            min_methods: Minimum number of distinct verification methods required
            min_fossils_per_method: Minimum fossils per method
            
        Returns:
            Dictionary containing cross-method temporal relationship analysis
        """
        # Retrieve fossils for analysis
        query_kwargs = {}
        if content_type:
            query_kwargs['content_type'] = content_type
        
        # Set a reasonable limit to avoid memory issues
        query_kwargs['limit'] = 10000
        
        fossils = self.fossil_db.query_fossils(**query_kwargs)
        
        # Convert to DataFrame
        df = self._fossils_to_dataframe(fossils)
        
        # Check if we have enough methods with enough data
        method_counts = df['verification_method'].value_counts()
        valid_methods = method_counts[method_counts >= min_fossils_per_method].index.tolist()
        
        if len(valid_methods) < min_methods:
            logger.warning(f"Insufficient methods with enough fossils: {len(valid_methods)} < {min_methods}")
            return {
                'status': 'insufficient_data',
                'methods_with_sufficient_fossils': len(valid_methods),
                'min_required': min_methods
            }
        
        # Filter to valid methods
        df = df[df['verification_method'].isin(valid_methods)]
        
        # Analyze lead/lag relationships
        lead_lag_analysis = self._analyze_method_lead_lag(df)
        
        # Analyze method succession patterns
        succession_analysis = self._analyze_method_succession(df)
        
        # Analyze method correlation over time
        correlation_analysis = self._analyze_method_correlation(df)
        
        # Combine results
        return {
            'status': 'success',
            'methods_analyzed': valid_methods,
            'fossils_analyzed': len(df),
            'lead_lag_analysis': lead_lag_analysis,
            'succession_analysis': succession_analysis,
            'correlation_analysis': correlation_analysis
        }
    
    def _fossils_to_dataframe(self, fossils: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert a list of fossil dictionaries to a pandas DataFrame.
        
        Args:
            fossils: List of fossil dictionaries
            
        Returns:
            DataFrame with normalized fossil data
        """
        if not fossils:
            return pd.DataFrame()
        
        # Extract basic fields for all fossils
        data = []
        for fossil in fossils:
            row = {
                'fossil_id': fossil['fossil_id'],
                'content_fingerprint': fossil['content_fingerprint'],
                'verification_score': fossil['verification_score'],
                'timestamp': fossil['timestamp'],
                'content_type': fossil['content_type'],
                'verification_method': fossil['verification_method'],
                'verifier_id': fossil['verifier_id'],
                'created_at': fossil['created_at']
            }
            
            # Extract metadata fields if available
            if 'metadata' in fossil and fossil['metadata']:
                try:
                    if isinstance(fossil['metadata'], str):
                        metadata = json.loads(fossil['metadata'])
                    else:
                        metadata = fossil['metadata']
                    
                    for key, value in metadata.items():
                        # Avoid duplicating existing keys
                        if key not in row:
                            row[f'metadata_{key}'] = value
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse metadata for fossil {fossil['fossil_id']}")
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        return df
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal trends in verification data.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with trend analysis
        """
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Convert to daily time series
        daily_scores = df.set_index('timestamp')['verification_score'].resample('D').mean()
        
        # Remove NaN values (days without data)
        daily_scores = daily_scores.dropna()
        
        # If not enough data points, return empty analysis
        if len(daily_scores) < 7:  # At least a week of data
            return {
                'has_trend': False,
                'reason': 'insufficient_data_points'
            }
        
        # Perform linear regression
        x = np.array(range(len(daily_scores)))
        y = daily_scores.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Check if the trend is significant
        trend_significant = p_value < 0.05 and abs(r_value) > 0.3
        
        # Generate data for trend visualization
        trend_line = intercept + slope * x
        
        # Calculate trend magnitude (percentage change over the period)
        if len(trend_line) > 1 and trend_line[0] != 0:
            trend_magnitude = (trend_line[-1] - trend_line[0]) / trend_line[0] * 100
        else:
            trend_magnitude = 0
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        return {
            'has_trend': trend_significant,
            '
