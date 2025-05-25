"""
Collective Trust Memory: Trust Archaeologist

This module implements the Trust Archaeologist for the Collective Trust Memory component
of ConfidenceID. It provides pattern recognition and analysis capabilities for the
verification fossil record, enabling the identification of temporal patterns,
anomalies, and trends in verification history.

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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats

from .fossil_record_db import FossilRecordDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrustArchaeologist:
    """
    Analyzes the verification fossil record to identify patterns, anomalies, and trends.
    
    The Trust Archaeologist excavates patterns from the collective memory of verification
    events, enabling insights that span across time, context, and verification methods.
    It helps ConfidenceID learn from past verification experiences and adapt to emerging
    patterns of both authentic and manipulated content.
    """
    
    def __init__(self, fossil_db: FossilRecordDB):
        """
        Initialize the Trust Archaeologist.
        
        Args:
            fossil_db: The Fossil Record Database to analyze
        """
        self.fossil_db = fossil_db
    
    def excavate_trust_patterns(self, 
                               content_fingerprint: Optional[str] = None,
                               temporal_range: Optional[Tuple[datetime, datetime]] = None,
                               context_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze historical verification patterns for similar content.
        
        This method excavates patterns from the verification fossil record, identifying
        temporal patterns, anomalies, and consistencies in verification history.
        
        Args:
            content_fingerprint: Optional content fingerprint to focus analysis on
            temporal_range: Optional time range (start_time, end_time) for analysis
            context_filter: Optional filters for context (e.g., content_type, method)
            
        Returns:
            An archaeology report containing identified patterns
        """
        # Retrieve relevant fossils from the database
        fossils = self._retrieve_fossils(content_fingerprint, temporal_range, context_filter)
        
        if not fossils:
            logger.warning("No fossils found for the specified criteria")
            return self._empty_archaeology_report()
        
        # Convert fossils to a DataFrame for easier analysis
        fossils_df = self._fossils_to_dataframe(fossils)
        
        # Analyze temporal patterns
        temporal_patterns = self.analyze_temporal_evolution(fossils_df)
        
        # Detect verification anomalies
        pattern_anomalies = self.detect_verification_anomalies(fossils_df)
        
        # Detect verification consistencies
        pattern_consistencies = self.detect_verification_consistencies(fossils_df)
        
        # Trace confidence evolution
        confidence_evolution = self.trace_confidence_evolution(fossils_df)
        
        # Generate archaeology report
        report = {
            'report_id': f"AR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'criteria': {
                'content_fingerprint': content_fingerprint,
                'temporal_range': [t.isoformat() if t else None for t in temporal_range] if temporal_range else None,
                'context_filter': context_filter
            },
            'fossils_analyzed': len(fossils),
            'temporal_patterns': temporal_patterns,
            'anomalies': pattern_anomalies,
            'consistencies': pattern_consistencies,
            'confidence_evolution': confidence_evolution
        }
        
        # Store the report in the database if it contains significant patterns
        if self._report_has_significant_patterns(report):
            pattern_id = self._store_report_as_pattern(report)
            report['pattern_id'] = pattern_id
        
        return report
    
    def analyze_temporal_evolution(self, fossils_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze the temporal evolution of verification patterns.
        
        Args:
            fossils_df: DataFrame containing verification fossils
            
        Returns:
            List of identified temporal patterns
        """
        if fossils_df.empty:
            return []
        
        temporal_patterns = []
        
        # Ensure datetime format for timestamp
        fossils_df['timestamp'] = pd.to_datetime(fossils_df['timestamp'])
        
        # Sort by timestamp
        fossils_df = fossils_df.sort_values('timestamp')
        
        # Analyze by content type
        content_types = fossils_df['content_type'].unique()
        for content_type in content_types:
            type_df = fossils_df[fossils_df['content_type'] == content_type]
            
            # Skip if not enough data
            if len(type_df) < 3:
                continue
            
            # Analyze verification score trends
            trend_analysis = self._analyze_score_trend(type_df)
            if trend_analysis:
                temporal_patterns.append({
                    'pattern_type': 'temporal_trend',
                    'content_type': content_type,
                    'description': trend_analysis['description'],
                    'confidence': trend_analysis['confidence'],
                    'data': trend_analysis['data']
                })
            
            # Analyze verification frequency patterns
            frequency_analysis = self._analyze_verification_frequency(type_df)
            if frequency_analysis:
                temporal_patterns.append({
                    'pattern_type': 'verification_frequency',
                    'content_type': content_type,
                    'description': frequency_analysis['description'],
                    'confidence': frequency_analysis['confidence'],
                    'data': frequency_analysis['data']
                })
            
            # Analyze seasonal or cyclical patterns
            cyclical_analysis = self._analyze_cyclical_patterns(type_df)
            if cyclical_analysis:
                temporal_patterns.append({
                    'pattern_type': 'cyclical_pattern',
                    'content_type': content_type,
                    'description': cyclical_analysis['description'],
                    'confidence': cyclical_analysis['confidence'],
                    'data': cyclical_analysis['data']
                })
        
        # Analyze cross-type temporal relationships
        if len(content_types) > 1:
            cross_type_analysis = self._analyze_cross_type_relationships(fossils_df)
            if cross_type_analysis:
                temporal_patterns.append({
                    'pattern_type': 'cross_type_relationship',
                    'description': cross_type_analysis['description'],
                    'confidence': cross_type_analysis['confidence'],
                    'data': cross_type_analysis['data']
                })
        
        return temporal_patterns
    
    def detect_verification_anomalies(self, fossils_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the verification history.
        
        Args:
            fossils_df: DataFrame containing verification fossils
            
        Returns:
            List of identified anomalies
        """
        if fossils_df.empty:
            return []
        
        anomalies = []
        
        # Ensure datetime format for timestamp
        fossils_df['timestamp'] = pd.to_datetime(fossils_df['timestamp'])
        
        # Detect verification score outliers
        score_outliers = self._detect_score_outliers(fossils_df)
        anomalies.extend(score_outliers)
        
        # Detect suspicious verification timing
        timing_anomalies = self._detect_timing_anomalies(fossils_df)
        anomalies.extend(timing_anomalies)
        
        # Detect verifier inconsistencies
        verifier_anomalies = self._detect_verifier_inconsistencies(fossils_df)
        anomalies.extend(verifier_anomalies)
        
        # Detect cross-method inconsistencies
        method_anomalies = self._detect_method_inconsistencies(fossils_df)
        anomalies.extend(method_anomalies)
        
        return anomalies
    
    def detect_verification_consistencies(self, fossils_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect consistencies in the verification history.
        
        Args:
            fossils_df: DataFrame containing verification fossils
            
        Returns:
            List of identified consistencies
        """
        if fossils_df.empty:
            return []
        
        consistencies = []
        
        # Detect consistent verification scores
        score_consistencies = self._detect_score_consistencies(fossils_df)
        consistencies.extend(score_consistencies)
        
        # Detect consistent verification methods
        method_consistencies = self._detect_method_consistencies(fossils_df)
        consistencies.extend(method_consistencies)
        
        # Detect consistent verifier patterns
        verifier_consistencies = self._detect_verifier_consistencies(fossils_df)
        consistencies.extend(verifier_consistencies)
        
        return consistencies
    
    def trace_confidence_evolution(self, fossils_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trace the evolution of confidence over time.
        
        Args:
            fossils_df: DataFrame containing verification fossils
            
        Returns:
            Dictionary containing confidence evolution data
        """
        if fossils_df.empty:
            return {'evolution_type': 'empty', 'data': {}}
        
        # Ensure datetime format for timestamp
        fossils_df['timestamp'] = pd.to_datetime(fossils_df['timestamp'])
        
        # Sort by timestamp
        fossils_df = fossils_df.sort_values('timestamp')
        
        # Calculate running statistics
        fossils_df['cumulative_mean'] = fossils_df['verification_score'].expanding().mean()
        fossils_df['cumulative_std'] = fossils_df['verification_score'].expanding().std()
        fossils_df['cumulative_min'] = fossils_df['verification_score'].expanding().min()
        fossils_df['cumulative_max'] = fossils_df['verification_score'].expanding().max()
        
        # Determine evolution type
        evolution_type = self._determine_evolution_type(fossils_df)
        
        # Generate evolution data
        evolution_data = {
            'timestamps': [ts.isoformat() for ts in fossils_df['timestamp']],
            'scores': fossils_df['verification_score'].tolist(),
            'cumulative_mean': fossils_df['cumulative_mean'].tolist(),
            'cumulative_std': fossils_df['cumulative_std'].tolist(),
            'cumulative_min': fossils_df['cumulative_min'].tolist(),
            'cumulative_max': fossils_df['cumulative_max'].tolist()
        }
        
        return {
            'evolution_type': evolution_type,
            'data': evolution_data
        }
    
    def generate_archaeology_report(self, 
                                  content_fingerprint: str,
                                  include_visualizations: bool = False) -> Dict[str, Any]:
        """
        Generate a comprehensive archaeology report for a specific content fingerprint.
        
        This report includes all temporal patterns, anomalies, and consistencies
        identified in the verification history of the content.
        
        Args:
            content_fingerprint: The content fingerprint to analyze
            include_visualizations: Whether to include visualization data in the report
            
        Returns:
            A comprehensive archaeology report
        """
        # Get all fossils for the content fingerprint
        fossils = self.fossil_db.get_fossils_by_fingerprint(content_fingerprint)
        
        if not fossils:
            logger.warning(f"No fossils found for fingerprint: {content_fingerprint}")
            return self._empty_archaeology_report()
        
        # Convert to DataFrame
        fossils_df = self._fossils_to_dataframe(fossils)
        
        # Generate basic report
        report = {
            'report_id': f"AR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'content_fingerprint': content_fingerprint,
            'fossils_analyzed': len(fossils),
            'first_verification': min(fossils_df['timestamp']).isoformat(),
            'last_verification': max(fossils_df['timestamp']).isoformat(),
            'verification_methods': fossils_df['verification_method'].unique().tolist(),
            'content_types': fossils_df['content_type'].unique().tolist(),
            'verifier_count': fossils_df['verifier_id'].nunique(),
            'average_verification_score': fossils_df['verification_score'].mean(),
            'score_std_dev': fossils_df['verification_score'].std(),
            'temporal_patterns': self.analyze_temporal_evolution(fossils_df),
            'anomalies': self.detect_verification_anomalies(fossils_df),
            'consistencies': self.detect_verification_consistencies(fossils_df),
            'confidence_evolution': self.trace_confidence_evolution(fossils_df)
        }
        
        # Add visualizations if requested
        if include_visualizations:
            report['visualizations'] = self._generate_visualizations(fossils_df)
        
        return report
    
    def analyze_cross_content_patterns(self, 
                                     content_type: Optional[str] = None,
                                     limit: int = 1000) -> Dict[str, Any]:
        """
        Analyze patterns across different content fingerprints.
        
        This method identifies patterns that span across multiple content items,
        such as common verification trends, similar anomalies, or consistent behavior.
        
        Args:
            content_type: Optional content type to focus analysis on
            limit: Maximum number of fossils to analyze
            
        Returns:
            Dictionary containing cross-content pattern analysis
        """
        # Retrieve fossils for analysis
        fossils = []
        if content_type:
            fossils = self.fossil_db.query_fossils(content_type=content_type, limit=limit)
        else:
            fossils = self.fossil_db.query_fossils(limit=limit)
        
        if not fossils:
            logger.warning("No fossils found for analysis")
            return {'patterns': [], 'confidence': 0.0}
        
        # Convert to DataFrame
        fossils_df = self._fossils_to_dataframe(fossils)
        
        # Identify clusters of similar content based on verification patterns
        clusters = self._cluster_content_by_verification(fossils_df)
        
        # Analyze patterns within and across clusters
        patterns = self._analyze_cluster_patterns(clusters, fossils_df)
        
        return {
            'clusters': len(clusters),
            'patterns': patterns,
            'confidence': self._calculate_pattern_confidence(patterns),
            'fossils_analyzed': len(fossils)
        }
    
    def _retrieve_fossils(self, 
                        content_fingerprint: Optional[str] = None,
                        temporal_range: Optional[Tuple[datetime, datetime]] = None,
                        context_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve fossils from the database based on criteria.
        
        Args:
            content_fingerprint: Optional content fingerprint to focus analysis on
            temporal_range: Optional time range (start_time, end_time) for analysis
            context_filter: Optional filters for context (e.g., content_type, method)
            
        Returns:
            List of fossils matching the criteria
        """
        query_kwargs = {}
        
        if content_fingerprint:
            query_kwargs['content_fingerprint'] = content_fingerprint
        
        if temporal_range:
            query_kwargs['start_time'] = temporal_range[0]
            query_kwargs['end_time'] = temporal_range[1]
        
        if context_filter:
            for key, value in context_filter.items():
                if key in ['content_type', 'verification_method', 'verifier_id']:
                    query_kwargs[key] = value
                elif key == 'min_score' and isinstance(value, (int, float)):
                    query_kwargs['min_score'] = value
                elif key == 'max_score' and isinstance(value, (int, float)):
                    query_kwargs['max_score'] = value
        
        # Set a reasonable limit to avoid memory issues
        query_kwargs['limit'] = context_filter.get('limit', 10000) if context_filter else 10000
        
        return self.fossil_db.query_fossils(**query_kwargs)
    
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
    
    def _analyze_score_trend(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze the trend in verification scores over time.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with trend analysis or None if no significant trend
        """
        if len(df) < 3:
            return None
        
        # Perform linear regression on scores over time
        x = np.array((df['timestamp'] - df['timestamp'].min()).dt.total_seconds())
        y = df['verification_score'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Check if the trend is significant
        if p_value < 0.05 and abs(r_value) > 0.3:
            trend_direction = "increasing" if slope > 0 else "decreasing"
            confidence = min(1.0, abs(r_value) * (1 - p_value))
            
            return {
                'description': f"Verification scores show a {trend_direction} trend over time (r={r_value:.2f}, p={p_value:.4f})",
                'confidence': confidence,
                'data': {
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value,
                    'p_value': p_value,
                    'std_err': std_err
                }
            }
        
        return None
    
    def _analyze_verification_frequency(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze the frequency of verification events over time.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with frequency analysis or None if no significant pattern
        """
        if len(df) < 5:
            return None
        
        # Resample by day to get verification counts
        daily_counts = df.resample('D', on='timestamp').size()
        
        # If not enough days with data, return None
        if len(daily_counts) < 3:
            return None
        
        # Check for patterns in verification frequency
        mean_verifications = daily_counts.mean()
        std_verifications = daily_counts.std()
        
        # Calculate autocorrelation to detect patterns
        autocorr = pd.Series(daily_counts).autocorr(lag=1)
        
        if abs(autocorr) > 0.3:
            pattern_type = "consistent" if autocorr > 0 else "alternating"
            confidence = min(1.0, abs(autocorr))
            
            return {
                'description': f"Verification frequency shows a {pattern_type} pattern (autocorr={autocorr:.2f})",
                'confidence': confidence,
                'data': {
                    'mean_daily_verifications': mean_verifications,
                    'std_daily_verifications': std_verifications,
                    'autocorrelation': autocorr,
                    'days_with_data': len(daily_counts)
                }
            }
        
        return None
    
    def _analyze_cyclical_patterns(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze cyclical or seasonal patterns in verification data.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with cyclical pattern analysis or None if no significant pattern
        """
        if len(df) < 10:
            return None
        
        # Extract datetime components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        patterns = []
        
        # Check for hourly patterns
        hourly_scores = df.groupby('hour')['verification_score'].mean()
        if hourly_scores.var() > 0.01:
            hour_max = hourly_scores.idxmax()
            hour_min = hourly_scores.idxmin()
            patterns.append(f"Scores tend to be higher at {hour_max}:00 and lower at {hour_min}:00")
        
        # Check for day-of-week patterns
        dow_scores = df.groupby('day_of_week')['verification_score'].mean()
        if dow_scores.var() > 0.01:
            dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_max = dow_names[dow_scores.idxmax()]
            dow_min = dow_names[dow_scores.idxmin()]
            patterns.append(f"Scores tend to be higher on {dow_max} and lower on {dow_min}")
        
        # Check for monthly patterns
        if df['timestamp'].max() - df['timestamp'].min() > timedelta(days=60):
            month_scores = df.groupby('month')['verification_score'].mean()
            if len(month_scores) > 2 and month_scores.var() > 0.01:
                month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                             'July', 'August', 'September', 'October', 'November', 'December']
                month_max = month_names[month_scores.idxmax() - 1]
                month_min = month_names[month_scores.idxmin() - 1]
                patterns.append(f"Scores tend to be higher in {month_max} and lower in {month_min}")
        
        if patterns:
            return {
                'description': ". ".join(patterns),
                'confidence': min(1.0, 0.5 + len(patterns) * 0.1),
                'data': {
                    'hourly_variance': hourly_scores.var(),
                    'day_of_week_variance': dow_scores.var(),
                    'month_variance': month_scores.var() if 'month_scores' in locals() else None
                }
            }
        
        return None
    
    def _analyze_cross_type_relationships(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze relationships between different content types in verification patterns.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with cross-type analysis or None if no significant relationship
        """
        content_types = df['content_type'].unique()
        if len(content_types) <= 1:
            return None
        
        relationships = []
        
        # Group by content type and date
        df['date'] = df['timestamp'].dt.date
        type_date_scores = df.groupby(['content_type', 'date'])['verification_score'].mean().reset_index()
        
        # Convert to wide format for correlation analysis
        pivot_df = type_date_scores.pivot(index='date', columns='content_type', values='verification_score')
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr()
        
        # Find significant correlations
        for i, type1 in enumerate(content_types):
            for j, type2 in enumerate(content_types):
                if i < j and type1 in corr_matrix.index and type2 in corr_matrix.columns:
                    corr = corr_matrix.loc[type1, type2]
                    if abs(corr) > 0.5:
                        relationship = "positively" if corr > 0 else "negatively"
                        relationships.append(f"{type1} and {type2} verification scores are {relationship} correlated (r={corr:.2f})")
        
        if relationships:
            return {
                'description': ". ".join(relationships),
                'confidence': min(1.0, 0.5 + len(relationships) * 0.1),
                'data': {
                    'correlation_matrix': corr_matrix.to_dict()
                }
            }
        
        return None
    
    def _detect_score_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect outliers in verification scores.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            List of score outlier anomalies
        """
        if len(df) < 5:
            return []
        
        outliers = []
        
        # Calculate z-scores for verification scores
        df['z_score'] = stats.zscore(df['verification_score'])
        
        # Identify outliers (|z| > 2)
        outlier_rows = df[abs(df['z_score']) > 2]
        
        if len(outlier_rows) > 0:
            for _, row in outlier_rows.iterrows():
                direction = "high" if row['z_score'] > 0 else "low"
                outliers.append({
                    'anomaly_type': 'score_outlier',
                    'description': f"Unusually {direction} verification score ({row['verification_score']:.2f}) detected",
                    'confidence': min(1.0, abs(row['z_score']) / 4),  # Scale confidence based on z-score
                    'fossil_id': row['fossil_id'],
                    'timestamp': row['timestamp'].isoformat(),
                    'verification_method': row['verification_method'],
                    'z_score': row['z_score']
                })
        
        return outliers
    
    def _detect_timing_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies in verification timing.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            List of timing anomalies
        """
        if len(df) < 5:
            return []
        
        anomalies = []
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate time differences between consecutive verifications
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        
        # Identify sudden bursts of verification activity
        df['short_interval'] = df['time_diff'] < df['time_diff'].median() / 5
        burst_starts = df[df['short_interval'] & ~df['short_interval'].shift(1, fill_value=False)]
        
        for _, row in burst_starts.iterrows():
            # Look for sequences of short intervals
            idx = df.index.get_loc(row.name)
            if idx + 2 < len(df) and df.iloc[idx+1]['short_interval'] and df.iloc[idx+2]['short_interval']:
                anomalies.append({
                    'anomaly_type': 'verification_burst',
                    'description': f"Unusual burst of verification activity detected",
                    'confidence': 0.7,
                    'fossil_id': row['fossil_id'],
                    'timestamp': row['timestamp'].isoformat(),
                    'verification_method': row['verification_method']
                })
        
        # Identify unusually long gaps in verification
        df['long_interval'] = df['time_diff'] > df['time_diff'].median() * 5
        long_gaps = df[df['long_interval']]
        
        for _, row in long_gaps.iterrows():
            anomalies.append({
                'anomaly_type': 'verification_gap',
                'description': f"Unusually long gap ({row['time_diff']/3600:.1f} hours) before verification",
                'confidence': min(1.0, row['time_diff'] / (df['time_diff'].median() * 10)),
                'fossil_id': row['fossil_id'],
                'timestamp': row['timestamp'].isoformat(),
                'verification_method': row['verification_method'],
                'gap_seconds': row['time_diff']
            })
        
        return anomalies
    
    def _detect_verifier_inconsistencies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect inconsistencies between different verifiers.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            List of verifier inconsistency anomalies
        """
        if len(df) < 5 or df['verifier_id'].nunique() <= 1:
            return []
        
        anomalies = []
        
        # Group by verifier and calculate mean and std of scores
        verifier_stats = df.groupby('verifier_id')['verification_score'].agg(['mean', 'std']).reset_index()
        
        # Calculate the mean difference between verifiers
        verifier_means = verifier_stats['mean'].values
        mean_diffs = []
        for i in range(len(verifier_means)):
            for j in range(i+1, len(verifier_means)):
                mean_diffs.append(abs(verifier_means[i] - verifier_means[j]))
        
        # If we have significant differences between verifiers
        if mean_diffs and max(mean_diffs) > 0.2:
            # Identify the most divergent verifiers
            large_diffs = []
            for i, row1 in verifier_stats.iterrows():
                for j, row2 in verifier_stats.iterrows():
                    if i < j:
                        diff = abs(row1['mean'] - row2['mean'])
                        if diff > 0.2:
                            large_diffs.append((row1['verifier_id'], row2['verifier_id'], diff))
            
            # Sort by difference magnitude
            large_diffs.sort(key=lambda x: x[2], reverse=True)
            
            # Report the largest differences
            for v1, v2, diff in large_diffs[:3]:  # Limit to top 3 to avoid overwhelming
                anomalies.append({
                    'anomaly_type': 'verifier_inconsistency',
                    'description': f"Significant difference in average scores between verifiers {v1} and {v2} (diff={diff:.2f})",
                    'confidence': min(1.0, diff * 2),  # Scale confidence based on difference
                    'verifier1': v1,
                    'verifier2': v2,
                    'difference': diff
                })
        
        return anomalies
    
    def _detect_method_inconsistencies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect inconsistencies between different verification methods.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            List of method inconsistency anomalies
        """
        if len(df) < 5 or df['verification_method'].nunique() <= 1:
            return []
        
        anomalies = []
        
        # Group by method and calculate mean and std of scores
        method_stats = df.groupby('verification_method')['verification_score'].agg(['mean', 'std', 'count']).reset_index()
        
        # Only consider methods with enough data
        method_stats = method_stats[method_stats['count'] >= 3]
        
        if len(method_stats) <= 1:
            return []
        
        # Calculate the mean difference between methods
        method_means = method_stats['mean'].values
        mean_diffs = []
        for i in range(len(method_means)):
            for j in range(i+1, len(method_means)):
                mean_diffs.append(abs(method_means[i] - method_means[j]))
        
        # If we have significant differences between methods
        if mean_diffs and max(mean_diffs) > 0.2:
            # Identify the most divergent methods
            large_diffs = []
            for i, row1 in method_stats.iterrows():
                for j, row2 in method_stats.iterrows():
                    if i < j:
                        diff = abs(row1['mean'] - row2['mean'])
                        if diff > 0.2:
                            large_diffs.append((row1['verification_method'], row2['verification_method'], diff))
            
            # Sort by difference magnitude
            large_diffs.sort(key=lambda x: x[2], reverse=True)
            
            # Report the largest differences
            for m1, m2, diff in large_diffs[:3]:  # Limit to top 3
                anomalies.append({
                    'anomaly_type': 'method_inconsistency',
                    'description': f"Significant difference in average scores between methods {m1} and {m2} (diff={diff:.2f})",
                    'confidence': min(1.0, diff * 2),  # Scale confidence based on difference
                    'method1': m1,
                    'method2': m2,
                    'difference': diff
                })
        
        return anomalies
    
    def _detect_score_consistencies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect consistent patterns in verification scores.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            List of score consistency patterns
        """
        if len(df) < 5:
            return []
        
        consistencies = []
        
        # Calculate overall score statistics
        mean_score = df['verification_score'].mean()
        std_score = df['verification_score'].std()
        
        # Check for consistently high scores
        if mean_score > 0.8 and std_score < 0.1:
            consistencies.append({
                'consistency_type': 'high_scores',
                'description': f"Consistently high verification scores (mean={mean_score:.2f}, std={std_score:.2f})",
                'confidence': min(1.0, mean_score + (1 - std_score)),
                'mean_score': mean_score,
                'std_score': std_score
            })
        
        # Check for consistently low scores
        elif mean_score < 0.2 and std_score < 0.1:
            consistencies.append({
                'consistency_type': 'low_scores',
                'description': f"Consistently low verification scores (mean={mean_score:.2f}, std={std_score:.2f})",
                'confidence': min(1.0, (1 - mean_score) + (1 - std_score)),
                'mean_score': mean_score,
                'std_score': std_score
            })
        
        # Check for very stable scores (low variance)
        elif std_score < 0.05 and len(df) >= 10:
            consistencies.append({
                'consistency_type': 'stable_scores',
                'description': f"Unusually stable verification scores (std={std_score:.2f})",
                'confidence': min(1.0, 1 - std_score * 10),
                'mean_score': mean_score,
                'std_score': std_score
            })
        
        return consistencies
    
    def _detect_method_consistencies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect consistent patterns in verification methods.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            List of method consistency patterns
        """
        if len(df) < 5:
            return []
        
        consistencies = []
        
        # Group by method and calculate statistics
        method_stats = df.groupby('verification_method').agg({
            'verification_score': ['mean', 'std', 'count'],
            'fossil_id': 'count'
        }).reset_index()
        
        # Rename columns
        method_stats.columns = ['verification_method', 'mean_score', 'std_score', 'count', 'fossil_count']
        
        # Find methods with consistent results
        consistent_methods = method_stats[(method_stats['std_score'] < 0.1) & (method_stats['count'] >= 5)]
        
        for _, row in consistent_methods.iterrows():
            consistency_type = 'consistent_high' if row['mean_score'] > 0.8 else (
                'consistent_low' if row['mean_score'] < 0.2 else 'consistent_mid'
            )
            
            consistencies.append({
                'consistency_type': f"{consistency_type}_method",
                'description': f"Method '{row['verification_method']}' consistently produces {consistency_type.split('_')[1]} scores (mean={row['mean_score']:.2f}, std={row['std_score']:.2f})",
                'confidence': min(1.0, (1 - row['std_score'] * 5) * min(1.0, row['count'] / 10)),
                'verification_method': row['verification_method'],
                'mean_score': row['mean_score'],
                'std_score': row['std_score'],
                'count': row['count']
            })
        
        # Check if a single method dominates
        total_fossils = df['fossil_id'].nunique()
        for _, row in method_stats.iterrows():
            method_ratio = row['fossil_count'] / total_fossils
            if method_ratio > 0.8 and total_fossils >= 10:
                consistencies.append({
                    'consistency_type': 'dominant_method',
                    'description': f"Method '{row['verification_method']}' dominates verification history ({method_ratio:.1%} of all verifications)",
                    'confidence': min(1.0, method_ratio),
                    'verification_method': row['verification_method'],
                    'ratio': method_ratio,
                    'count': row['fossil_count']
                })
        
        return consistencies
    
    def _detect_verifier_consistencies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect consistent patterns in verifiers.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            List of verifier consistency patterns
        """
        if len(df) < 5 or df['verifier_id'].nunique() <= 1:
            return []
        
        consistencies = []
        
        # Group by verifier and calculate statistics
        verifier_stats = df.groupby('verifier_id').agg({
            'verification_score': ['mean', 'std', 'count'],
            'fossil_id': 'count'
        }).reset_index()
        
        # Rename columns
        verifier_stats.columns = ['verifier_id', 'mean_score', 'std_score', 'count', 'fossil_count']
        
        # Find verifiers with consistent results
        consistent_verifiers = verifier_stats[(verifier_stats['std_score'] < 0.1) & (verifier_stats['count'] >= 5)]
        
        for _, row in consistent_verifiers.iterrows():
            consistency_type = 'consistent_high' if row['mean_score'] > 0.8 else (
                'consistent_low' if row['mean_score'] < 0.2 else 'consistent_mid'
            )
            
            consistencies.append({
                'consistency_type': f"{consistency_type}_verifier",
                'description': f"Verifier '{row['verifier_id']}' consistently produces {consistency_type.split('_')[1]} scores (mean={row['mean_score']:.2f}, std={row['std_score']:.2f})",
                'confidence': min(1.0, (1 - row['std_score'] * 5) * min(1.0, row['count'] / 10)),
                'verifier_id': row['verifier_id'],
                'mean_score': row['mean_score'],
                'std_score': row['std_score'],
                'count': row['count']
            })
        
        # Check if a single verifier dominates
        total_fossils = df['fossil_id'].nunique()
        for _, row in verifier_stats.iterrows():
            verifier_ratio = row['fossil_count'] / total_fossils
            if verifier_ratio > 0.8 and total_fossils >= 10:
                consistencies.append({
                    'consistency_type': 'dominant_verifier',
                    'description': f"Verifier '{row['verifier_id']}' dominates verification history ({verifier_ratio:.1%} of all verifications)",
                    'confidence': min(1.0, verifier_ratio),
                    'verifier_id': row['verifier_id'],
                    'ratio': verifier_ratio,
                    'count': row['fossil_count']
                })
        
        return consistencies
    
    def _determine_evolution_type(self, df: pd.DataFrame) -> str:
        """
        Determine the type of confidence evolution over time.
        
        Args:
            df: DataFrame containing verification data with cumulative statistics
            
        Returns:
            A string describing the evolution type
        """
        if len(df) < 5:
            return "insufficient_data"
        
        # Calculate linear regression on scores over time
        x = np.array((df['timestamp'] - df['timestamp'].min()).dt.total_seconds())
        y = df['verification_score'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Check for significant trends
        if p_value < 0.05 and abs(r_value) > 0.3:
            if slope > 0:
                return "improving_confidence"
            else:
                return "declining_confidence"
        
        # Check for stability
        if df['verification_score'].std() < 0.1:
            return "stable_confidence"
        
        # Check for volatility
        volatility = np.diff(df['verification_score'].values)
        if np.mean(np.abs(volatility)) > 0.2:
            return "volatile_confidence"
        
        # Check for oscillations
        if len(df) >= 10:
            # Calculate autocorrelation with lag 2
            autocorr = pd.Series(df['verification_score']).autocorr(lag=2)
            if autocorr < -0.3:
                return "oscillating_confidence"
        
        # Default
        return "mixed_confidence"
    
    def _cluster_content_by_verification(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify clusters of content based on verification patterns.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            List of cluster descriptions
        """
        if len(df) < 10 or df['content_fingerprint'].nunique() < 3:
            return []
        
        # Prepare data for clustering
        # Aggregate by content fingerprint
        content_stats = df.groupby('content_fingerprint').agg({
            'verification_score': ['mean', 'std', 'count'],
            'verification_method': lambda x: x.nunique(),
            'verifier_id': lambda x: x.nunique(),
            'content_type': lambda x: x.mode().iloc[0] if not x.empty else None
        }).reset_index()
        
        # Rename columns
        content_stats.columns = [
            'content_fingerprint', 'mean_score', 'std_score', 'verification_count',
            'method_count', 'verifier_count', 'content_type'
        ]
        
        # Only proceed if we have enough content items
        if len(content_stats) < 3:
            return []
        
        # Features for clustering
        features = content_stats[['mean_score', 'std_score', 'verification_count', 'method_count', 'verifier_count']]
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=1.0, min_samples=2)
        content_stats['cluster'] = dbscan.fit_predict(features_scaled)
        
        # Process clusters
        clusters = []
        for cluster_id in sorted(content_stats['cluster'].unique()):
            if cluster_id == -1:
                # Noise points
                continue
            
            cluster_items = content_stats[content_stats['cluster'] == cluster_id]
            
            # Calculate cluster statistics
            cluster_info = {
                'cluster_id': f"C{cluster_id}",
                'content_count': len(cluster_items),
                'mean_verification_score': cluster_items['mean_score'].mean(),
                'content_types': cluster_items['content_type'].unique().tolist(),
                'content_fingerprints': cluster_items['content_fingerprint'].tolist(),
                'description': self._generate_cluster_description(cluster_items)
            }
            
            clusters.append(cluster_info)
        
        return clusters
    
    def _analyze_cluster_patterns(self, clusters: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze patterns within and across identified clusters.
        
        Args:
            clusters: List of cluster descriptions
            df: Original DataFrame containing verification data
            
        Returns:
            List of identified patterns
        """
        if not clusters:
            return []
        
        patterns = []
        
        # Analyze each cluster
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            fingerprints = cluster['content_fingerprints']
            
            # Get all fossils for this cluster
            cluster_fossils = df[df['content_fingerprint'].isin(fingerprints)]
            
            # Time-based patterns
            time_pattern = self._analyze_cluster_time_pattern(cluster_fossils)
            if time_pattern:
                time_pattern['cluster_id'] = cluster_id
                patterns.append(time_pattern)
            
            # Method effectiveness patterns
            method_pattern = self._analyze_cluster_method_pattern(cluster_fossils)
            if method_pattern:
                method_pattern['cluster_id'] = cluster_id
                patterns.append(method_pattern)
            
            # Verifier bias patterns
            verifier_pattern = self._analyze_cluster_verifier_pattern(cluster_fossils)
            if verifier_pattern:
                verifier_pattern['cluster_id'] = cluster_id
                patterns.append(verifier_pattern)
        
        # Cross-cluster patterns
        if len(clusters) > 1:
            cross_patterns = self._analyze_cross_cluster_patterns(clusters, df)
            patterns.extend(cross_patterns)
        
        return patterns
    
    def _generate_cluster_description(self, cluster_items: pd.DataFrame) -> str:
        """
        Generate a human-readable description of a cluster.
        
        Args:
            cluster_items: DataFrame containing items in the cluster
            
        Returns:
            String description of the cluster
        """
        mean_score = cluster_items['mean_score'].mean()
        score_desc = "high" if mean_score > 0.8 else ("low" if mean_score < 0.2 else "medium")
        
        content_types = cluster_items['content_type'].unique()
        type_desc = ", ".join(content_types)
        
        std_score = cluster_items['std_score'].mean()
        consistency_desc = "consistent" if std_score < 0.1 else ("varied" if std_score > 0.2 else "moderately consistent")
        
        return f"Cluster of {len(cluster_items)} {type_desc} items with {consistency_desc} {score_desc} verification scores (avg={mean_score:.2f})"
    
    def _analyze_cluster_time_pattern(self, cluster_fossils: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze time-based patterns within a cluster.
        
        Args:
            cluster_fossils: DataFrame containing fossils for a cluster
            
        Returns:
            Dictionary with time pattern analysis or None if no significant pattern
        """
        if len(cluster_fossils) < 5:
            return None
        
        # Sort by timestamp
        cluster_fossils = cluster_fossils.sort_values('timestamp')
        
        # Check for time trends
        x = np.array((cluster_fossils['timestamp'] - cluster_fossils['timestamp'].min()).dt.total_seconds())
        y = cluster_fossils['verification_score'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Check if the trend is significant
        if p_value < 0.05 and abs(r_value) > 0.3:
            trend_direction = "increasing" if slope > 0 else "decreasing"
            
            return {
                'pattern_type': 'cluster_time_trend',
                'description': f"Cluster shows {trend_direction} verification scores over time (r={r_value:.2f}, p={p_value:.4f})",
                'confidence': min(1.0, abs(r_value) * (1 - p_value)),
                'data': {
                    'slope': slope,
                    'r_value': r_value,
                    'p_value': p_value
                }
            }
        
        return None
    
    def _analyze_cluster_method_pattern(self, cluster_fossils: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze verification method patterns within a cluster.
        
        Args:
            cluster_fossils: DataFrame containing fossils for a cluster
            
        Returns:
            Dictionary with method pattern analysis or None if no significant pattern
        """
        if len(cluster_fossils) < 5 or cluster_fossils['verification_method'].nunique() <= 1:
            return None
        
        # Group by method
        method_stats = cluster_fossils.groupby('verification_method')['verification_score'].agg(['mean', 'count']).reset_index()
        
        # Find the most effective method
        method_stats = method_stats[method_stats['count'] >= 3]  # Only consider methods with enough data
        
        if len(method_stats) <= 1:
            return None
        
        best_method = method_stats.loc[method_stats['mean'].idxmax()]
        worst_method = method_stats.loc[method_stats['mean'].idxmin()]
        
        if best_method['mean'] - worst_method['mean'] > 0.2:
            return {
                'pattern_type': 'cluster_method_effectiveness',
                'description': f"Method '{best_method['verification_method']}' is most effective for this cluster (score={best_method['mean']:.2f}), while '{worst_method['verification_method']}' is least effective (score={worst_method['mean']:.2f})",
                'confidence': min(1.0, (best_method['mean'] - worst_method['mean']) * 2),
                'data': {
                    'best_method': best_method['verification_method'],
                    'best_score': best_method['mean'],
                    'worst_method': worst_method['verification_method'],
                    'worst_score': worst_method['mean'],
                    'difference': best_method['mean'] - worst_method['mean']
                }
            }
        
        return None
    
    def _analyze_cluster_verifier_pattern(self, cluster_fossils: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze verifier patterns within a cluster.
        
        Args:
            cluster_fossils: DataFrame containing fossils for a cluster
            
        Returns:
            Dictionary with verifier pattern analysis or None if no significant pattern
        """
        if len(cluster_fossils) < 5 or cluster_fossils['verifier_id'].nunique() <= 1:
            return None
        
        # Group by verifier
        verifier_stats = cluster_fossils.groupby('verifier_id')['verification_score'].agg(['mean', 'count']).reset_index()
        
        # Find significant verifier patterns
        verifier_stats = verifier_stats[verifier_stats['count'] >= 3]  # Only consider verifiers with enough data
        
        if len(verifier_stats) <= 1:
            return None
        
        verifier_means = verifier_stats['mean'].values
        mean_diff = max(verifier_means) - min(verifier_means)
        
        if mean_diff > 0.2:
            high_verifier = verifier_stats.loc[verifier_stats['mean'].idxmax()]
            low_verifier = verifier_stats.loc[verifier_stats['mean'].idxmin()]
            
            return {
                'pattern_type': 'cluster_verifier_bias',
                'description': f"Verifier '{high_verifier['verifier_id']}' consistently scores this cluster higher (avg={high_verifier['mean']:.2f}) than '{low_verifier['verifier_id']}' (avg={low_verifier['mean']:.2f})",
                'confidence': min(1.0, mean_diff * 2),
                'data': {
                    'high_verifier': high_verifier['verifier_id'],
                    'high_score': high_verifier['mean'],
                    'low_verifier': low_verifier['verifier_id'],
                    'low_score': low_verifier['mean'],
                    'difference': mean_diff
                }
            }
        
        return None
    
    def _analyze_cross_cluster_patterns(self, clusters: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze patterns across different clusters.
        
        Args:
            clusters: List of cluster descriptions
            df: Original DataFrame containing verification data
            
        Returns:
            List of identified cross-cluster patterns
        """
        if len(clusters) <= 1:
            return []
        
        patterns = []
        
        # Analyze temporal relationships between clusters
        time_pattern = self._analyze_cross_cluster_temporal(clusters, df)
        if time_pattern:
            patterns.append(time_pattern)
        
        # Analyze method effectiveness across clusters
        method_pattern = self._analyze_cross_cluster_methods(clusters, df)
        if method_pattern:
            patterns.extend(method_pattern)
        
        # Analyze verifier behavior across clusters
        verifier_pattern = self._analyze_cross_cluster_verifiers(clusters, df)
        if verifier_pattern:
            patterns.extend(verifier_pattern)
        
        return patterns
    
    def _analyze_cross_cluster_temporal(self, clusters: List[Dict[str, Any]], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze temporal relationships between clusters.
        
        Args:
            clusters: List of cluster descriptions
            df: Original DataFrame containing verification data
            
        Returns:
            Dictionary with temporal analysis or None if no significant pattern
        """
        # Get timestamps for each cluster
        cluster_times = {}
        for cluster in clusters:
            fingerprints = cluster['content_fingerprints']
            cluster_fossils = df[df['content_fingerprint'].isin(fingerprints)]
            if len(cluster_fossils) > 0:
                cluster_times[cluster['cluster_id']] = {
                    'first': cluster_fossils['timestamp'].min(),
                    'last': cluster_fossils['timestamp'].max(),
                    'mean': cluster_fossils['timestamp'].mean()
                }
        
        if len(cluster_times) <= 1:
            return None
        
        # Check for temporal sequences
        clusters_by_time = sorted(cluster_times.keys(), key=lambda c: cluster_times[c]['mean'])
        
        # If the clusters are clearly separated in time
        time_gaps = []
        for i in range(1, len(clusters_by_time)):
            prev_cluster = clusters_by_time[i-1]
            curr_cluster = clusters_by_time[i]
            
            prev_last = cluster_times[prev_cluster]['last']
            curr_first = cluster_times[curr_cluster]['first']
            
            gap = (curr_first - prev_last).total_seconds() / 3600  # gap in hours
            time_gaps.append((prev_cluster, curr_cluster, gap))
        
        # If we have significant time gaps
        significant_gaps = [(c1, c2, gap) for c1, c2, gap in time_gaps if gap > 24]  # gaps larger than a day
        
        if significant_gaps:
            description_parts = []
            for c1, c2, gap in significant_gaps:
                gap_days = gap / 24
                description_parts.append(f"Cluster {c1} precedes {c2} by {gap_days:.1f} days")
            
            return {
                'pattern_type': 'cross_cluster_temporal_sequence',
                'description': ". ".join(description_parts),
                'confidence': min(1.0, 0.5 + len(significant_gaps) * 0.1),
                'data': {
                    'clusters_by_time': clusters_by_time,
                    'significant_gaps': significant_gaps
                }
            }
        
        return None
    
    def _analyze_cross_cluster_methods(self, clusters: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze verification method effectiveness across clusters.
        
        Args:
            clusters: List of cluster descriptions
            df: Original DataFrame containing verification data
            
        Returns:
            List of method effectiveness patterns
        """
        if len(clusters) <= 1 or df['verification_method'].nunique() <= 1:
            return []
        
        patterns = []
        
        # Analyze method effectiveness per cluster
        method_effectiveness = {}
        for cluster in clusters:
            fingerprints = cluster['content_fingerprints']
            cluster_fossils = df[df['content_fingerprint'].isin(fingerprints)]
            
            if len(cluster_fossils) < 5:
                continue
            
            # Group by method
            method_stats = cluster_fossils.groupby('verification_method')['verification_score'].agg(['mean', 'count']).reset_index()
            method_stats = method_stats[method_stats['count'] >= 3]  # Only consider methods with enough data
            
            if len(method_stats) <= 1:
                continue
            
            method_effectiveness[cluster['cluster_id']] = {
                row['verification_method']: row['mean'] for _, row in method_stats.iterrows()
            }
        
        if len(method_effectiveness) <= 1:
            return []
        
        # Find methods that perform differently across clusters
        all_methods = set()
        for cluster_methods in method_effectiveness.values():
            all_methods.update(cluster_methods.keys())
        
        for method in all_methods:
            clusters_with_method = {c: scores[method] for c, scores in method_effectiveness.items() if method in scores}
            
            if len(clusters_with_method) <= 1:
                continue
            
            # Calculate variance in method effectiveness across clusters
            method_scores = list(clusters_with_method.values())
            method_variance = np.var(method_scores)
            
            if method_variance > 0.01:
                best_cluster = max(clusters_with_method.items(), key=lambda x: x[1])[0]
                worst_cluster = min(clusters_with_method.items(), key=lambda x: x[1])[0]
                
                patterns.append({
                    'pattern_type': 'cross_cluster_method_variance',
                    'description': f"Method '{method}' is much more effective for cluster {best_cluster} (score={clusters_with_method[best_cluster]:.2f}) than for cluster {worst_cluster} (score={clusters_with_method[worst_cluster]:.2f})",
                    'confidence': min(1.0, method_variance * 50),  # Scale confidence based on variance
                    'data': {
                        'method': method,
                        'best_cluster': best_cluster,
                        'best_score': clusters_with_method[best_cluster],
                        'worst_cluster': worst_cluster,
                        'worst_score': clusters_with_method[worst_cluster],
                        'difference': clusters_with_method[best_cluster] - clusters_with_method[worst_cluster]
                    }
                })
        
        return patterns
    
    def _analyze_cross_cluster_verifiers(self, clusters: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze verifier behavior across clusters.
        
        Args:
            clusters: List of cluster descriptions
            df: Original DataFrame containing verification data
            
        Returns:
            List of verifier behavior patterns
        """
        if len(clusters) <= 1 or df['verifier_id'].nunique() <= 1:
            return []
        
        patterns = []
        
        # Analyze verifier behavior per cluster
        verifier_behavior = {}
        for cluster in clusters:
            fingerprints = cluster['content_fingerprints']
            cluster_fossils = df[df['content_fingerprint'].isin(fingerprints)]
            
            if len(cluster_fossils) < 5:
                continue
            
            # Group by verifier
            verifier_stats = cluster_fossils.groupby('verifier_id')['verification_score'].agg(['mean', 'count']).reset_index()
            verifier_stats = verifier_stats[verifier_stats['count'] >= 3]  # Only consider verifiers with enough data
            
            if len(verifier_stats) <= 1:
                continue
            
            verifier_behavior[cluster['cluster_id']] = {
                row['verifier_id']: row['mean'] for _, row in verifier_stats.iterrows()
            }
        
        if len(verifier_behavior) <= 1:
            return []
        
        # Find verifiers that behave differently across clusters
        all_verifiers = set()
        for cluster_verifiers in verifier_behavior.values():
            all_verifiers.update(cluster_verifiers.keys())
        
        for verifier in all_verifiers:
            clusters_with_verifier = {c: scores[verifier] for c, scores in verifier_behavior.items() if verifier in scores}
            
            if len(clusters_with_verifier) <= 1:
                continue
            
            # Calculate variance in verifier behavior across clusters
            verifier_scores = list(clusters_with_verifier.values())
            verifier_variance = np.var(verifier_scores)
            
            if verifier_variance > 0.01:
                best_cluster = max(clusters_with_verifier.items(), key=lambda x: x[1])[0]
                worst_cluster = min(clusters_with_verifier.items(), key=lambda x: x[1])[0]
                
                patterns.append({
                    'pattern_type': 'cross_cluster_verifier_bias',
                    'description': f"Verifier '{verifier}' assigns much higher scores to cluster {best_cluster} (score={clusters_with_verifier[best_cluster]:.2f}) than to cluster {worst_cluster} (score={clusters_with_verifier[worst_cluster]:.2f})",
                    'confidence': min(1.0, verifier_variance * 50),  # Scale confidence based on variance
                    'data': {
                        'verifier': verifier,
                        'best_cluster': best_cluster,
                        'best_score': clusters_with_verifier[best_cluster],
                        'worst_cluster': worst_cluster,
                        'worst_score': clusters_with_verifier[worst_cluster],
                        'difference': clusters_with_verifier[best_cluster] - clusters_with_verifier[worst_cluster]
                    }
                })
        
        return patterns
    
    def _calculate_pattern_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """
        Calculate an overall confidence score for a set of patterns.
        
        Args:
            patterns: List of identified patterns
            
        Returns:
            Overall confidence score
        """
        if not patterns:
            return 0.0
        
        # Weight patterns by their individual confidence
        weighted_sum = sum(pattern.get('confidence', 0.5) for pattern in patterns)
        
        # Factor in the number of patterns (more patterns = higher confidence, up to a point)
        pattern_factor = min(1.0, len(patterns) / 10)
        
        # Calculate overall confidence
        overall_confidence = min(1.0, weighted_sum / len(patterns) * (0.5 + 0.5 * pattern_factor))
        
        return overall_confidence
    
    def _empty_archaeology_report(self) -> Dict[str, Any]:
        """
        Generate an empty archaeology report for when no fossils are found.
        
        Returns:
            Empty report dictionary
        """
        return {
            'report_id': f"AR-{datetime.now().strftime('%Y%m%d%H%M%S')}-empty",
            'timestamp': datetime.now().isoformat(),
            'criteria': {},
            'fossils_analyzed': 0,
            'temporal_patterns': [],
            'anomalies': [],
            'consistencies': [],
            'confidence_evolution': {'evolution_type': 'empty', 'data': {}}
        }
    
    def _report_has_significant_patterns(self, report: Dict[str, Any]) -> bool:
        """
        Determine if a report contains significant patterns worth storing.
        
        Args:
            report: Archaeology report
            
        Returns:
            True if the report contains significant patterns, False otherwise
        """
        # Check if there are any patterns or anomalies
        has_patterns = len(report.get('temporal_patterns', [])) > 0
        has_anomalies = len(report.get('anomalies', [])) > 0
        has_consistencies = len(report.get('consistencies', [])) > 0
        
        # Check for significant confidence evolution
        confidence_evolution = report.get('confidence_evolution', {})
        significant_evolution = confidence_evolution.get('evolution_type', '') not in ['empty', 'insufficient_data', 'mixed_confidence']
        
        return has_patterns or has_anomalies or has_consistencies or significant_evolution
    
    def _store_report_as_pattern(self, report: Dict[str, Any]) -> str:
        """
        Store an archaeology report as a pattern in the database.
        
        Args:
            report: Archaeology report
            
        Returns:
            The pattern ID
        """
        # Extract related fossils
        related_fossils = []
        for fossil in report.get('fossils', []):
            if 'fossil_id' in fossil:
                related_fossils.append(fossil['fossil_id'])
        
        # Generate a description based on the report
        description_parts = []
        
        # Add temporal patterns to description
        for pattern in report.get('temporal_patterns', []):
            if 'description' in pattern:
                description_parts.append(pattern['description'])
        
        # Add top anomalies to description (limit to 3)
        anomalies = sorted(report.get('anomalies', []), key=lambda x: x.get('confidence', 0), reverse=True)
        for anomaly in anomalies[:3]:
            if 'description' in anomaly:
                description_parts.append(anomaly['description'])
        
        # Add top consistencies to description (limit to 3)
        consistencies = sorted(report.get('consistencies', []), key=lambda x: x.get('confidence', 0), reverse=True)
        for consistency in consistencies[:3]:
            if 'description' in consistency:
                description_parts.append(consistency['description'])
        
        # Add confidence evolution to description
        confidence_evolution = report.get('confidence_evolution', {})
        if confidence_evolution.get('evolution_type') not in ['empty', 'insufficient_data', 'mixed_confidence']:
            description_parts.append(f"Confidence evolution: {confidence_evolution.get('evolution_type')}")
        
        # Generate the final description
        description = ". ".join(description_parts)
        if not description:
            description = "Archaeological analysis of verification fossils"
        
        # Calculate confidence based on the number and confidence of patterns
        pattern_confidence = self._calculate_pattern_confidence(
            report.get('temporal_patterns', []) + 
            report.get('anomalies', []) + 
            report.get('consistencies', [])
        )
        
        # Store the pattern
        pattern_id = self.fossil_db.store_pattern(
            pattern_type='archaeological_analysis',
            pattern_description=description,
            confidence_score=pattern_confidence,
            related_fossils=related_fossils,
            metadata={
                'report_id': report.get('report_id'),
                'temporal_patterns_count': len(report.get('temporal_patterns', [])),
                'anomalies_count': len(report.get('anomalies', [])),
                'consistencies_count': len(report.get('consistencies', [])),
                'confidence_evolution_type': report.get('confidence_evolution', {}).get('evolution_type', 'unknown')
            }
        )
        
        logger.info(f"Stored archaeology report as pattern with ID: {pattern_id}")
        return pattern_id
    
    def _generate_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate visualizations for an archaeology report.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary containing visualization data
        """
        visualizations = {}
        
        # Only proceed if we have enough data
        if len(df) < 5:
            return visualizations
        
        # Ensure datetime format for timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Generate verification score timeline
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot verification scores
            ax.plot(df['timestamp'], df['verification_score'], marker='o', linestyle='-', label='Verification Score')
            
            # Calculate and plot running average
            df['running_avg'] = df['verification_score'].rolling(window=5, min_periods=1).mean()
            ax.plot(df['timestamp'], df['running_avg'], linestyle='--', color='red', label='Running Average (5)')
            
            # Add labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Verification Score')
            ax.set_title('Verification Score Timeline')
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format dates
            fig.autofmt_xdate()
            
            # Save to buffer
            from io import BytesIO
            import base64
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert to base64 for embedding
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['score_timeline'] = f"data:image/png;base64,{img_str}"
            
            # Close figure to free memory
            plt.close(fig)
        
        except Exception as e:
            logger.warning(f"Error generating verification score timeline: {e}")
        
        # Generate method comparison visualization
        try:
            if df['verification_method'].nunique() > 1:
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Group by method and calculate statistics
                method_stats = df.groupby('verification_method')['verification_score'].agg(['mean', 'std', 'count']).reset_index()
                method_stats = method_stats[method_stats['count'] >= 3]  # Only consider methods with enough data
                
                if len(method_stats) > 0:
                    # Sort by mean score
                    method_stats = method_stats.sort_values('mean', ascending=False)
                    
                    # Create bar chart
                    bars = ax.bar(method_stats['verification_method'], method_stats['mean'], yerr=method_stats['std'], 
                                 capsize=5, alpha=0.7)
                    
                    # Add labels and title
                    ax.set_xlabel('Verification Method')
                    ax.set_ylabel('Average Score')
                    ax.set_title('Verification Methods Comparison')
                    
                    # Add count labels
                    for i, bar in enumerate(bars):
                        count = method_stats.iloc[i]['count']
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, 
                               f"n={count}", ha='center', va='bottom')
                    
                    # Add grid
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Save to buffer
                    buffer = BytesIO()
                    fig.savefig(buffer, format='png')
                    buffer.seek(0)
                    
                    # Convert to base64 for embedding
                    img_str = base64.b64encode(buffer.read()).decode('utf-8')
                    visualizations['method_comparison'] = f"data:image/png;base64,{img_str}"
                    
                    # Close figure to free memory
                    plt.close(fig)
        
        except Exception as e:
            logger.warning(f"Error generating method comparison visualization: {e}")
        
        return visualizations
