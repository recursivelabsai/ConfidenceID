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
            'trend_direction': trend_direction,
            'trend_significance': {
                'slope': slope,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err
            },
            'trend_magnitude_percent': trend_magnitude,
            'data_for_visualization': {
                'x': x.tolist(),
                'actual_values': y.tolist(),
                'trend_line': trend_line.tolist(),
                'dates': daily_scores.index.strftime('%Y-%m-%d').tolist()
            }
        }
    
    def _analyze_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in verification data.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with seasonality analysis
        """
        # Convert to daily time series
        daily_scores = df.set_index('timestamp')['verification_score'].resample('D').mean()
        
        # Remove NaN values (days without data)
        daily_scores = daily_scores.dropna()
        
        # If not enough data points, return empty analysis
        if len(daily_scores) < 14:  # Need at least two weeks of data
            return {
                'has_seasonality': False,
                'reason': 'insufficient_data_points'
            }
        
        # Check if we have enough data for meaningful seasonal decomposition
        # Need at least 2 full cycles for the smallest period we're checking
        min_period = 7  # Weekly seasonality
        if len(daily_scores) < min_period * 2:
            return {
                'has_seasonality': False,
                'reason': 'insufficient_data_for_seasonal_decomposition'
            }
        
        # Dictionary to store results for different periods
        seasonality_results = {}
        detected_seasonality = False
        
        # Check for common periods
        periods_to_check = []
        
        # Only check weekly if we have enough data
        if len(daily_scores) >= 14:  # At least 2 weeks
            periods_to_check.append(('weekly', 7))
        
        # Only check biweekly if we have enough data
        if len(daily_scores) >= 28:  # At least 4 weeks
            periods_to_check.append(('biweekly', 14))
        
        # Only check monthly if we have enough data
        if len(daily_scores) >= 60:  # At least 2 months
            periods_to_check.append(('monthly', 30))
        
        # Only check quarterly if we have enough data
        if len(daily_scores) >= 180:  # At least 6 months
            periods_to_check.append(('quarterly', 90))
        
        for period_name, period_length in periods_to_check:
            try:
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(daily_scores, period=period_length, extrapolate_trend='freq')
                
                # Get seasonal component
                seasonal = decomposition.seasonal
                
                # Calculate strength of seasonality (ratio of seasonal to combined seasonal+residual variance)
                seasonal_var = np.var(seasonal)
                residual_var = np.var(decomposition.resid)
                if seasonal_var + residual_var > 0:
                    seasonality_strength = seasonal_var / (seasonal_var + residual_var)
                else:
                    seasonality_strength = 0
                
                # Check if seasonality is significant
                is_significant = seasonality_strength > 0.3
                
                if is_significant:
                    detected_seasonality = True
                
                # Extract peak and trough days
                seasonal_by_period = seasonal.values[:period_length]
                peak_day = np.argmax(seasonal_by_period)
                trough_day = np.argmin(seasonal_by_period)
                
                # Store results
                seasonality_results[period_name] = {
                    'is_significant': is_significant,
                    'strength': seasonality_strength,
                    'peak_day': int(peak_day),
                    'trough_day': int(trough_day),
                    'peak_to_trough_magnitude': float(np.max(seasonal_by_period) - np.min(seasonal_by_period)),
                    'seasonal_pattern': seasonal_by_period.tolist()
                }
            
            except Exception as e:
                logger.warning(f"Error analyzing {period_name} seasonality: {e}")
                seasonality_results[period_name] = {
                    'is_significant': False,
                    'error': str(e)
                }
        
        # Check for daily patterns if we have time-of-day data
        has_hourly_pattern = False
        hourly_pattern = {}
        
        if len(df) >= 48:  # At least 48 data points for hourly analysis
            # Extract hour from timestamp
            df['hour'] = df['timestamp'].dt.hour
            
            # Group by hour and calculate statistics
            hourly_stats = df.groupby('hour')['verification_score'].agg(['mean', 'std', 'count']).reset_index()
            
            # Only consider hours with enough data
            hourly_stats = hourly_stats[hourly_stats['count'] >= 3]
            
            if len(hourly_stats) >= 6:  # At least 6 hours with enough data
                # Calculate variance in hourly means
                hourly_var = np.var(hourly_stats['mean'])
                
                # Check if hourly pattern is significant
                hourly_significant = hourly_var > 0.01
                
                if hourly_significant:
                    has_hourly_pattern = True
                    
                    # Find peak and trough hours
                    peak_hour = int(hourly_stats.loc[hourly_stats['mean'].idxmax()]['hour'])
                    trough_hour = int(hourly_stats.loc[hourly_stats['mean'].idxmin()]['hour'])
                    
                    # Store hourly pattern
                    hourly_pattern = {
                        'is_significant': True,
                        'variance': hourly_var,
                        'peak_hour': peak_hour,
                        'trough_hour': trough_hour,
                        'hourly_means': dict(zip(hourly_stats['hour'].astype(int).tolist(), 
                                                hourly_stats['mean'].tolist())),
                        'peak_to_trough_magnitude': float(hourly_stats['mean'].max() - hourly_stats['mean'].min())
                    }
        
        # Check for day-of-week patterns
        has_dow_pattern = False
        dow_pattern = {}
        
        if len(df) >= 14:  # At least 14 data points for day-of-week analysis
            # Extract day of week from timestamp
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Group by day of week and calculate statistics
            dow_stats = df.groupby('day_of_week')['verification_score'].agg(['mean', 'std', 'count']).reset_index()
            
            # Only consider days with enough data
            dow_stats = dow_stats[dow_stats['count'] >= 3]
            
            if len(dow_stats) >= 3:  # At least 3 days with enough data
                # Calculate variance in daily means
                dow_var = np.var(dow_stats['mean'])
                
                # Check if day-of-week pattern is significant
                dow_significant = dow_var > 0.01
                
                if dow_significant:
                    has_dow_pattern = True
                    
                    # Find peak and trough days
                    peak_dow = int(dow_stats.loc[dow_stats['mean'].idxmax()]['day_of_week'])
                    trough_dow = int(dow_stats.loc[dow_stats['mean'].idxmin()]['day_of_week'])
                    
                    # Convert numeric day of week to name
                    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    # Store day-of-week pattern
                    dow_pattern = {
                        'is_significant': True,
                        'variance': dow_var,
                        'peak_day': dow_names[peak_dow],
                        'trough_day': dow_names[trough_dow],
                        'daily_means': {dow_names[int(day)]: mean for day, mean in 
                                      zip(dow_stats['day_of_week'].tolist(), dow_stats['mean'].tolist())},
                        'peak_to_trough_magnitude': float(dow_stats['mean'].max() - dow_stats['mean'].min())
                    }
        
        return {
            'has_seasonality': detected_seasonality or has_hourly_pattern or has_dow_pattern,
            'period_analysis': seasonality_results,
            'hourly_pattern': hourly_pattern if has_hourly_pattern else {'is_significant': False},
            'day_of_week_pattern': dow_pattern if has_dow_pattern else {'is_significant': False}
        }
    
    def _analyze_event_density(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the density of verification events over time.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with event density analysis
        """
        # Convert to daily counts
        daily_counts = df.groupby(df['timestamp'].dt.date).size()
        
        # If not enough data points, return empty analysis
        if len(daily_counts) < 7:  # At least a week of data
            return {
                'has_pattern': False,
                'reason': 'insufficient_data_points'
            }
        
        # Calculate statistics
        mean_density = daily_counts.mean()
        std_density = daily_counts.std()
        cv_density = std_density / mean_density if mean_density > 0 else 0  # Coefficient of variation
        
        # Identify days with unusually high/low density
        high_threshold = mean_density + 2 * std_density
        low_threshold = max(0, mean_density - 2 * std_density)
        
        high_density_days = daily_counts[daily_counts > high_threshold]
        low_density_days = daily_counts[daily_counts < low_threshold]
        
        # Check for trends in density
        x = np.array(range(len(daily_counts)))
        y = daily_counts.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Check if the trend is significant
        density_trend_significant = p_value < 0.05 and abs(r_value) > 0.3
        
        # Calculate percentage of days with activity
        total_date_range = (df['timestamp'].max().date() - df['timestamp'].min().date()).days + 1
        activity_percentage = len(daily_counts) / total_date_range * 100
        
        # Determine if there's a burst pattern
        # (periods of high activity followed by periods of low activity)
        has_burst_pattern = False
        burst_pattern = {}
        
        if len(daily_counts) >= 14:  # Need at least two weeks to detect bursts
            # Calculate autocorrelation
            autocorr = pd.Series(daily_counts).autocorr(lag=1)
            
            # Calculate runs of high and low activity
            is_high = daily_counts > mean_density
            runs = [(k, sum(1 for _ in g)) for k, g in itertools.groupby(is_high)]
            
            # Check if we have significant runs
            if len(runs) >= 4:  # At least a few transitions
                # Calculate average run length
                avg_high_run = np.mean([length for is_high, length in runs if is_high])
                avg_low_run = np.mean([length for is_high, length in runs if not is_high])
                
                # Determine if burst pattern is significant
                has_burst_pattern = abs(autocorr) < 0.3 and avg_high_run >= 2 and avg_low_run >= 2
                
                if has_burst_pattern:
                    burst_pattern = {
                        'average_high_run_days': avg_high_run,
                        'average_low_run_days': avg_low_run,
                        'autocorrelation': autocorr,
                        'run_count': len(runs)
                    }
        
        return {
            'statistics': {
                'mean_verifications_per_day': mean_density,
                'std_verifications_per_day': std_density,
                'coefficient_of_variation': cv_density,
                'max_verifications_per_day': daily_counts.max(),
                'min_verifications_per_day': daily_counts.min(),
                'activity_percentage': activity_percentage
            },
            'density_trend': {
                'has_trend': density_trend_significant,
                'slope': slope,
                'r_value': r_value,
                'p_value': p_value
            },
            'unusual_days': {
                'high_density_days': [(date.strftime('%Y-%m-%d'), count) for date, count in 
                                    zip(high_density_days.index, high_density_days.values)],
                'low_density_days': [(date.strftime('%Y-%m-%d'), count) for date, count in 
                                   zip(low_density_days.index, low_density_days.values)],
            },
            'burst_pattern': {
                'has_burst_pattern': has_burst_pattern,
                'details': burst_pattern if has_burst_pattern else {}
            },
            'data_for_visualization': {
                'dates': [date.strftime('%Y-%m-%d') for date in daily_counts.index],
                'counts': daily_counts.values.tolist()
            }
        }
    
    def _analyze_periodicity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze periodic patterns in verification data using spectral analysis.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with periodicity analysis
        """
        # Convert to daily time series
        daily_scores = df.set_index('timestamp')['verification_score'].resample('D').mean()
        
        # Remove NaN values (days without data)
        daily_scores = daily_scores.dropna()
        
        # If not enough data points, return empty analysis
        if len(daily_scores) < 30:  # Need at least a month of data
            return {
                'has_periodicity': False,
                'reason': 'insufficient_data_points'
            }
        
        # Detrend the series to focus on periodic components
        # First, fit a linear trend
        x = np.array(range(len(daily_scores)))
        y = daily_scores.values
        
        slope, intercept, _, _, _ = stats.linregress(x, y)
        trend = intercept + slope * x
        detrended = y - trend
        
        # Calculate autocorrelation function (ACF)
        max_lag = min(40, len(detrended) // 2)  # Up to 40 days or half the series length
        acf_values = acf(detrended, nlags=max_lag, fft=True)
        
        # Find peaks in ACF (potential periods)
        # Exclude lag 0 (which is always 1)
        peaks, _ = find_peaks(acf_values[1:], height=0.2)
        peaks = peaks + 1  # Adjust indices to account for excluded lag 0
        
        # Sort peaks by correlation strength
        sorted_peaks = sorted([(peak, acf_values[peak]) for peak in peaks], key=lambda x: x[1], reverse=True)
        
        # Extract significant periods
        significant_periods = []
        
        for period, correlation in sorted_peaks:
            if correlation > 0.3:  # Consider only reasonably strong correlations
                significant_periods.append({
                    'period_days': int(period),
                    'correlation': float(correlation),
                    'significance': 'strong' if correlation > 0.5 else 'moderate'
                })
        
        # Check for periodicity in the frequency domain
        # Compute the power spectral density
        has_spectral_peaks = False
        spectral_peaks = []
        
        if len(detrended) >= 60:  # Only if we have enough data
            try:
                from scipy import signal
                
                # Calculate power spectral density
                f, Pxx = signal.periodogram(detrended)
                
                # Convert frequencies to periods (in days)
                periods = 1 / f[1:]  # Exclude the DC component (f=0)
                powers = Pxx[1:]
                
                # Find peaks in the spectrum
                spectral_peak_indices, _ = find_peaks(powers, height=np.mean(powers) * 2)
                
                if len(spectral_peak_indices) > 0:
                    has_spectral_peaks = True
                    
                    # Sort peaks by power
                    sorted_spectral_peaks = sorted([(periods[i], powers[i]) for i in spectral_peak_indices], 
                                                 key=lambda x: x[1], reverse=True)
                    
                    # Extract top peaks
                    for period, power in sorted_spectral_peaks[:3]:  # Top 3 peaks
                        if period <= len(detrended) / 2:  # Only periods shorter than half the series length
                            spectral_peaks.append({
                                'period_days': float(period),
                                'power': float(power),
                                'normalized_power': float(power / np.mean(powers))
                            })
            except Exception as e:
                logger.warning(f"Error in spectral analysis: {e}")
        
        return {
            'has_periodicity': len(significant_periods) > 0 or has_spectral_peaks,
            'time_domain_analysis': {
                'significant_periods': significant_periods,
                'acf_values': acf_values.tolist(),
                'acf_lags': list(range(len(acf_values)))
            },
            'frequency_domain_analysis': {
                'has_spectral_peaks': has_spectral_peaks,
                'spectral_peaks': spectral_peaks
            },
            'interpretation': {
                'strongest_period': significant_periods[0]['period_days'] if significant_periods else None,
                'period_description': self._interpret_period(significant_periods[0]['period_days']) if significant_periods else None
            }
        }
    
    def _interpret_period(self, period_days: int) -> str:
        """
        Interpret a detected period in human-readable terms.
        
        Args:
            period_days: Period length in days
            
        Returns:
            Human-readable interpretation of the period
        """
        if period_days == 7 or period_days == 6 or period_days == 8:
            return "weekly"
        elif period_days == 14 or period_days == 13 or period_days == 15:
            return "biweekly"
        elif period_days >= 28 and period_days <= 31:
            return "monthly"
        elif period_days >= 89 and period_days <= 92:
            return "quarterly"
        elif period_days >= 180 and period_days <= 186:
            return "semi-annual"
        elif period_days >= 365 and period_days <= 371:
            return "annual"
        else:
            return f"{period_days}-day"
    
    def _analyze_method_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how verification methods have evolved over time.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with method evolution analysis
        """
        # Check if we have multiple methods
        methods = df['verification_method'].unique()
        
        if len(methods) <= 1:
            return {
                'has_evolution': False,
                'reason': 'only_one_method'
            }
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Divide the time range into periods
        time_range = df['timestamp'].max() - df['timestamp'].min()
        days = time_range.days
        
        # Determine appropriate period length
        if days < 30:
            period = 'D'  # Daily
            period_name = 'daily'
        elif days < 180:
            period = 'W'  # Weekly
            period_name = 'weekly'
        else:
            period = 'M'  # Monthly
            period_name = 'monthly'
        
        # Create time bins
        df['time_bin'] = pd.to_datetime(df['timestamp']).dt.to_period(period)
        
        # Group by time bin and method
        method_counts = df.groupby(['time_bin', 'verification_method']).size().unstack(fill_value=0)
        
        # Calculate method proportions
        method_props = method_counts.div(method_counts.sum(axis=1), axis=0)
        
        # Check for clear trends in method usage
        method_trends = {}
        significant_trend_found = False
        
        for method in methods:
            if method in method_props.columns:
                # Get method proportion over time
                method_prop = method_props[method].reset_index(drop=True)
                
                if len(method_prop) >= 5:  # Need at least 5 time periods for regression
                    # Perform linear regression
                    x = np.array(range(len(method_prop)))
                    y = method_prop.values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Check if the trend is significant
                    trend_significant = p_value < 0.05 and abs(r_value) > 0.3
                    
                    if trend_significant:
                        significant_trend_found = True
                    
                    # Calculate trend magnitude (percentage change)
                    if len(y) > 1 and y[0] != 0:
                        start_prop = y[0]
                        end_prop = y[-1]
                        change_percentage = (end_prop - start_prop) / start_prop * 100
                    else:
                        change_percentage = 0
                    
                    # Store trend information
                    method_trends[method] = {
                        'trend_significant': trend_significant,
                        'slope': slope,
                        'r_value': r_value,
                        'p_value': p_value,
                        'change_percentage': change_percentage,
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'proportion_values': method_prop.tolist()
                    }
        
        # Analyze method effectiveness evolution
        method_effectiveness = {}
        effectiveness_evolution = {}
        
        for method in methods:
            # Group by time bin and calculate mean score for this method
            method_df = df[df['verification_method'] == method]
            
            if len(method_df) >= 10:  # Need at least 10 data points
                method_scores = method_df.groupby('time_bin')['verification_score'].mean()
                
                if len(method_scores) >= 5:  # Need at least 5 time periods
                    # Perform linear regression on scores
                    x = np.array(range(len(method_scores)))
                    y = method_scores.values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Check if the trend is significant
                    trend_significant = p_value < 0.05 and abs(r_value) > 0.3
                    
                    # Calculate trend magnitude (percentage change)
                    if len(y) > 1 and y[0] != 0:
                        start_score = y[0]
                        end_score = y[-1]
                        change_percentage = (end_score - start_score) / start_score * 100
                    else:
                        change_percentage = 0
                    
                    # Store effectiveness information
                    method_effectiveness[method] = {
                        'trend_significant': trend_significant,
                        'slope': slope,
                        'r_value': r_value,
                        'p_value': p_value,
                        'change_percentage': change_percentage,
                        'direction': 'improving' if slope > 0 else 'declining',
                        'score_values': y.tolist(),
                        'current_effectiveness': float(y[-1])
                    }
        
        # Analyze whether the most effective methods have changed over time
        if len(method_effectiveness) >= 2:
            # Divide into early and late periods
            mid_point = len(df) // 2
            early_df = df.iloc[:mid_point]
            late_df = df.iloc[mid_point:]
            
            # Calculate effectiveness in each period
            early_effectiveness = early_df.groupby('verification_method')['verification_score'].mean()
            late_effectiveness = late_df.groupby('verification_method')['verification_score'].mean()
            
            # Find best method in each period
            if len(early_effectiveness) > 0 and len(late_effectiveness) > 0:
                early_best = early_effectiveness.idxmax()
                late_best = late_effectiveness.idxmax()
                
                early_score = early_effectiveness.max()
                late_score = late_effectiveness.max()
                
                # Check if the best method has changed
                best_method_changed = early_best != late_best
                
                effectiveness_evolution = {
                    'best_method_changed': best_method_changed,
                    'early_period': {
                        'best_method': early_best,
                        'best_score': float(early_score),
                        'methods_ranked': [(m, float(s)) for m, s in 
                                        sorted(early_effectiveness.items(), key=lambda x: x[1], reverse=True)]
                    },
                    'late_period': {
                        'best_method': late_best,
                        'best_score': float(late_score),
                        'methods_ranked': [(m, float(s)) for m, s in 
                                        sorted(late_effectiveness.items(), key=lambda x: x[1], reverse=True)]
                    }
                }
        
        # Create data for visualization
        visualization_data = {
            'time_periods': [str(period) for period in method_counts.index],
            'method_counts': {method: method_counts[method].tolist() if method in method_counts.columns else [] 
                             for method in methods},
            'method_proportions': {method: method_props[method].tolist() if method in method_props.columns else [] 
                                  for method in methods}
        }
        
        return {
            'has_evolution': significant_trend_found or bool(effectiveness_evolution.get('best_method_changed', False)),
            'time_period_type': period_name,
            'method_usage_trends': method_trends,
            'method_effectiveness_trends': method_effectiveness,
            'effectiveness_evolution': effectiveness_evolution,
            'visualization_data': visualization_data
        }
    
    def _analyze_verifier_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how verifiers have evolved over time.
        
        Args:
            df: DataFrame containing verification data
            
        Returns:
            Dictionary with verifier evolution analysis
        """
        # Check if we have verifier information
        if 'verifier_id' not in df.columns or df['verifier_id'].isna().all():
            return {
                'has_evolution': False,
                'reason': 'no_verifier_data'
            }
        
        # Drop rows with null verifier_id
        df = df.dropna(subset=['verifier_id'])
        
        # Check if we have multiple verifiers
        verifiers = df['verifier_id'].unique()
        
        if len(verifiers) <= 1:
            return {
                'has_evolution': False,
                'reason': 'only_one_verifier'
            }
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Divide the time range into periods
        time_range = df['timestamp'].max() - df['timestamp'].min()
        days = time_range.days
        
        # Determine appropriate period length
        if days < 30:
            period = 'D'  # Daily
            period_name = 'daily'
        elif days < 180:
            period = 'W'  # Weekly
            period_name = 'weekly'
        else:
            period = 'M'  # Monthly
            period_name = 'monthly'
        
        # Create time bins
        df['time_bin'] = pd.to_datetime(df['timestamp']).dt.to_period(period)
        
        # Analyze verifier count evolution
        verifier_counts_by_period = df.groupby('time_bin')['verifier_id'].nunique()
        
        # Check for trend in verifier count
        x_count = np.array(range(len(verifier_counts_by_period)))
        y_count = verifier_counts_by_period.values
        
        count_slope, count_intercept, count_r, count_p, count_stderr = stats.linregress(x_count, y_count)
        
        count_trend_significant = count_p < 0.05 and abs(count_r) > 0.3
        
        # Analyze new verifier emergence
        first_appearance = df.groupby('verifier_id')['timestamp'].min()
        verifiers_by_period = {}
        
        for period in df['time_bin'].unique():
            period_start = period.start_time
            period_end = period.end_time
            
            # Find verifiers that first appeared in this period
            new_verifiers = first_appearance[(first_appearance >= period_start) & 
                                           (first_appearance <= period_end)].index.tolist()
            
            verifiers_by_period[str(period)] = new_verifiers
        
        # Analyze verifier churn
        active_verifiers_by_period = df.groupby('time_bin')['verifier_id'].unique()
        
        verifier_churn = {}
        previous_active = set()
        
        for period, active in active_verifiers_by_period.items():
            active_set = set(active)
            
            if previous_active:
                new = active_set - previous_active
                departed = previous_active - active_set
                retained = previous_active.intersection(active_set)
                
                churn_rate = len(departed) / len(previous_active) if previous_active else 0
                
                verifier_churn[str(period)] = {
                    'new_count': len(new),
                    'departed_count': len(departed),
                    'retained_count': len(retained),
                    'churn_rate': churn_rate
                }
            
            previous_active = active_set
        
        #
