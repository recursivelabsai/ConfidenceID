"""
Field Dynamics Engine for Temporal Trust Field

This module implements the core equations and algorithms that govern how
trust fields evolve over time. It handles the application of verification
events to the trust field tensor, calculating decay, amplification, and
interference effects.

The field dynamics are governed by a set of differential equations that
model trust as a dynamic field with properties similar to physical fields,
including velocity, acceleration, decay, and amplification.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

from .schemas_and_types import (
    VerificationEvent,
    TrustFieldTensor,
    TemporalWeight,
    FieldDynamicsParameters,
    TrustFieldConfiguration,
    TrustFieldDimension,
    ContentType,
    TrustFieldSnapshot
)

logger = logging.getLogger(__name__)


class FieldDynamicsEngine:
    """
    Engine that implements the core equations for trust field evolution.
    
    This class is responsible for updating the trust field tensor in response
    to verification events, calculating temporal weights, and simulating
    the field's evolution over time.
    """
    
    def __init__(self, configuration: TrustFieldConfiguration):
        """
        Initialize the field dynamics engine with a configuration.
        
        Args:
            configuration: Configuration for the trust field
        """
        self.configuration = configuration
        self.dynamics_parameters = configuration.dynamics_parameters
        
        # Initialize the field tensor
        self.initialize_field()
        
        # History of field snapshots for analysis
        self.snapshots: List[TrustFieldSnapshot] = []
        
        # Track the last update time
        self.last_update_time = datetime.now()
    
    def initialize_field(self):
        """Initialize the trust field tensor with default values."""
        tensor_data = self.configuration.create_initial_tensor()
        
        self.field_tensor = TrustFieldTensor(
            tensor_data=tensor_data,
            dimensions=self.configuration.dimensions,
            content_types=self.configuration.content_types,
            context_keys=self.configuration.context_keys,
            context_values=self.configuration.context_values,
            timestamp=datetime.now()
        )
        
        # Take an initial snapshot
        self.take_snapshot("initialization")
    
    def take_snapshot(self, trigger: str = None):
        """
        Take a snapshot of the current field state.
        
        Args:
            trigger: Optional string describing what triggered this snapshot
        """
        snapshot = TrustFieldSnapshot(
            tensor=self.field_tensor,
            timestamp=datetime.now(),
            triggered_by_event=trigger
        )
        self.snapshots.append(snapshot)
        
        # Limit the number of snapshots to prevent memory issues
        max_snapshots = 100  # Could be configurable
        if len(self.snapshots) > max_snapshots:
            self.snapshots = self.snapshots[-max_snapshots:]
    
    def calculate_temporal_weight(self, 
                                 event: VerificationEvent, 
                                 current_time: datetime = None,
                                 current_context: Dict[str, str] = None) -> TemporalWeight:
        """
        Calculate the temporal weight of a verification event.
        
        Args:
            event: The verification event
            current_time: Current time (defaults to now)
            current_context: Current context (for context similarity)
            
        Returns:
            A TemporalWeight object
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Calculate base weight (importance of this verification type)
        base_weight = 1.0  # Default weight
        # Could be adjusted based on event type, content type, source reliability, etc.
        
        # Calculate temporal decay based on time elapsed
        time_elapsed = (current_time - event.timestamp).total_seconds()
        max_history_window_seconds = self.dynamics_parameters.max_history_window * 24 * 60 * 60
        
        if time_elapsed <= 0:
            temporal_decay = 1.0  # No decay for current or future events
        elif time_elapsed > max_history_window_seconds:
            temporal_decay = 0.0  # Complete decay for very old events
        else:
            # Calculate decay rate for this specific event type and content
            decay_rate = self.dynamics_parameters.get_decay_rate(
                event_type=event.event_type,
                content_type=event.content_type,
                context=event.context
            )
            
            # Apply exponential decay
            # e^(-decay_rate * time_elapsed)
            temporal_decay = np.exp(-decay_rate * (time_elapsed / max_history_window_seconds))
        
        # Calculate context similarity if current context is provided
        context_similarity = 1.0  # Default if no context provided
        if current_context and event.context:
            # Simple similarity: proportion of matching context keys
            matches = sum(1 for k, v in current_context.items() 
                         if k in event.context and event.context[k] == v)
            total = len(set(current_context.keys()).union(event.context.keys()))
            context_similarity = matches / total if total > 0 else 0.0
        
        # Confidence factor from the event
        confidence_factor = event.confidence
        
        return TemporalWeight(
            base_weight=base_weight,
            temporal_decay=temporal_decay,
            context_similarity=context_similarity,
            confidence_factor=confidence_factor
        )
    
    def apply_verification_event(self, 
                               event: VerificationEvent, 
                               current_time: datetime = None,
                               current_context: Dict[str, str] = None):
        """
        Apply a verification event to update the trust field.
        
        Args:
            event: The verification event
            current_time: Current time (defaults to now)
            current_context: Current context (for context similarity)
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Calculate temporal weight
        weight = self.calculate_temporal_weight(
            event=event,
            current_time=current_time,
            current_context=current_context
        )
        
        # Skip events with negligible weight
        if weight.effective_weight < self.configuration.min_event_weight:
            logger.debug(f"Skipping event {event.event_id} due to low weight: {weight.effective_weight}")
            return
        
        # Locate the indices in the tensor for this content type
        content_idx = self.field_tensor.content_types.index(event.content_type)
        
        # Determine context indices
        context_indices = []
        for key in self.field_tensor.context_keys:
            if key in event.context and event.context[key] in self.field_tensor.context_values.get(key, []):
                # Find the index of this context value
                value_idx = self.field_tensor.context_values[key].index(event.context[key])
                context_indices.append(value_idx)
            else:
                # If context not provided or not in allowed values, use 0 (default)
                context_indices.append(0)
        
        # Calculate amplification factor for this event
        amplification_factor = self.dynamics_parameters.get_amplification_factor(
            event_type=event.event_type,
            content_type=event.content_type,
            context=event.context
        )
        
        # Update the field tensor dimensions
        for i, dim in enumerate(self.field_tensor.dimensions):
            if dim == TrustFieldDimension.VALUE:
                # Update trust value
                indices = tuple([i, content_idx] + context_indices)
                current_value = self.field_tensor.tensor_data[indices]
                
                # Calculate new value based on verification score and weight
                value_change = (event.verification_score - current_value) * weight.effective_weight
                new_value = current_value + value_change
                
                # Ensure value stays in valid range [0, 1]
                new_value = max(0.0, min(1.0, new_value))
                
                # Apply change with damping to prevent wild oscillations
                damped_change = value_change * (1 - self.dynamics_parameters.damping_factor)
                damped_new_value = current_value + damped_change
                
                # Only apply changes above noise tolerance
                if abs(damped_new_value - current_value) > self.dynamics_parameters.noise_tolerance:
                    self.field_tensor.tensor_data[indices] = damped_new_value
            
            elif dim == TrustFieldDimension.VELOCITY:
                # Update velocity based on verification
                indices = tuple([i, content_idx] + context_indices)
                current_velocity = self.field_tensor.tensor_data[indices]
                
                # Calculate velocity as rate of change in value
                # High verification score increases positive velocity, low score decreases
                velocity_change = (event.verification_score - 0.5) * weight.effective_weight
                new_velocity = current_velocity + velocity_change
                
                # Apply damping to velocity
                new_velocity *= (1 - self.dynamics_parameters.damping_factor)
                
                # Only apply changes above noise tolerance
                if abs(new_velocity - current_velocity) > self.dynamics_parameters.noise_tolerance:
                    self.field_tensor.tensor_data[indices] = new_velocity
            
            elif dim == TrustFieldDimension.ACCELERATION:
                # Update acceleration
                indices = tuple([i, content_idx] + context_indices)
                
                # Simplified acceleration update based on velocity change
                # Consistent verification in same direction increases acceleration
                acceleration_change = weight.effective_weight * (
                    1 if event.verification_score > 0.5 else -1
                ) * 0.1  # Scale factor
                
                current_acceleration = self.field_tensor.tensor_data[indices]
                new_acceleration = current_acceleration + acceleration_change
                
                # Apply strong damping to acceleration to prevent instability
                new_acceleration *= (1 - self.dynamics_parameters.damping_factor * 2)
                
                # Only apply changes above noise tolerance
                if abs(new_acceleration - current_acceleration) > self.dynamics_parameters.noise_tolerance:
                    self.field_tensor.tensor_data[indices] = new_acceleration
            
            elif dim == TrustFieldDimension.DECAY:
                # Update decay rate
                indices = tuple([i, content_idx] + context_indices)
                current_decay = self.field_tensor.tensor_data[indices]
                
                # Adjust decay based on verification consistency
                # More consistent verification (high weight) might reduce decay
                decay_change = -weight.effective_weight * 0.01  # Small adjustment
                new_decay = current_decay + decay_change
                
                # Ensure decay stays in reasonable range
                new_decay = max(0.01, min(0.5, new_decay))
                
                # Only apply changes above noise tolerance
                if abs(new_decay - current_decay) > self.dynamics_parameters.noise_tolerance:
                    self.field_tensor.tensor_data[indices] = new_decay
            
            elif dim == TrustFieldDimension.AMPLIFICATION:
                # Update amplification factor
                indices = tuple([i, content_idx] + context_indices)
                current_amplification = self.field_tensor.tensor_data[indices]
                
                # Adjust amplification based on verification
                # Strong verification might increase amplification
                amp_change = weight.effective_weight * amplification_factor * 0.01
                new_amplification = current_amplification + amp_change
                
                # Ensure amplification stays in reasonable range
                new_amplification = max(0.05, min(0.5, new_amplification))
                
                # Only apply changes above noise tolerance
                if abs(new_amplification - current_amplification) > self.dynamics_parameters.noise_tolerance:
                    self.field_tensor.tensor_data[indices] = new_amplification
        
        # Apply interference effects between contexts if enabled
        if self.dynamics_parameters.interference_strength > 0:
            self._apply_interference_effects(content_idx, context_indices, weight.effective_weight)
        
        # Update the timestamp on the tensor
        self.field_tensor.timestamp = current_time
        
        # Take a snapshot if this is a significant event
        if weight.effective_weight > 0.5:  # Threshold for "significant"
            self.take_snapshot(event.event_id)
    
    def _apply_interference_effects(self, content_idx: int, context_indices: List[int], weight: float):
        """
        Apply interference effects between contexts in the trust field.
        
        Args:
            content_idx: Index of the content type
            context_indices: Indices of the contexts being updated
            weight: Effective weight of the verification
        """
        # This is a simplified implementation of interference
        # A more sophisticated version would model wave-like interference patterns
        
        # Only apply interference to value dimension for simplicity
        value_idx = self.field_tensor.dimensions.index(TrustFieldDimension.VALUE)
        
        # Calculate interference strength
        interference = self.dynamics_parameters.interference_strength * weight
        
        # Apply interference to nearby contexts
        for i in range(len(self.field_tensor.context_keys)):
            # Skip if this context key has only one value
            if len(self.field_tensor.context_values[self.field_tensor.context_keys[i]]) <= 1:
                continue
            
            # Current context index for this key
            current_ctx_idx = context_indices[i]
            
            # Apply interference to adjacent context values (simplification)
            # In a real implementation, interference could depend on semantic similarity
            for offset in [-1, 1]:  # Check neighboring contexts
                neighbor_idx = current_ctx_idx + offset
                
                # Skip if out of bounds
                if neighbor_idx < 0 or neighbor_idx >= len(self.field_tensor.context_values[self.field_tensor.context_keys[i]]):
                    continue
                
                # Create indices for this neighbor
                neighbor_indices = context_indices.copy()
                neighbor_indices[i] = neighbor_idx
                
                # Apply interference effect
                indices = tuple([value_idx, content_idx] + neighbor_indices)
                current_value = self.field_tensor.tensor_data[indices]
                
                # Simplified interference: neighboring contexts are affected slightly
                # Could be positive or negative depending on the relationship
                interference_effect = interference * (0.5 - np.random.random())  # Random direction
                
                # Apply interference effect
                new_value = current_value + interference_effect
                
                # Ensure value stays in valid range [0, 1]
                new_value = max(0.0, min(1.0, new_value))
                
                # Only apply changes above noise tolerance
                if abs(new_value - current_value) > self.dynamics_parameters.noise_tolerance:
                    self.field_tensor.tensor_data[indices] = new_value
    
    def apply_batch_verification_events(self, 
                                      events: List[VerificationEvent], 
                                      current_time: datetime = None,
                                      current_context: Dict[str, str] = None):
        """
        Apply a batch of verification events to update the trust field.
        
        Args:
            events: List of verification events
            current_time: Current time (defaults to now)
            current_context: Current context (for context similarity)
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Sort events by timestamp (oldest first)
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Process events up to the configured maximum
        max_events = self.configuration.max_events_per_update
        if len(sorted_events) > max_events:
            logger.warning(f"Limiting batch to {max_events} events (from {len(sorted_events)})")
            sorted_events = sorted_events[:max_events]
        
        # Apply each event
        for event in sorted_events:
            self.apply_verification_event(
                event=event,
                current_time=current_time,
                current_context=current_context
            )
    
    def simulate_time_evolution(self, 
                              time_delta: timedelta, 
                              current_time: datetime = None) -> TrustFieldTensor:
        """
        Simulate the natural evolution of the trust field over time without new events.
        
        This models how trust naturally decays, velocities dampen, etc. when no new
        verification events occur.
        
        Args:
            time_delta: Amount of time to simulate
            current_time: Current time (defaults to now)
            
        Returns:
            Updated trust field tensor
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Time since last update
        time_since_update = (current_time - self.last_update_time).total_seconds()
        
        # Skip if no significant time has passed
        if time_since_update < 1.0:  # Less than 1 second
            return self.field_tensor
        
        # Create a copy of the tensor to avoid modifying the original during calculations
        tensor_data = np.copy(self.field_tensor.tensor_data)
        
        # Get indices for each dimension
        value_idx = self.field_tensor.dimensions.index(TrustFieldDimension.VALUE)
        velocity_idx = self.field_tensor.dimensions.index(TrustFieldDimension.VELOCITY)
        accel_idx = self.field_tensor.dimensions.index(TrustFieldDimension.ACCELERATION)
        decay_idx = self.field_tensor.dimensions.index(TrustFieldDimension.DECAY)
        amplification_idx = self.field_tensor.dimensions.index(TrustFieldDimension.AMPLIFICATION)
        
        # Scale factor based on time delta (convert to days for alignment with other rates)
        time_factor = time_delta.total_seconds() / (24 * 60 * 60)  # Convert to days
        
        # Update each content type and context combination
        for content_idx in range(len(self.field_tensor.content_types)):
            # Create a list of slices for iteration
            # This will allow us to iterate over all context combinations
            context_shapes = [len(self.field_tensor.context_values[key]) for key in self.field_tensor.context_keys]
            context_indices = [range(shape) for shape in context_shapes]
            
            # Use numpy's broadcasting to update all contexts efficiently
            # For clarity, we'll use explicit loops, but this could be optimized
            
            # If no contexts, create a dummy iteration with empty list
            if not context_shapes:
                context_combinations = [[]]
            else:
                # Generate all combinations of context indices
                import itertools
                context_combinations = list(itertools.product(*context_indices))
            
            for context_idx_tuple in context_combinations:
                # Convert to list for easier manipulation
                context_idx_list = list(context_idx_tuple)
                
                # Create index tuples for each dimension
                value_indices = tuple([value_idx, content_idx] + context_idx_list)
                velocity_indices = tuple([velocity_idx, content_idx] + context_idx_list)
                accel_indices = tuple([accel_idx, content_idx] + context_idx_list)
                decay_indices = tuple([decay_idx, content_idx] + context_idx_list)
                amplification_indices = tuple([amplification_idx, content_idx] + context_idx_list)
                
                # Get current values
                current_value = tensor_data[value_indices]
                current_velocity = tensor_data[velocity_indices]
                current_accel = tensor_data[accel_indices]
                current_decay = tensor_data[decay_indices]
                current_amplification = tensor_data[amplification_indices]
                
                # Update acceleration (damping effect)
                new_accel = current_accel * (1 - self.dynamics_parameters.damping_factor * time_factor)
                
                # Update velocity based on acceleration and damping
                velocity_change = current_accel * time_factor
                dampened_velocity = current_velocity * (1 - self.dynamics_parameters.damping_factor * time_factor)
                new_velocity = dampened_velocity + velocity_change
                
                # Update value based on velocity
                value_change = new_velocity * time_factor
                
                # Apply natural decay
                decay_effect = -current_decay * time_factor * current_value
                
                # Combine effects to get new value
                new_value = current_value + value_change + decay_effect
                
                # Ensure value stays in range [0, 1]
                new_value = max(0.0, min(1.0, new_value))
                
                # Update tensor data
                tensor_data[value_indices] = new_value
                tensor_data[velocity_indices] = new_velocity
                tensor_data[accel_indices] = new_accel
                
                # Decay and amplification naturally tend toward equilibrium values
                equilibrium_decay = 0.1  # Default equilibrium decay rate
                tensor_data[decay_indices] = current_decay + (equilibrium_decay - current_decay) * 0.01 * time_factor
                
                equilibrium_amplification = 0.2  # Default equilibrium amplification factor
                tensor_data[amplification_indices] = (current_amplification + 
                                                  (equilibrium_amplification - current_amplification) * 0.01 * time_factor)
        
        # Create updated field tensor
        updated_field_tensor = TrustFieldTensor(
            tensor_data=tensor_data,
            dimensions=self.field_tensor.dimensions,
            content_types=self.field_tensor.content_types,
            context_keys=self.field_tensor.context_keys,
            context_values=self.field_tensor.context_values,
            timestamp=current_time
        )
        
        # Update internal state
        self.field_tensor = updated_field_tensor
        self.last_update_time = current_time
        
        # Take a snapshot if significant time has passed
        if time_factor > 1.0:  # More than a day
            self.take_snapshot("time_evolution")
        
        return self.field_tensor
    
    def get_trust_field_at_time(self, target_time: datetime) -> TrustFieldTensor:
        """
        Get the state of the trust field at a specific point in time.
        
        If the target time is in the future, the field is simulated forward.
        If it's in the past, the closest historical snapshot is returned.
        
        Args:
            target_time: The time to get the trust field for
            
        Returns:
            Trust field tensor at the specified time
        """
        current_time = datetime.now()
        
        # Case 1: Target time is now or in the future
        if target_time >= current_time:
            # Simulate forward
            time_delta = target_time - current_time
            return self.simulate_time_evolution(time_delta, target_time)
        
        # Case 2: Target time is in the past
        else:
            # Find the closest snapshot
            if not self.snapshots:
                # No snapshots available, return current field
                logger.warning(f"No historical snapshots available for time {target_time}")
                return self.field_tensor
            
            # Find snapshot closest to target time
            closest_snapshot = min(self.snapshots, key=lambda s: abs((s.timestamp - target_time).total_seconds()))
            
            # Log warning if closest snapshot is far from target time
            time_diff = abs((closest_snapshot.timestamp - target_time).total_seconds())
            if time_diff > 3600:  # More than an hour
                logger.warning(f"Closest snapshot is {time_diff/3600:.2f} hours away from target time")
            
            return closest_snapshot.tensor
    
    def calculate_field_stability(self, 
                                time_window: Union[str, timedelta] = "1d") -> float:
        """
        Calculate the stability of the trust field over a time window.
        
        Args:
            time_window: Time window to consider, either a timedelta or a string like "1d", "12h"
            
        Returns:
            Stability score from 0.0 (unstable) to 1.0 (completely stable)
        """
        # Convert string time window to timedelta if needed
        if isinstance(time_window, str):
            if time_window.endswith('d'):
                days = float(time_window[:-1])
                time_window = timedelta(days=days)
            elif time_window.endswith('h'):
                hours = float(time_window[:-1])
                time_window = timedelta(hours=hours)
            elif time_window.endswith('m'):
                minutes = float(time_window[:-1])
                time_window = timedelta(minutes=minutes)
            else:
                raise ValueError(f"Unrecognized time window format: {time_window}")
        
        # Get current time
        current_time = datetime.now()
        
        # Get start time
        start_time = current_time - time_window
        
        # Filter snapshots within the time window
        relevant_snapshots = [s for s in self.snapshots if s.timestamp >= start_time]
        
        # If no snapshots in window, return current stability
        if not relevant_snapshots:
            return self.field_tensor.get_field_stability()
        
        # Calculate stability from snapshots
        stability_values = []
        
        # Add current field stability
        stability_values.append(self.field_tensor.get_field_stability())
        
        # Add stability from each snapshot
        for snapshot in relevant_snapshots:
            if snapshot.stability_score is not None:
                stability_values.append(snapshot.stability_score)
            else:
                stability_values.append(snapshot.tensor.get_field_stability())
        
        # Calculate mean stability
        mean_stability = sum(stability_values) / len(stability_values)
        
        # Calculate standard deviation (measure of stability fluctuation)
        if len(stability_values) > 1:
            std_dev = np.std(stability_values)
            # Adjust mean stability based on fluctuation
            # Higher fluctuation reduces effective stability
            mean_stability = mean_stability * (1 - std_dev)
        
        # Ensure result is in range [0, 1]
        return max(0.0, min(1.0, mean_stability))
    
    def analyze_trust_field(self, 
                         start_time: datetime = None,
                         end_time: datetime = None) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of the trust field over time.
        
        Args:
            start_time: Start of analysis period (defaults to earliest snapshot)
            end_time: End of analysis period (defaults to now)
            
        Returns:
            Dictionary containing analysis results
        """
        current_time = datetime.now()
        
        # Set default time range if not specified
        if end_time is None:
            end_time = current_time
        
        if start_time is None:
            if self.snapshots:
                start_time = min(s.timestamp for s in self.snapshots)
            else:
                # Default to 30 days ago if no snapshots
                start_time = current_time - timedelta(days=30)
        
        # Ensure start_time is before end_time
        if start_time >= end_time:
            raise ValueError(f"Start time ({start_time}) must be before end time ({end_time})")
        
        # Filter snapshots within the time range
        relevant_snapshots = [s for s in self.snapshots if start_time <= s.timestamp <= end_time]
        
        # Add current field as the latest point
        field_states = [(current_time, self.field_tensor)]
        field_states.extend([(s.timestamp, s.tensor) for s in relevant_snapshots])
        
        # Sort by timestamp
        field_states.sort(key=lambda x: x[0])
        
        # Extract stability values
        stability_values = [(ts, tensor.get_field_stability()) for ts, tensor in field_states]
        
        # Calculate average stability
        if stability_values:
            average_stability = sum(s for _, s in stability_values) / len(stability_values)
        else:
            average_stability = self.field_tensor.get_field_stability()
        
        # Calculate stability trend (simple linear regression)
        stability_trend = 0.0
        if len(stability_values) > 1:
            # Convert timestamps to seconds since epoch
            x = [(ts - start_time).total_seconds() for ts, _ in stability_values]
            y = [s for _, s in stability_values]
            
            # Normalize x to [0, 1] for numerical stability
            if max(x) > min(x):
                x = [(xi - min(x)) / (max(x) - min(x)) for xi in x]
            
            # Simple linear regression to get slope
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)
            
            # Calculate slope (avoid division by zero)
            if n * sum_x2 - sum_x * sum_x != 0:
                stability_trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Analyze content type trends
        content_type_trends = {}
        for i, content_type in enumerate(self.field_tensor.content_types):
            # Get stability per content type over time
            content_stability = []
            for _, tensor in field_states:
                # Average stability across all contexts
                dim_idx = tensor.dimensions.index(TrustFieldDimension.VALUE)
                velocity_idx = tensor.dimensions.index(TrustFieldDimension.VELOCITY)
                
                # Extract slices for this content type
                value_slice = tensor.tensor_data[dim_idx, i, ...]
                velocity_slice = tensor.tensor_data[velocity_idx, i, ...]
                
                # Calculate stability for this content type
                avg_value = np.mean(value_slice)
                avg_velocity_magnitude = np.mean(np.abs(velocity_slice))
                
                # Stability is inverse of velocity magnitude
                stability = 1.0 - avg_velocity_magnitude
                content_stability.append(stability)
            
            # Calculate average and trend
            if content_stability:
                avg_stability = sum(content_stability) / len(content_stability)
                
                # Calculate trend
                trend = 0.0
                if len(content_stability) > 1:
                    # Simple linear regression as above
                    x = list(range(len(content_stability)))
                    y = content_stability
                    
                    n = len(x)
                    sum_x = sum(x)
                    sum_y = sum(y)
                    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
                    sum_x2 = sum(xi * xi for xi in x)
                    
                    if n * sum_x2 - sum_x * sum_x != 0:
                        trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                content_type_trends[content_type.value] = {
                    "stability": avg_stability,
                    "trend": trend
                }
        
        # Context-specific insights
        # For simplicity, just analyze overall stability per context
        context_specific_insights = {}
        for i, key in enumerate(self.field_tensor.context_keys):
            context_insights = {}
            for j, value in enumerate(self.field_tensor.context_values[key]):
                # Calculate average stability for this context value
                context_stability = []
                for _, tensor in field_states:
                    # Create a slice for this context
                    dim_idx = tensor.dimensions.index(TrustFieldDimension.VALUE)
                    
                    # Create context indices
                    ctx_indices = [slice(None)] * len(tensor.context_keys)  # All values
                    ctx_indices[i] = j  # Specific value for this context key
                    
                    # Extract slice for all content types, this context value
                    value_slice = tensor.tensor_data[dim_idx, :, *ctx_indices]
                    
                    # Average stability
                    context_stability.append(np.mean(value_slice))
                
                if context_stability:
                    avg_stability = sum(context_stability) / len(context_stability)
                    context_insights[value] = {
                        "stability": avg_stability
                    }
            
            context_specific_insights[key] = context_insights
        
        # Identify critical events
        # For simplicity, find events with largest impact on stability
        critical_events = []
        if len(relevant_snapshots) > 1:
            # Calculate stability changes between snapshots
            stability_changes = []
            for i in range(1, len(relevant_snapshots)):
                prev_stability = relevant_snapshots[i-1].stability_score or relevant_snapshots[i-1].tensor.get_field_stability()
                curr_stability = relevant_snapshots[i].stability_score or relevant_snapshots[i].tensor.get_field_stability()
                change = curr_stability - prev_stability
                
                if relevant_snapshots[i].triggered_by_event:
                    stability_changes.append((abs(change), relevant_snapshots[i].triggered_by_event))
            
            # Sort by absolute change (largest first)
            stability_changes.sort(reverse=True)
            
            # Get top events
            critical_events = [event_id for _, event_id in stability_changes[:5] if event_id != "time_evolution"]
        
        # Detect anomalies
        # For simplicity, look for unusual stability changes
        anomalies = []
        if len(stability_values) > 2:
            # Calculate typical stability change
            stability_diffs = []
            for i in range(1, len(stability_values)):
                diff = stability_values[i][1] - stability_values[i-1][1]
                stability_diffs.append(diff)
            
            mean_diff = sum(stability_diffs) / len(stability_diffs)
            std_diff = np.std(stability_diffs) if len(stability_diffs) > 1 else 0.1
            
            # Look for changes more than 2 standard deviations from mean
            for i in range(1, len(stability_values)):
                ts, stab = stability_values[i]
                prev_stab = stability_values[i-1][1]
                diff = stab - prev_stab
                
                if abs(diff - mean_diff) > 2 * std_diff:
                    anomalies.append({
                        "timestamp": ts.isoformat(),
                        "stability_change": diff,
                        "description": "Unusual stability change",
                        "severity": "high" if abs(diff - mean_diff) > 3 * std_diff else "medium"
                    })
        
        # Generate stability recommendations
        stability_recommendations = []
        
        # Check overall stability
        if average_stability < self.configuration.low_stability_threshold:
            stability_recommendations.append({
                "type": "overall_stability",
                "severity": "high",
                "description": "Trust field has low overall stability",
                "suggestion": "Consider reducing the number of frequent contradictory verification events"
            })
        
        # Check for negative trends
        if stability_trend < -0.1:
            stability_recommendations.append({
                "type": "stability_trend",
                "severity": "medium",
                "description": "Trust field stability is decreasing over time",
                "suggestion": "Increase verification frequency or consistency to stabilize"
            })
        
        # Check content type specific issues
        for content_type, trends in content_type_trends.items():
            if trends["stability"] < self.configuration.low_stability_threshold:
                stability_recommendations.append({
                    "type": "content_type_stability",
                    "content_type": content_type,
                    "severity": "medium",
                    "description": f"Low stability for {content_type} content",
                    "suggestion": f"Improve verification consistency for {content_type} content"
                })
        
        # Prepare visualization data (simplified)
        visualization_data = {
            "stability_time_series": [(ts.isoformat(), stab) for ts, stab in stability_values],
            "content_type_trends": content_type_trends,
            "anomalies": anomalies
        }
        
        # Construct and return the full analysis result
        return {
            "field_id": id(self.field_tensor),  # Use object ID as field ID
            "analysis_timestamp": current_time.isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "average_stability": average_stability,
            "stability_trend": stability_trend,
            "content_type_trends": content_type_trends,
            "context_specific_insights": context_specific_insights,
            "critical_events": critical_events,
            "anomalies": anomalies,
            "stability_recommendations": stability_recommendations,
            "visualization_data": visualization_data
        }


# Factory function to create a field dynamics engine with default configuration
def create_default_field_dynamics_engine() -> FieldDynamicsEngine:
    """
    Create a field dynamics engine with default configuration.
    
    Returns:
        A FieldDynamicsEngine with sensible defaults
    """
    config = create_default_trust_field_configuration()
    return FieldDynamicsEngine(config)
