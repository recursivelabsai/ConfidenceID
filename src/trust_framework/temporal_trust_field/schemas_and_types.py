"""
Data structures and type definitions for the Temporal Trust Field module.

This module defines the core data structures used to represent trust as a
dynamic, evolving field rather than a static value. These structures capture
the multidimensional nature of trust across time, including velocity, acceleration,
decay constants, and amplification factors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
from enum import Enum


class VerificationEventType(str, Enum):
    """Types of verification events that can influence the trust field."""
    WATERMARK_DETECTION = "watermark_detection"
    ARTIFACT_DETECTION = "artifact_detection"
    SEMANTIC_COHERENCE = "semantic_coherence"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    MANUAL_VERIFICATION = "manual_verification"
    CROSS_MODAL_VERIFICATION = "cross_modal_verification"
    CONSENSUS_VERIFICATION = "consensus_verification"
    ARCHAEOLOGICAL_PATTERN = "archaeological_pattern"


class ContentType(str, Enum):
    """Content types that can be verified."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class TrustFieldDimension(str, Enum):
    """Dimensions of the trust field tensor."""
    VALUE = "value"                  # The trust value itself
    VELOCITY = "velocity"            # Rate of change in trust
    ACCELERATION = "acceleration"    # Change in rate of change
    DECAY = "decay"                  # Decay constant for trust signals
    AMPLIFICATION = "amplification"  # Amplification factor for reinforcement


@dataclass
class VerificationEvent:
    """
    Represents a single verification event that influences the trust field.
    
    Each verification event carries information about what was verified,
    the verification result, when it occurred, and contextual information.
    """
    event_id: str
    event_type: VerificationEventType
    content_type: ContentType
    verification_score: float  # 0.0 to 1.0
    timestamp: datetime
    context: Dict[str, any] = field(default_factory=dict)
    source: Optional[str] = None
    confidence: float = 1.0  # Confidence in the verification result
    related_events: List[str] = field(default_factory=list)  # IDs of related events
    metadata: Dict[str, any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "content_type": self.content_type.value,
            "verification_score": self.verification_score,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "source": self.source,
            "confidence": self.confidence,
            "related_events": self.related_events,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationEvent':
        """Create from dictionary after deserialization."""
        return cls(
            event_id=data["event_id"],
            event_type=VerificationEventType(data["event_type"]),
            content_type=ContentType(data["content_type"]),
            verification_score=data["verification_score"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data.get("context", {}),
            source=data.get("source"),
            confidence=data.get("confidence", 1.0),
            related_events=data.get("related_events", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class TemporalWeight:
    """
    Represents the temporal impact of a verification event.
    
    This quantifies how strongly an event influences the trust field
    based on recency, relevance, and other temporal factors.
    """
    base_weight: float  # Base weight of the event
    temporal_decay: float  # Decay factor based on time since event
    context_similarity: float  # Similarity to current context (0.0 to 1.0)
    confidence_factor: float  # Adjustment based on verification confidence
    
    @property
    def effective_weight(self) -> float:
        """Calculate the effective weight of the event."""
        return self.base_weight * self.temporal_decay * self.context_similarity * self.confidence_factor


@dataclass
class TrustFieldTensor:
    """
    Multi-dimensional tensor representing the trust field.
    
    This complex structure captures the full state of trust as a field,
    with multiple dimensions for different aspects of trust dynamics.
    """
    # Core tensor data (dimensions vary by implementation)
    # Shape: [dimensions, content_types, contexts...]
    tensor_data: np.ndarray
    
    # Dimensional information
    dimensions: List[TrustFieldDimension]
    content_types: List[ContentType]
    context_keys: List[str]  # Context dimensions (e.g., "domain", "audience")
    context_values: Dict[str, List[str]]  # Possible values for each context key
    
    # Metadata
    timestamp: datetime  # When this tensor state was calculated
    version: str = "1.0.0"
    
    def get_trust_value(self, 
                      content_type: ContentType, 
                      context: Dict[str, str] = None) -> float:
        """
        Get the current trust value for a specific content type and context.
        
        Args:
            content_type: The content type to get trust for
            context: Optional context dictionary
            
        Returns:
            The trust value (0.0 to 1.0)
        """
        # Implementation would extract the appropriate value from tensor_data
        # This is a simplified placeholder
        dim_idx = self.dimensions.index(TrustFieldDimension.VALUE)
        content_idx = self.content_types.index(content_type)
        
        # If no context provided, average across all contexts
        if not context:
            # Average across all context dimensions
            sliced = self.tensor_data[dim_idx, content_idx, ...]
            return float(np.mean(sliced))
        
        # Otherwise, locate the specific context indices
        context_indices = []
        for i, key in enumerate(self.context_keys):
            if key in context:
                value_idx = self.context_values[key].index(context[key])
                context_indices.append(value_idx)
            else:
                # If context key not provided, average across all values
                context_indices.append(slice(None))
        
        # Extract the value
        indices = tuple([dim_idx, content_idx] + context_indices)
        return float(self.tensor_data[indices])
    
    def get_trust_velocity(self,
                          content_type: ContentType,
                          context: Dict[str, str] = None) -> float:
        """
        Get the current rate of change in trust for a specific content type and context.
        
        Args:
            content_type: The content type to get trust velocity for
            context: Optional context dictionary
            
        Returns:
            The trust velocity (can be positive or negative)
        """
        # Similar implementation to get_trust_value but for velocity dimension
        dim_idx = self.dimensions.index(TrustFieldDimension.VELOCITY)
        content_idx = self.content_types.index(content_type)
        
        # If no context provided, average across all contexts
        if not context:
            # Average across all context dimensions
            sliced = self.tensor_data[dim_idx, content_idx, ...]
            return float(np.mean(sliced))
        
        # Otherwise, locate the specific context indices
        context_indices = []
        for i, key in enumerate(self.context_keys):
            if key in context:
                value_idx = self.context_values[key].index(context[key])
                context_indices.append(value_idx)
            else:
                # If context key not provided, average across all values
                context_indices.append(slice(None))
        
        # Extract the velocity
        indices = tuple([dim_idx, content_idx] + context_indices)
        return float(self.tensor_data[indices])
    
    def get_field_stability(self) -> float:
        """
        Calculate the overall stability of the trust field.
        
        Returns:
            Stability score from 0.0 (completely unstable) to 1.0 (perfectly stable)
        """
        # A stable field has low velocity and acceleration
        # This is a simplified placeholder implementation
        velocity_idx = self.dimensions.index(TrustFieldDimension.VELOCITY)
        accel_idx = self.dimensions.index(TrustFieldDimension.ACCELERATION)
        
        velocity_magnitude = np.mean(np.abs(self.tensor_data[velocity_idx, ...]))
        accel_magnitude = np.mean(np.abs(self.tensor_data[accel_idx, ...]))
        
        # Convert to stability score (inversely related to magnitudes)
        stability = 1.0 - (velocity_magnitude + accel_magnitude) / 2.0
        return max(0.0, min(1.0, stability))  # Clamp to [0.0, 1.0]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "tensor_data": self.tensor_data.tolist(),
            "dimensions": [dim.value for dim in self.dimensions],
            "content_types": [ct.value for ct in self.content_types],
            "context_keys": self.context_keys,
            "context_values": self.context_values,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrustFieldTensor':
        """Create from dictionary after deserialization."""
        return cls(
            tensor_data=np.array(data["tensor_data"]),
            dimensions=[TrustFieldDimension(dim) for dim in data["dimensions"]],
            content_types=[ContentType(ct) for ct in data["content_types"]],
            context_keys=data["context_keys"],
            context_values=data["context_values"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", "1.0.0")
        )


@dataclass
class TrustFieldSnapshot:
    """
    A snapshot of the trust field at a specific point in time.
    
    Used for historical tracking and temporal analysis.
    """
    tensor: TrustFieldTensor
    timestamp: datetime
    triggered_by_event: Optional[str] = None  # Event ID that triggered this snapshot
    stability_score: float = None  # Calculated stability at this point
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate stability if not provided."""
        if self.stability_score is None:
            self.stability_score = self.tensor.get_field_stability()


@dataclass
class TrustFieldAnalysisResult:
    """
    Results from analyzing a trust field over time.
    
    Contains insights about field stability, trends, and patterns.
    """
    field_id: str
    analysis_timestamp: datetime
    start_time: datetime
    end_time: datetime
    
    # Overall metrics
    average_stability: float
    stability_trend: float  # Positive means increasing stability
    
    # Detailed analysis
    content_type_trends: Dict[str, Dict[str, float]]  # For each content type: stability, trend
    context_specific_insights: Dict[str, any]  # Context-specific observations
    critical_events: List[str]  # IDs of events with significant impact
    anomalies: List[Dict[str, any]]  # Detected anomalies in the field
    
    # Recommendations
    stability_recommendations: List[Dict[str, any]]  # Recommendations to improve stability
    
    # Visualization data (for rendering)
    visualization_data: Dict[str, any] = field(default_factory=dict)


@dataclass
class FieldDynamicsParameters:
    """
    Parameters that govern the evolution of the trust field over time.
    
    These parameters control how trust signals decay, amplify, and propagate
    through the field, and can be tuned for different content types and contexts.
    """
    # Base decay rates for different verification types
    decay_rates: Dict[VerificationEventType, float]
    
    # Amplification factors for different verification types
    amplification_factors: Dict[VerificationEventType, float]
    
    # Content-specific modifiers
    content_type_modifiers: Dict[ContentType, Dict[str, float]]
    
    # Context-specific modifiers
    context_modifiers: Dict[str, Dict[str, float]]
    
    # Interference parameters (how signals interact)
    interference_strength: float = 0.5  # 0.0 = no interference, 1.0 = strong interference
    
    # Temporal window parameters
    max_history_window: int = 30  # Days of history to consider
    recency_weight_factor: float = 0.1  # How much to weight recent events
    
    # Field stability parameters
    damping_factor: float = 0.2  # Dampens oscillations in the field
    noise_tolerance: float = 0.05  # Small fluctuations below this are ignored
    
    def get_decay_rate(self, 
                      event_type: VerificationEventType, 
                      content_type: ContentType,
                      context: Dict[str, str] = None) -> float:
        """
        Calculate the content and context-specific decay rate for a verification type.
        
        Args:
            event_type: Type of verification event
            content_type: Type of content being verified
            context: Optional context dictionary
            
        Returns:
            The adjusted decay rate
        """
        # Start with base rate for the event type
        base_rate = self.decay_rates.get(event_type, 0.1)  # Default if not specified
        
        # Apply content type modifier
        content_modifier = self.content_type_modifiers.get(content_type, {}).get("decay_modifier", 1.0)
        
        # Apply context modifiers if provided
        context_modifier = 1.0
        if context:
            for key, value in context.items():
                if key in self.context_modifiers and value in self.context_modifiers[key]:
                    context_modifier *= self.context_modifiers[key][value].get("decay_modifier", 1.0)
        
        # Calculate final decay rate
        return base_rate * content_modifier * context_modifier
    
    def get_amplification_factor(self, 
                                event_type: VerificationEventType, 
                                content_type: ContentType,
                                context: Dict[str, str] = None) -> float:
        """
        Calculate the content and context-specific amplification factor for a verification type.
        
        Args:
            event_type: Type of verification event
            content_type: Type of content being verified
            context: Optional context dictionary
            
        Returns:
            The adjusted amplification factor
        """
        # Similar implementation to get_decay_rate but for amplification
        base_factor = self.amplification_factors.get(event_type, 0.2)  # Default if not specified
        
        # Apply content type modifier
        content_modifier = self.content_type_modifiers.get(content_type, {}).get("amplification_modifier", 1.0)
        
        # Apply context modifiers if provided
        context_modifier = 1.0
        if context:
            for key, value in context.items():
                if key in self.context_modifiers and value in self.context_modifiers[key]:
                    context_modifier *= self.context_modifiers[key][value].get("amplification_modifier", 1.0)
        
        # Calculate final amplification factor
        return base_factor * content_modifier * context_modifier
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "decay_rates": {k.value: v for k, v in self.decay_rates.items()},
            "amplification_factors": {k.value: v for k, v in self.amplification_factors.items()},
            "content_type_modifiers": {k.value: v for k, v in self.content_type_modifiers.items()},
            "context_modifiers": self.context_modifiers,
            "interference_strength": self.interference_strength,
            "max_history_window": self.max_history_window,
            "recency_weight_factor": self.recency_weight_factor,
            "damping_factor": self.damping_factor,
            "noise_tolerance": self.noise_tolerance
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FieldDynamicsParameters':
        """Create from dictionary after deserialization."""
        # Convert string keys back to enums for the dictionaries that use enums as keys
        decay_rates = {VerificationEventType(k): v for k, v in data["decay_rates"].items()}
        amplification_factors = {VerificationEventType(k): v for k, v in data["amplification_factors"].items()}
        content_type_modifiers = {ContentType(k): v for k, v in data["content_type_modifiers"].items()}
        
        return cls(
            decay_rates=decay_rates,
            amplification_factors=amplification_factors,
            content_type_modifiers=content_type_modifiers,
            context_modifiers=data["context_modifiers"],
            interference_strength=data.get("interference_strength", 0.5),
            max_history_window=data.get("max_history_window", 30),
            recency_weight_factor=data.get("recency_weight_factor", 0.1),
            damping_factor=data.get("damping_factor", 0.2),
            noise_tolerance=data.get("noise_tolerance", 0.05)
        )


@dataclass
class TrustFieldConfiguration:
    """
    Configuration for initializing and updating a trust field.
    
    This defines the dimensions, content types, contexts, and dynamic parameters
    that govern the behavior of a trust field.
    """
    # Dimensional configuration
    dimensions: List[TrustFieldDimension]
    content_types: List[ContentType]
    context_keys: List[str]
    context_values: Dict[str, List[str]]
    
    # Dynamic parameters
    dynamics_parameters: FieldDynamicsParameters
    
    # Initialization parameters
    initial_trust_value: float = 0.5  # Default trust value for new fields
    initial_velocity: float = 0.0  # Default velocity (no change)
    initial_acceleration: float = 0.0  # Default acceleration (no change in velocity)
    initial_decay: float = 0.1  # Default decay rate
    initial_amplification: float = 0.2  # Default amplification factor
    
    # Event processing configuration
    min_event_weight: float = 0.01  # Events with weight below this are ignored
    max_events_per_update: int = 100  # Maximum events to process in one update
    
    # Stability thresholds
    high_stability_threshold: float = 0.8  # Above this is considered highly stable
    low_stability_threshold: float = 0.3  # Below this is considered unstable
    critical_stability_threshold: float = 0.1  # Below this requires intervention
    
    def get_tensor_shape(self) -> Tuple[int, ...]:
        """
        Calculate the shape of the trust field tensor based on configuration.
        
        Returns:
            A tuple representing the tensor shape
        """
        # Shape: [dimensions, content_types, context_values...]
        shape = [len(self.dimensions), len(self.content_types)]
        
        # Add dimensions for each context key
        for key in self.context_keys:
            shape.append(len(self.context_values[key]))
        
        return tuple(shape)
    
    def create_initial_tensor(self) -> np.ndarray:
        """
        Create an initial tensor with default values based on configuration.
        
        Returns:
            A numpy array initialized with default values
        """
        shape = self.get_tensor_shape()
        tensor = np.zeros(shape)
        
        # Set initial values for each dimension
        dim_values = {
            TrustFieldDimension.VALUE: self.initial_trust_value,
            TrustFieldDimension.VELOCITY: self.initial_velocity,
            TrustFieldDimension.ACCELERATION: self.initial_acceleration,
            TrustFieldDimension.DECAY: self.initial_decay,
            TrustFieldDimension.AMPLIFICATION: self.initial_amplification
        }
        
        # Set initial values for each dimension
        for i, dim in enumerate(self.dimensions):
            tensor[i, ...] = dim_values.get(dim, 0.0)
        
        return tensor
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "dimensions": [dim.value for dim in self.dimensions],
            "content_types": [ct.value for ct in self.content_types],
            "context_keys": self.context_keys,
            "context_values": self.context_values,
            "dynamics_parameters": self.dynamics_parameters.to_dict(),
            "initial_trust_value": self.initial_trust_value,
            "initial_velocity": self.initial_velocity,
            "initial_acceleration": self.initial_acceleration,
            "initial_decay": self.initial_decay,
            "initial_amplification": self.initial_amplification,
            "min_event_weight": self.min_event_weight,
            "max_events_per_update": self.max_events_per_update,
            "high_stability_threshold": self.high_stability_threshold,
            "low_stability_threshold": self.low_stability_threshold,
            "critical_stability_threshold": self.critical_stability_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrustFieldConfiguration':
        """Create from dictionary after deserialization."""
        return cls(
            dimensions=[TrustFieldDimension(dim) for dim in data["dimensions"]],
            content_types=[ContentType(ct) for ct in data["content_types"]],
            context_keys=data["context_keys"],
            context_values=data["context_values"],
            dynamics_parameters=FieldDynamicsParameters.from_dict(data["dynamics_parameters"]),
            initial_trust_value=data.get("initial_trust_value", 0.5),
            initial_velocity=data.get("initial_velocity", 0.0),
            initial_acceleration=data.get("initial_acceleration", 0.0),
            initial_decay=data.get("initial_decay", 0.1),
            initial_amplification=data.get("initial_amplification", 0.2),
            min_event_weight=data.get("min_event_weight", 0.01),
            max_events_per_update=data.get("max_events_per_update", 100),
            high_stability_threshold=data.get("high_stability_threshold", 0.8),
            low_stability_threshold=data.get("low_stability_threshold", 0.3),
            critical_stability_threshold=data.get("critical_stability_threshold", 0.1)
        )


# Factory function to create a default configuration
def create_default_trust_field_configuration() -> TrustFieldConfiguration:
    """
    Create a default configuration for a trust field.
    
    Returns:
        A TrustFieldConfiguration with sensible defaults
    """
    # Default dimensions
    dimensions = [
        TrustFieldDimension.VALUE,
        TrustFieldDimension.VELOCITY,
        TrustFieldDimension.ACCELERATION,
        TrustFieldDimension.DECAY,
        TrustFieldDimension.AMPLIFICATION
    ]
    
    # Default content types
    content_types = [
        ContentType.TEXT,
        ContentType.IMAGE,
        ContentType.AUDIO,
        ContentType.VIDEO,
        ContentType.MULTIMODAL
    ]
    
    # Default context keys and values
    context_keys = ["domain", "audience"]
    context_values = {
        "domain": ["news", "entertainment", "education", "social", "commercial", "scientific"],
        "audience": ["general", "professional", "academic", "specialized"]
    }
    
    # Default decay rates for different verification types
    decay_rates = {
        VerificationEventType.WATERMARK_DETECTION: 0.05,
        VerificationEventType.ARTIFACT_DETECTION: 0.08,
        VerificationEventType.SEMANTIC_COHERENCE: 0.12,
        VerificationEventType.TEMPORAL_CONSISTENCY: 0.10,
        VerificationEventType.MANUAL_VERIFICATION: 0.03,
        VerificationEventType.CROSS_MODAL_VERIFICATION: 0.07,
        VerificationEventType.CONSENSUS_VERIFICATION: 0.04,
        VerificationEventType.ARCHAEOLOGICAL_PATTERN: 0.06
    }
    
    # Default amplification factors for different verification types
    amplification_factors = {
        VerificationEventType.WATERMARK_DETECTION: 0.15,
        VerificationEventType.ARTIFACT_DETECTION: 0.12,
        VerificationEventType.SEMANTIC_COHERENCE: 0.10,
        VerificationEventType.TEMPORAL_CONSISTENCY: 0.13,
        VerificationEventType.MANUAL_VERIFICATION: 0.25,
        VerificationEventType.CROSS_MODAL_VERIFICATION: 0.18,
        VerificationEventType.CONSENSUS_VERIFICATION: 0.20,
        VerificationEventType.ARCHAEOLOGICAL_PATTERN: 0.17
    }
    
    # Content-specific modifiers
    content_type_modifiers = {
        ContentType.TEXT: {
            "decay_modifier": 1.0,
            "amplification_modifier": 1.0
        },
        ContentType.IMAGE: {
            "decay_modifier": 1.2,  # Image trust decays slightly faster
            "amplification_modifier": 0.9
        },
        ContentType.AUDIO: {
            "decay_modifier": 1.1,
            "amplification_modifier": 0.95
        },
        ContentType.VIDEO: {
            "decay_modifier": 1.3,  # Video trust decays fastest
            "amplification_modifier": 0.85
        },
        ContentType.MULTIMODAL: {
            "decay_modifier": 1.15,
            "amplification_modifier": 1.1  # Multimodal verification has stronger reinforcement
        }
    }
    
    # Context-specific modifiers
    context_modifiers = {
        "domain": {
            "news": {
                "decay_modifier": 1.3,  # News trust decays faster
                "amplification_modifier": 0.9
            },
            "entertainment": {
                "decay_modifier": 0.8,  # Entertainment trust decays slower
                "amplification_modifier": 1.1
            },
            "education": {
                "decay_modifier": 0.7,  # Educational content maintains trust longer
                "amplification_modifier": 1.2
            },
            "social": {
                "decay_modifier": 1.4,  # Social content trust decays fastest
                "amplification_modifier": 0.8
            },
            "commercial": {
                "decay_modifier": 1.2,
                "amplification_modifier": 0.9
            },
            "scientific": {
                "decay_modifier": 0.6,  # Scientific content maintains trust longest
                "amplification_modifier": 1.3
            }
        },
        "audience": {
            "general": {
                "decay_modifier": 1.0,
                "amplification_modifier": 1.0
            },
            "professional": {
                "decay_modifier": 0.9,
                "amplification_modifier": 1.1
            },
            "academic": {
                "decay_modifier": 0.8,
                "amplification_modifier": 1.2
            },
            "specialized": {
                "decay_modifier": 0.7,
                "amplification_modifier": 1.3
            }
        }
    }
    
    # Create dynamics parameters
    dynamics_parameters = FieldDynamicsParameters(
        decay_rates=decay_rates,
        amplification_factors=amplification_factors,
        content_type_modifiers=content_type_modifiers,
        context_modifiers=context_modifiers,
        interference_strength=0.5,
        max_history_window=30,
        recency_weight_factor=0.1,
        damping_factor=0.2,
        noise_tolerance=0.05
    )
    
    # Create and return the configuration
    return TrustFieldConfiguration(
        dimensions=dimensions,
        content_types=content_types,
        context_keys=context_keys,
        context_values=context_values,
        dynamics_parameters=dynamics_parameters
    )
