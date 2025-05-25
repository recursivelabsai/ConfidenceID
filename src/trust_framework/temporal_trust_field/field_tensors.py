"""
Temporal Trust Field Tensors

This module defines the core data structures for the Temporal Trust Field component
of ConfidenceID. It includes the TrustFieldTensor which represents trust as a 
multidimensional tensor with properties like velocity, acceleration, and stability,
as well as supporting data structures for verification events.

The implementation is based on the temporal trust field theory described in 
claude.metalayer.txt (Layer 8.1).
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class VerificationEvent:
    """
    Represents a verification event that contributes to the trust field.
    
    Each verification event captures details about a specific verification
    instance, including when it occurred, what was verified, the verification
    result, and contextual information.
    """
    
    # Unique identifier for the verification event
    event_id: str
    
    # When the verification occurred
    timestamp: datetime
    
    # The verification score (0-1)
    verification_score: float
    
    # The type of content being verified (e.g., 'text', 'image', 'audio', 'video', 'cross_modal')
    content_type: str
    
    # Vector representation of the context in which verification occurred
    context: Optional[np.ndarray] = None
    
    # The verification method used (e.g., 'watermark', 'artifact_detection', 'semantic_coherence')
    method: str = "unknown"
    
    # The entity that performed the verification (e.g., node ID in decentralized network)
    verifier: Optional[str] = None
    
    # The content fingerprint (privacy-preserving hash)
    content_fingerprint: Optional[str] = None
    
    # Additional metadata about the verification event
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the verification event after initialization."""
        # Ensure verification_score is in the valid range
        self.verification_score = max(0.0, min(1.0, self.verification_score))
        
        # If timestamp is a string, convert it to datetime
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        
        # If context is a list, convert it to numpy array
        if isinstance(self.context, list):
            self.context = np.array(self.context)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the verification event to a dictionary for storage/transmission."""
        result = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "verification_score": self.verification_score,
            "content_type": self.content_type,
            "method": self.method,
        }
        
        # Add optional fields if they exist
        if self.context is not None:
            result["context"] = self.context.tolist()
        
        if self.verifier is not None:
            result["verifier"] = self.verifier
        
        if self.content_fingerprint is not None:
            result["content_fingerprint"] = self.content_fingerprint
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationEvent':
        """Create a verification event from a dictionary."""
        # Convert context list back to numpy array if it exists
        if "context" in data and isinstance(data["context"], list):
            data["context"] = np.array(data["context"])
        
        # Create the verification event
        return cls(**data)


@dataclass
class TrustFieldTensor:
    """
    Represents the state of a trust field as a multidimensional tensor.
    
    The trust field tensor captures not just a static confidence value, but
    a dynamic representation of trust that includes velocity, acceleration,
    decay rate, and other properties that govern how trust evolves over time.
    """
    
    # The current confidence value (0-1)
    confidence: float
    
    # Rate of change in confidence (can be positive or negative)
    velocity: float
    
    # Rate of change in velocity (can be positive or negative)
    acceleration: float
    
    # Rate at which trust decays over time
    decay_rate: float
    
    # Factor by which trust is amplified based on consistent verification
    amplification: float
    
    # Measure of how stable the trust field is (0-1)
    stability: float
    
    # When the trust field was last updated
    last_updated: datetime
    
    # Directional vector of trust (optional)
    direction: Optional[np.ndarray] = None
    
    # Domain-specific trust components (optional)
    components: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the trust field tensor after initialization."""
        # Ensure confidence is in the valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure stability is in the valid range
        self.stability = max(0.0, min(1.0, self.stability))
        
        # Ensure amplification is positive
        self.amplification = max(1.0, self.amplification)
        
        # Ensure decay_rate is positive
        self.decay_rate = max(0.0, self.decay_rate)
        
        # If last_updated is a string, convert it to datetime
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)
        
        # If direction is a list, convert it to numpy array
        if isinstance(self.direction, list):
            self.direction = np.array(self.direction)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the trust field tensor to a dictionary for storage/transmission."""
        result = {
            "confidence": self.confidence,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "decay_rate": self.decay_rate,
            "amplification": self.amplification,
            "stability": self.stability,
            "last_updated": self.last_updated.isoformat(),
        }
        
        # Add optional fields if they exist
        if self.direction is not None:
            result["direction"] = self.direction.tolist()
        
        if self.components:
            result["components"] = self.components
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustFieldTensor':
        """Create a trust field tensor from a dictionary."""
        # Convert direction list back to numpy array if it exists
        if "direction" in data and isinstance(data["direction"], list):
            data["direction"] = np.array(data["direction"])
        
        # Create the trust field tensor
        return cls(**data)


@dataclass
class TrustFieldSnapshot:
    """
    Represents a snapshot of a trust field at a specific point in time.
    
    This is useful for tracking the evolution of trust fields over time,
    enabling analysis of trends, anomalies, and the effectiveness of
    verification strategies.
    """
    
    # The trust field tensor at this snapshot
    tensor: TrustFieldTensor
    
    # When the snapshot was taken
    timestamp: datetime
    
    # Optional contextual information about the snapshot
    context: Optional[Dict[str, Any]] = None
    
    # Optional metadata about the snapshot
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the trust field snapshot after initialization."""
        # If timestamp is a string, convert it to datetime
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the trust field snapshot to a dictionary for storage/transmission."""
        result = {
            "tensor": self.tensor.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }
        
        # Add optional fields if they exist
        if self.context is not None:
            result["context"] = self.context
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustFieldSnapshot':
        """Create a trust field snapshot from a dictionary."""
        # Convert tensor dictionary to TrustFieldTensor
        if "tensor" in data and isinstance(data["tensor"], dict):
            data["tensor"] = TrustFieldTensor.from_dict(data["tensor"])
        
        # Create the trust field snapshot
        return cls(**data)


class TrustFieldTimeSeries:
    """
    Represents a time series of trust field snapshots.
    
    This enables tracking and analysis of how trust evolves over time,
    which is essential for understanding the temporal dynamics of trust
    and identifying patterns, trends, and anomalies.
    """
    
    def __init__(self, snapshots: Optional[List[TrustFieldSnapshot]] = None):
        """
        Initialize a trust field time series.
        
        Args:
            snapshots: Optional list of trust field snapshots
        """
        self.snapshots = snapshots or []
    
    def add_snapshot(self, snapshot: TrustFieldSnapshot) -> None:
        """
        Add a trust field snapshot to the time series.
        
        Args:
            snapshot: The trust field snapshot to add
        """
        self.snapshots.append(snapshot)
        # Sort snapshots by timestamp to maintain chronological order
        self.snapshots.sort(key=lambda s: s.timestamp)
    
    def get_snapshots(self, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> List[TrustFieldSnapshot]:
        """
        Get trust field snapshots within a specified time range.
        
        Args:
            start_time: Optional start time for the range
            end_time: Optional end time for the range
            
        Returns:
            List of trust field snapshots within the specified time range
        """
        if start_time is None and end_time is None:
            return self.snapshots
        
        filtered_snapshots = []
        for snapshot in self.snapshots:
            if start_time is not None and snapshot.timestamp < start_time:
                continue
            if end_time is not None and snapshot.timestamp > end_time:
                continue
            filtered_snapshots.append(snapshot)
        
        return filtered_snapshots
    
    def get_confidence_timeseries(self) -> Dict[datetime, float]:
        """
        Get a time series of confidence values.
        
        Returns:
            Dictionary mapping timestamps to confidence values
        """
        return {snapshot.timestamp: snapshot.tensor.confidence for snapshot in self.snapshots}
    
    def get_velocity_timeseries(self) -> Dict[datetime, float]:
        """
        Get a time series of velocity values.
        
        Returns:
            Dictionary mapping timestamps to velocity values
        """
        return {snapshot.timestamp: snapshot.tensor.velocity for snapshot in self.snapshots}
    
    def get_stability_timeseries(self) -> Dict[datetime, float]:
        """
        Get a time series of stability values.
        
        Returns:
            Dictionary mapping timestamps to stability values
        """
        return {snapshot.timestamp: snapshot.tensor.stability for snapshot in self.snapshots}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the trust field time series to a dictionary for storage/transmission."""
        return {
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustFieldTimeSeries':
        """Create a trust field time series from a dictionary."""
        # Convert snapshot dictionaries to TrustFieldSnapshot objects
        snapshots = []
        if "snapshots" in data and isinstance(data["snapshots"], list):
            for snapshot_data in data["snapshots"]:
                snapshots.append(TrustFieldSnapshot.from_dict(snapshot_data))
        
        # Create the trust field time series
        time_series = cls()
        time_series.snapshots = snapshots
        return time_series


@dataclass
class TrustFieldVisualizationConfig:
    """
    Configuration for visualizing trust fields.
    
    This provides options for customizing the visualization of trust fields,
    such as color schemes, dimensions to display, and time ranges.
    """
    
    # Time range to visualize
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Which dimensions of the trust field to visualize
    show_confidence: bool = True
    show_velocity: bool = True
    show_acceleration: bool = False
    show_stability: bool = True
    
    # Color scheme for different dimensions
    confidence_color: str = "#32CD32"  # Lime green
    velocity_color: str = "#1E90FF"    # Dodger blue
    acceleration_color: str = "#FF4500"  # Orange red
    stability_color: str = "#FFD700"   # Gold
    
    # Display as heatmap or line chart
    visualization_type: str = "line"  # "line" or "heatmap"
    
    # Whether to show anomalies
    highlight_anomalies: bool = True
    
    # Additional visualization options
    options: Dict[str, Any] = field(default_factory=dict)
