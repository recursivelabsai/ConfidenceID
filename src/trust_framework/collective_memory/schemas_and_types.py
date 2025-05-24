"""
Data structures and type definitions for the Collective Trust Memory module.

This module defines the core data structures used to store, retrieve, and analyze
verification events as "fossils" in a collective memory system. These structures
enable pattern recognition, anomaly detection, and archaeological excavation of
verification history.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import hashlib
import json
from enum import Enum
import uuid

from ..temporal_trust_field.schemas_and_types import ContentType, VerificationEventType


class PrivacyLevel(str, Enum):
    """Privacy levels for verification fossils."""
    PUBLIC = "public"              # Fully accessible to all network participants
    PROTECTED = "protected"        # Limited to authorized participants
    PRIVATE = "private"            # Only accessible to the creating node
    ANONYMIZED = "anonymized"      # Content fingerprint only, no context or source


class PatternType(str, Enum):
    """Types of patterns that can be recognized in the fossil record."""
    TEMPORAL = "temporal"          # Patterns over time
    CONTEXTUAL = "contextual"      # Patterns across contexts
    CROSS_MODAL = "cross_modal"    # Patterns across modalities
    SOURCE = "source"              # Patterns related to verification sources
    CONFIDENCE = "confidence"      # Patterns in confidence scores
    ANOMALY = "anomaly"            # Anomalous verification patterns
    CONSISTENCY = "consistency"    # Consistent verification patterns


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected in the fossil record."""
    CONFIDENCE_SPIKE = "confidence_spike"           # Sudden change in confidence
    VERIFICATION_CONTRADICTION = "contradiction"    # Contradictory verification results
    CONTEXT_MISMATCH = "context_mismatch"           # Verification in unexpected context
    TEMPORAL_DISCONTINUITY = "temporal_discontinuity"  # Unexpected temporal pattern
    CROSS_MODAL_INCONSISTENCY = "cross_modal_inconsistency"  # Inconsistency across modalities
    SOURCE_DEVIATION = "source_deviation"           # Unusual behavior from verification source
    PATTERN_BREAK = "pattern_break"                 # Break in established pattern


@dataclass
class VerificationFossil:
    """
    A fossil record of a verification event, stored in the collective memory.
    
    Fossils are the fundamental unit of the collective memory system. They capture
    not just the verification result but also rich contextual information and metadata
    that enable pattern recognition and archaeological analysis.
    """
    fossil_id: str                                # Unique identifier for this fossil
    content_fingerprint: str                      # Privacy-preserving hash of the content
    event_type: VerificationEventType             # Type of verification event
    content_type: ContentType                     # Type of content verified
    verification_score: float                     # 0.0 to 1.0
    timestamp: datetime                           # When the verification occurred
    privacy_level: PrivacyLevel                   # Privacy level for this fossil
    context: Dict[str, any] = field(default_factory=dict)  # Context of verification
    source_id: Optional[str] = None               # ID of verification source (node, system)
    source_type: Optional[str] = None             # Type of source (AI, human, hybrid)
    confidence: float = 1.0                       # Confidence in the verification result
    metadata: Dict[str, any] = field(default_factory=dict)  # Additional metadata
    related_fossils: List[str] = field(default_factory=list)  # IDs of related fossils
    tags: List[str] = field(default_factory=list)  # Searchable tags
    
    @classmethod
    def from_verification_event(cls, 
                               event_dict: Dict[str, any], 
                               content_fingerprint: str,
                               privacy_level: PrivacyLevel = PrivacyLevel.PROTECTED) -> 'VerificationFossil':
        """
        Create a fossil from a verification event dictionary.
        
        Args:
            event_dict: Dictionary containing verification event data
            content_fingerprint: Hash of the content being verified
            privacy_level: Privacy level for this fossil
            
        Returns:
            A VerificationFossil object
        """
        # Generate a unique ID for this fossil
        fossil_id = f"fossil-{uuid.uuid4()}"
        
        # Extract fields from the event dictionary
        event_type = VerificationEventType(event_dict["event_type"])
        content_type = ContentType(event_dict["content_type"])
        verification_score = float(event_dict["verification_score"])
        
        # Convert timestamp string to datetime if necessary
        if isinstance(event_dict["timestamp"], str):
            timestamp = datetime.fromisoformat(event_dict["timestamp"])
        else:
            timestamp = event_dict["timestamp"]
        
        # Create and return the fossil
        return cls(
            fossil_id=fossil_id,
            content_fingerprint=content_fingerprint,
            event_type=event_type,
            content_type=content_type,
            verification_score=verification_score,
            timestamp=timestamp,
            privacy_level=privacy_level,
            context=event_dict.get("context", {}),
            source_id=event_dict.get("source_id"),
            source_type=event_dict.get("source_type"),
            confidence=event_dict.get("confidence", 1.0),
            metadata=event_dict.get("metadata", {}),
            related_fossils=event_dict.get("related_fossils", []),
            tags=event_dict.get("tags", [])
        )
    
    def to_dict(self, include_private: bool = False) -> Dict:
        """
        Convert to dictionary for serialization.
        
        Args:
            include_private: Whether to include private fields
            
        Returns:
            Dictionary representation of the fossil
        """
        # Base dictionary with non-private fields
        result = {
            "fossil_id": self.fossil_id,
            "content_fingerprint": self.content_fingerprint,
            "event_type": self.event_type.value,
            "content_type": self.content_type.value,
            "verification_score": self.verification_score,
            "timestamp": self.timestamp.isoformat(),
            "privacy_level": self.privacy_level.value,
        }
        
        # Add context only if not anonymized
        if self.privacy_level != PrivacyLevel.ANONYMIZED:
            result["context"] = self.context
            result["tags"] = self.tags
        
        # Add source information only if privacy level permits
        if include_private or self.privacy_level in [PrivacyLevel.PUBLIC, PrivacyLevel.PROTECTED]:
            result["source_id"] = self.source_id
            result["source_type"] = self.source_type
            result["confidence"] = self.confidence
            result["related_fossils"] = self.related_fossils
        
        # Add metadata only if privacy level permits and caller requests it
        if include_private and self.privacy_level != PrivacyLevel.ANONYMIZED:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationFossil':
        """
        Create from dictionary after deserialization.
        
        Args:
            data: Dictionary data
            
        Returns:
            A VerificationFossil object
        """
        return cls(
            fossil_id=data["fossil_id"],
            content_fingerprint=data["content_fingerprint"],
            event_type=VerificationEventType(data["event_type"]),
            content_type=ContentType(data["content_type"]),
            verification_score=data["verification_score"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            privacy_level=PrivacyLevel(data["privacy_level"]),
            context=data.get("context", {}),
            source_id=data.get("source_id"),
            source_type=data.get("source_type"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            related_fossils=data.get("related_fossils", []),
            tags=data.get("tags", [])
        )
    
    def anonymize(self) -> 'VerificationFossil':
        """
        Create an anonymized copy of this fossil.
        
        Returns:
            An anonymized copy of this fossil
        """
        return VerificationFossil(
            fossil_id=self.fossil_id,
            content_fingerprint=self.content_fingerprint,
            event_type=self.event_type,
            content_type=self.content_type,
            verification_score=self.verification_score,
            timestamp=self.timestamp,
            privacy_level=PrivacyLevel.ANONYMIZED,
            # No context, source, or metadata in anonymized version
        )


@dataclass
class TemporalPattern:
    """
    A pattern detected in verification events over time.
    """
    pattern_id: str                               # Unique identifier for this pattern
    pattern_type: PatternType                     # Type of pattern
    content_fingerprints: List[str]               # Fingerprints of content involved
    start_time: datetime                          # Start of pattern period
    end_time: datetime                            # End of pattern period
    fossil_ids: List[str]                         # IDs of fossils that form this pattern
    detection_time: datetime                      # When the pattern was detected
    confidence: float                             # 0.0 to 1.0 confidence in pattern
    description: str                              # Human-readable description
    metrics: Dict[str, float] = field(default_factory=dict)  # Quantitative metrics
    visualizable: bool = True                     # Whether this pattern can be visualized
    visualization_data: Dict[str, any] = field(default_factory=dict)  # Data for visualization
    metadata: Dict[str, any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "content_fingerprints": self.content_fingerprints,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "fossil_ids": self.fossil_ids,
            "detection_time": self.detection_time.isoformat(),
            "confidence": self.confidence,
            "description": self.description,
            "metrics": self.metrics,
            "visualizable": self.visualizable,
            "visualization_data": self.visualization_data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TemporalPattern':
        """Create from dictionary after deserialization."""
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=PatternType(data["pattern_type"]),
            content_fingerprints=data["content_fingerprints"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            fossil_ids=data["fossil_ids"],
            detection_time=datetime.fromisoformat(data["detection_time"]),
            confidence=data["confidence"],
            description=data["description"],
            metrics=data.get("metrics", {}),
            visualizable=data.get("visualizable", True),
            visualization_data=data.get("visualization_data", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class VerificationAnomaly:
    """
    An anomaly detected in the verification fossil record.
    """
    anomaly_id: str                              # Unique identifier for this anomaly
    anomaly_type: AnomalyType                    # Type of anomaly
    content_fingerprints: List[str]              # Fingerprints of content involved
    fossil_ids: List[str]                        # IDs of fossils that form this anomaly
    detection_time: datetime                     # When the anomaly was detected
    severity: float                              # 0.0 to 1.0 severity
    confidence: float                            # 0.0 to 1.0 confidence in detection
    description: str                             # Human-readable description
    expected_behavior: Optional[str] = None      # Description of expected behavior
    potential_causes: List[str] = field(default_factory=list)  # Potential causes
    recommended_actions: List[str] = field(default_factory=list)  # Recommended actions
    metrics: Dict[str, float] = field(default_factory=dict)  # Quantitative metrics
    visualizable: bool = True                    # Whether this anomaly can be visualized
    visualization_data: Dict[str, any] = field(default_factory=dict)  # Data for visualization
    metadata: Dict[str, any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "content_fingerprints": self.content_fingerprints,
            "fossil_ids": self.fossil_ids,
            "detection_time": self.detection_time.isoformat(),
            "severity": self.severity,
            "confidence": self.confidence,
            "description": self.description,
            "expected_behavior": self.expected_behavior,
            "potential_causes": self.potential_causes,
            "recommended_actions": self.recommended_actions,
            "metrics": self.metrics,
            "visualizable": self.visualizable,
            "visualization_data": self.visualization_data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationAnomaly':
        """Create from dictionary after deserialization."""
        return cls(
            anomaly_id=data["anomaly_id"],
            anomaly_type=AnomalyType(data["anomaly_type"]),
            content_fingerprints=data["content_fingerprints"],
            fossil_ids=data["fossil_ids"],
            detection_time=datetime.fromisoformat(data["detection_time"]),
            severity=data["severity"],
            confidence=data["confidence"],
            description=data["description"],
            expected_behavior=data.get("expected_behavior"),
            potential_causes=data.get("potential_causes", []),
            recommended_actions=data.get("recommended_actions", []),
            metrics=data.get("metrics", {}),
            visualizable=data.get("visualizable", True),
            visualization_data=data.get("visualization_data", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class VerificationConsistency:
    """
    A consistency pattern detected in the verification fossil record.
    """
    consistency_id: str                          # Unique identifier for this consistency
    content_fingerprints: List[str]              # Fingerprints of content involved
    fossil_ids: List[str]                        # IDs of fossils that form this consistency
    detection_time: datetime                     # When the consistency was detected
    strength: float                              # 0.0 to 1.0 strength of consistency
    confidence: float                            # 0.0 to 1.0 confidence in detection
    description: str                             # Human-readable description
    stability_metric: float                      # Measure of stability over time
    duration: int                                # Duration in days
    metrics: Dict[str, float] = field(default_factory=dict)  # Quantitative metrics
    visualizable: bool = True                    # Whether this consistency can be visualized
    visualization_data: Dict[str, any] = field(default_factory=dict)  # Data for visualization
    metadata: Dict[str, any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "consistency_id": self.consistency_id,
            "content_fingerprints": self.content_fingerprints,
            "fossil_ids": self.fossil_ids,
            "detection_time": self.detection_time.isoformat(),
            "strength": self.strength,
            "confidence": self.confidence,
            "description": self.description,
            "stability_metric": self.stability_metric,
            "duration": self.duration,
            "metrics": self.metrics,
            "visualizable": self.visualizable,
            "visualization_data": self.visualization_data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationConsistency':
        """Create from dictionary after deserialization."""
        return cls(
            consistency_id=data["consistency_id"],
            content_fingerprints=data["content_fingerprints"],
            fossil_ids=data["fossil_ids"],
            detection_time=datetime.fromisoformat(data["detection_time"]),
            strength=data["strength"],
            confidence=data["confidence"],
            description=data["description"],
            stability_metric=data["stability_metric"],
            duration=data["duration"],
            metrics=data.get("metrics", {}),
            visualizable=data.get("visualizable": self.visualizable,
            "visualization_data": self.visualization_data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationConsistency':
        """Create from dictionary after deserialization."""
        return cls(
            consistency_id=data["consistency_id"],
            content_fingerprints=data["content_fingerprints"],
            fossil_ids=data["fossil_ids"],
            detection_time=datetime.fromisoformat(data["detection_time"]),
            strength=data["strength"],
            confidence=data["confidence"],
            description=data["description"],
            stability_metric=data["stability_metric"],
            duration=data["duration"],
            metrics=data.get("metrics", {}),
            visualizable=data.get("visualizable", True),
            visualization_data=data.get("visualization_data", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConfidenceEvolution:
    """
    Tracks the evolution of confidence scores for a specific content over time.
    """
    content_fingerprint: str                     # Fingerprint of the content
    fossil_ids: List[str]                        # IDs of fossils in chronological order
    scores: List[float]                          # Confidence scores in chronological order
    timestamps: List[datetime]                   # Timestamps for each score
    trend: float                                 # Overall trend direction (-1.0 to 1.0)
    stability: float                             # Stability of scores over time (0.0 to 1.0)
    description: str                             # Human-readable description
    visualization_data: Dict[str, any] = field(default_factory=dict)  # Data for visualization
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "content_fingerprint": self.content_fingerprint,
            "fossil_ids": self.fossil_ids,
            "scores": self.scores,
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "trend": self.trend,
            "stability": self.stability,
            "description": self.description,
            "visualization_data": self.visualization_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConfidenceEvolution':
        """Create from dictionary after deserialization."""
        return cls(
            content_fingerprint=data["content_fingerprint"],
            fossil_ids=data["fossil_ids"],
            scores=data["scores"],
            timestamps=[datetime.fromisoformat(ts) for ts in data["timestamps"]],
            trend=data["trend"],
            stability=data["stability"],
            description=data["description"],
            visualization_data=data.get("visualization_data", {})
        )


@dataclass
class TemporalRange:
    """
    Represents a range of time for querying the fossil record.
    """
    start: Optional[datetime] = None  # Start of range (inclusive)
    end: Optional[datetime] = None    # End of range (inclusive)
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if a timestamp is within this range."""
        if self.start and timestamp < self.start:
            return False
        if self.end and timestamp > self.end:
            return False
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {}
        if self.start:
            result["start"] = self.start.isoformat()
        if self.end:
            result["end"] = self.end.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TemporalRange':
        """Create from dictionary after deserialization."""
        start = datetime.fromisoformat(data["start"]) if "start" in data else None
        end = datetime.fromisoformat(data["end"]) if "end" in data else None
        return cls(start=start, end=end)


@dataclass
class ContextFilter:
    """
    Filter for contexts when querying the fossil record.
    """
    exact_match: Dict[str, str] = field(default_factory=dict)  # Keys that must exactly match
    partial_match: Dict[str, List[str]] = field(default_factory=dict)  # Keys with any of these values
    exclude: Dict[str, List[str]] = field(default_factory=dict)  # Keys that should not have these values
    
    def matches(self, context: Dict[str, any]) -> bool:
        """Check if a context matches this filter."""
        # Check exact matches
        for key, value in self.exact_match.items():
            if key not in context or context[key] != value:
                return False
        
        # Check partial matches
        for key, values in self.partial_match.items():
            if key not in context or context[key] not in values:
                return False
        
        # Check exclusions
        for key, values in self.exclude.items():
            if key in context and context[key] in values:
                return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "exact_match": self.exact_match,
            "partial_match": self.partial_match,
            "exclude": self.exclude
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ContextFilter':
        """Create from dictionary after deserialization."""
        return cls(
            exact_match=data.get("exact_match", {}),
            partial_match=data.get("partial_match", {}),
            exclude=data.get("exclude", {})
        )


@dataclass
class ContentFingerprint:
    """
    A privacy-preserving fingerprint of content, used as the primary key in the fossil record.
    """
    fingerprint: str                             # The actual fingerprint hash
    fingerprint_type: str = "sha256"             # Hashing algorithm used
    source_modalities: List[str] = field(default_factory=list)  # Modalities used to generate the fingerprint
    
    @classmethod
    def generate(cls, content: Union[str, bytes], modalities: List[str] = None) -> 'ContentFingerprint':
        """
        Generate a fingerprint from content.
        
        Args:
            content: The content to fingerprint (string or bytes)
            modalities: List of modalities used to generate the fingerprint
            
        Returns:
            A ContentFingerprint object
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        fingerprint = hashlib.sha256(content).hexdigest()
        
        return cls(
            fingerprint=fingerprint,
            fingerprint_type="sha256",
            source_modalities=modalities or ["unknown"]
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "fingerprint": self.fingerprint,
            "fingerprint_type": self.fingerprint_type,
            "source_modalities": self.source_modalities
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ContentFingerprint':
        """Create from dictionary after deserialization."""
        return cls(
            fingerprint=data["fingerprint"],
            fingerprint_type=data.get("fingerprint_type", "sha256"),
            source_modalities=data.get("source_modalities", ["unknown"])
        )


@dataclass
class ArchaeologyQuery:
    """
    A query for excavating patterns from the fossil record.
    """
    content_fingerprint: Optional[str] = None    # Specific content fingerprint to focus on
    temporal_range: Optional[TemporalRange] = None  # Time range to consider
    context_filter: Optional[ContextFilter] = None  # Filter for contexts
    verification_types: List[VerificationEventType] = field(default_factory=list)  # Types to include
    content_types: List[ContentType] = field(default_factory=list)  # Content types to include
    pattern_types: List[PatternType] = field(default_factory=list)  # Pattern types to look for
    related_fingerprints: List[str] = field(default_factory=list)  # Related content to include
    include_anomalies: bool = True               # Whether to include anomalies
    include_consistencies: bool = True           # Whether to include consistencies
    include_evolution: bool = True               # Whether to include confidence evolution
    max_results: int = 100                       # Maximum number of results
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "include_anomalies": self.include_anomalies,
            "include_consistencies": self.include_consistencies,
            "include_evolution": self.include_evolution,
            "max_results": self.max_results
        }
        
        if self.content_fingerprint:
            result["content_fingerprint"] = self.content_fingerprint
        
        if self.temporal_range:
            result["temporal_range"] = self.temporal_range.to_dict()
        
        if self.context_filter:
            result["context_filter"] = self.context_filter.to_dict()
        
        if self.verification_types:
            result["verification_types"] = [vt.value for vt in self.verification_types]
        
        if self.content_types:
            result["content_types"] = [ct.value for ct in self.content_types]
        
        if self.pattern_types:
            result["pattern_types"] = [pt.value for pt in self.pattern_types]
        
        if self.related_fingerprints:
            result["related_fingerprints"] = self.related_fingerprints
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ArchaeologyQuery':
        """Create from dictionary after deserialization."""
        # Convert enum string values back to enums
        verification_types = []
        if "verification_types" in data:
            verification_types = [VerificationEventType(vt) for vt in data["verification_types"]]
        
        content_types = []
        if "content_types" in data:
            content_types = [ContentType(ct) for ct in data["content_types"]]
        
        pattern_types = []
        if "pattern_types" in data:
            pattern_types = [PatternType(pt) for pt in data["pattern_types"]]
        
        # Create temporal range and context filter if provided
        temporal_range = None
        if "temporal_range" in data:
            temporal_range = TemporalRange.from_dict(data["temporal_range"])
        
        context_filter = None
        if "context_filter" in data:
            context_filter = ContextFilter.from_dict(data["context_filter"])
        
        return cls(
            content_fingerprint=data.get("content_fingerprint"),
            temporal_range=temporal_range,
            context_filter=context_filter,
            verification_types=verification_types,
            content_types=content_types,
            pattern_types=pattern_types,
            related_fingerprints=data.get("related_fingerprints", []),
            include_anomalies=data.get("include_anomalies", True),
            include_consistencies=data.get("include_consistencies", True),
            include_evolution=data.get("include_evolution", True),
            max_results=data.get("max_results", 100)
        )


@dataclass
class TrustArchaeologyReport:
    """
    A comprehensive report from archaeological analysis of the fossil record.
    """
    report_id: str                               # Unique identifier for this report
    query: ArchaeologyQuery                      # The query that generated this report
    generation_time: datetime                    # When the report was generated
    
    # Excavated patterns and insights
    temporal_patterns: List[TemporalPattern] = field(default_factory=list)
    anomalies: List[VerificationAnomaly] = field(default_factory=list)
    consistencies: List[VerificationConsistency] = field(default_factory=list)
    confidence_evolution: Optional[ConfidenceEvolution] = None
    
    # Summary and recommendations
    pattern_summary: str = ""                    # Summary of patterns found
    anomaly_summary: str = ""                    # Summary of anomalies found
    consistency_summary: str = ""                # Summary of consistencies found
    evolution_summary: str = ""                  # Summary of confidence evolution
    recommendations: List[str] = field(default_factory=list)  # Recommendations based on findings
    
    # Metadata
    fossil_count: int = 0                        # Number of fossils analyzed
    pattern_confidence: float = 1.0              # Overall confidence in patterns
    report_version: str = "1.0"                  # Version of the report format
    metadata: Dict[str, any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "report_id": self.report_id,
            "query": self.query.to_dict(),
            "generation_time": self.generation_time.isoformat(),
            "temporal_patterns": [tp.to_dict() for tp in self.temporal_patterns],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "consistencies": [c.to_dict() for c in self.consistencies],
            "pattern_summary": self.pattern_summary,
            "anomaly_summary": self.anomaly_summary,
            "consistency_summary": self.consistency_summary,
            "evolution_summary": self.evolution_summary,
            "recommendations": self.recommendations,
            "fossil_count": self.fossil_count,
            "pattern_confidence": self.pattern_confidence,
            "report_version": self.report_version,
            "metadata": self.metadata
        }
        
        if self.confidence_evolution:
            result["confidence_evolution"] = self.confidence_evolution.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrustArchaeologyReport':
        """Create from dictionary after deserialization."""
        # Convert nested objects
        query = ArchaeologyQuery.from_dict(data["query"])
        temporal_patterns = [TemporalPattern.from_dict(tp) for tp in data["temporal_patterns"]]
        anomalies = [VerificationAnomaly.from_dict(a) for a in data["anomalies"]]
        consistencies = [VerificationConsistency.from_dict(c) for c in data["consistencies"]]
        
        confidence_evolution = None
        if "confidence_evolution" in data:
            confidence_evolution = ConfidenceEvolution.from_dict(data["confidence_evolution"])
        
        return cls(
            report_id=data["report_id"],
            query=query,
            generation_time=datetime.fromisoformat(data["generation_time"]),
            temporal_patterns=temporal_patterns,
            anomalies=anomalies,
            consistencies=consistencies,
            confidence_evolution=confidence_evolution,
            pattern_summary=data.get("pattern_summary", ""),
            anomaly_summary=data.get("anomaly_summary", ""),
            consistency_summary=data.get("consistency_summary", ""),
            evolution_summary=data.get("evolution_summary", ""),
            recommendations=data.get("recommendations", []),
            fossil_count=data.get("fossil_count", 0),
            pattern_confidence=data.get("pattern_confidence", 1.0),
            report_version=data.get("report_version", "1.0"),
            metadata=data.get("metadata", {})
        )


# Factory functions for creating default objects

def create_default_archaeology_query() -> ArchaeologyQuery:
    """
    Create a default archaeology query with sensible defaults.
    
    Returns:
        An ArchaeologyQuery with default settings
    """
    return ArchaeologyQuery(
        temporal_range=TemporalRange(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now()
        ),
        include_anomalies=True,
        include_consistencies=True,
        include_evolution=True,
        max_results=100
    )


def create_content_fingerprint_from_text(text: str) -> str:
    """
    Generate a content fingerprint from text.
    
    Args:
        text: The text to fingerprint
        
    Returns:
        A content fingerprint string
    """
    return ContentFingerprint.generate(text, ["text"]).fingerprint


def create_empty_archaeology_report(query: ArchaeologyQuery) -> TrustArchaeologyReport:
    """
    Create an empty archaeology report for a query.
    
    Args:
        query: The archaeology query
        
    Returns:
        An empty TrustArchaeologyReport
    """
    return TrustArchaeologyReport(
        report_id=f"report-{uuid.uuid4()}",
        query=query,
        generation_time=datetime.now(),
        fossil_count=0,
        pattern_confidence=0.0
    )
