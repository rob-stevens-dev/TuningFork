"""Audit logging for TuningFork operations.

This module provides comprehensive audit logging capabilities for tracking
changes, user actions, system events, and compliance requirements.

Classes:
    AuditLogger: Main audit logging interface
    AuditEvent: Structured audit event representation
    AuditTrail: Audit trail management and querying
    ComplianceLogger: Specialized compliance logging

Example:
    >>> audit = AuditLogger("system.configuration")
    >>> audit.log_change(
    ...     action="update_database_config",
    ...     resource="database.prod_db",
    ...     user="admin@company.com",
    ...     old_value={"max_connections": 100},
    ...     new_value={"max_connections": 200}
    ... )
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .structured import StructuredLogger
from ..core.exceptions import TuningForkException, ValidationError
from ..core.utils import ValidationUtils


class AuditEventType(Enum):
    """Types of audit events."""
    
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    
    # Authentication events
    USER_LOGIN = "auth.login"
    USER_LOGOUT = "auth.logout"
    AUTH_FAILURE = "auth.failure"
    SESSION_EXPIRED = "auth.session_expired"
    
    # Configuration events
    CONFIG_CREATE = "config.create"
    CONFIG_UPDATE = "config.update"
    CONFIG_DELETE = "config.delete"
    CONFIG_RESTORE = "config.restore"
    
    # Database events
    DB_CONNECT = "database.connect"
    DB_DISCONNECT = "database.disconnect"
    DB_QUERY = "database.query"
    DB_MODIFY = "database.modify"
    
    # Analysis events
    ANALYSIS_START = "analysis.start"
    ANALYSIS_COMPLETE = "analysis.complete"
    ANALYSIS_FAIL = "analysis.fail"
    
    # Optimization events
    OPTIMIZATION_START = "optimization.start"
    OPTIMIZATION_APPLY = "optimization.apply"
    OPTIMIZATION_ROLLBACK = "optimization.rollback"
    OPTIMIZATION_COMPLETE = "optimization.complete"
    
    # Access events
    RESOURCE_ACCESS = "access.resource"
    PERMISSION_DENIED = "access.denied"
    PRIVILEGE_ESCALATION = "access.privilege_escalation"
    
    # Data events
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    DATA_DELETE = "data.delete"
    BACKUP_CREATE = "backup.create"
    BACKUP_RESTORE = "backup.restore"
    
    # Security events
    SECURITY_VIOLATION = "security.violation"
    INTRUSION_ATTEMPT = "security.intrusion"
    POLICY_VIOLATION = "security.policy_violation"
    
    # Administrative events
    ADMIN_ACTION = "admin.action"
    USER_CREATED = "admin.user_created"
    USER_DELETED = "admin.user_deleted"
    ROLE_ASSIGNED = "admin.role_assigned"
    
    # Custom events
    CUSTOM = "custom"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Structured audit event representation.
    
    This class represents a single audit event with all necessary
    information for compliance, security, and operational tracking.
    
    Attributes:
        event_id: Unique event identifier
        event_type: Type of audit event
        timestamp: Event timestamp (ISO format)
        severity: Event severity level
        actor: User or system that triggered the event
        resource: Resource being acted upon
        action: Specific action performed
        outcome: Result of the action (success/failure)
        details: Additional event details
        context: Environmental context
        compliance_tags: Tags for compliance frameworks
        retention_period: How long to retain this event
    """
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.CUSTOM
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    severity: AuditSeverity = AuditSeverity.MEDIUM
    actor: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: List[str] = field(default_factory=list)
    retention_period: Optional[int] = None  # Days
    
    def __post_init__(self) -> None:
        """Validate audit event after initialization."""
        # Ensure event_type is AuditEventType enum
        if isinstance(self.event_type, str):
            try:
                self.event_type = AuditEventType(self.event_type)
            except ValueError:
                self.event_type = AuditEventType.CUSTOM
        
        # Ensure severity is AuditSeverity enum
        if isinstance(self.severity, str):
            try:
                self.severity = AuditSeverity(self.severity)
            except ValueError:
                self.severity = AuditSeverity.MEDIUM
        
        # Validate timestamp format
        try:
            datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        except ValueError:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary.
        
        Returns:
            Dictionary representation of audit event
        """
        data = asdict(self)
        # Convert enums to strings
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    def to_json(self) -> str:
        """Convert audit event to JSON string.
        
        Returns:
            JSON representation of audit event
        """
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create audit event from dictionary.
        
        Args:
            data: Dictionary containing audit event data
            
        Returns:
            AuditEvent instance
        """
        # Handle enum conversions
        if 'event_type' in data and isinstance(data['event_type'], str):
            try:
                data['event_type'] = AuditEventType(data['event_type'])
            except ValueError:
                data['event_type'] = AuditEventType.CUSTOM
        
        if 'severity' in data and isinstance(data['severity'], str):
            try:
                data['severity'] = AuditSeverity(data['severity'])
            except ValueError:
                data['severity'] = AuditSeverity.MEDIUM
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AuditEvent':
        """Create audit event from JSON string.
        
        Args:
            json_str: JSON string containing audit event data
            
        Returns:
            AuditEvent instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def add_compliance_tag(self, tag: str) -> None:
        """Add compliance framework tag.
        
        Args:
            tag: Compliance tag (e.g., "SOX", "GDPR", "HIPAA")
        """
        if tag not in self.compliance_tags:
            self.compliance_tags.append(tag)
    
    def set_retention_period(self, days: int) -> None:
        """Set retention period for this event.
        
        Args:
            days: Number of days to retain the event
        """
        if days < 0:
            raise ValidationError("Retention period must be non-negative")
        self.retention_period = days
    
    @property
    def expiry_date(self) -> Optional[datetime]:
        """Get event expiry date based on retention period.
        
        Returns:
            Expiry datetime or None if no retention period set
        """
        if self.retention_period is None:
            return None
        
        event_time = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        return event_time + timedelta(days=self.retention_period)
    
    @property
    def is_expired(self) -> bool:
        """Check if event has expired based on retention period.
        
        Returns:
            True if event has expired
        """
        expiry = self.expiry_date
        if expiry is None:
            return False
        
        return datetime.now(timezone.utc) > expiry
    
    def __repr__(self) -> str:
        """Return string representation of audit event."""
        return (
            f"AuditEvent("
            f"id={self.event_id[:8]}..., "
            f"type={self.event_type.value}, "
            f"actor={self.actor}, "
            f"action={self.action})"
        )


class AuditLogger:
    """Comprehensive audit logging system.
    
    This class provides audit logging capabilities for tracking changes,
    user actions, system events, and compliance requirements.
    
    Attributes:
        name: Logger name
        logger: Underlying structured logger
        events: In-memory event storage
        compliance_mode: Whether compliance mode is enabled
        
    Example:
        >>> audit = AuditLogger("system.database")
        >>> audit.log_change(
        ...     action="update_config",
        ...     resource="database.prod",
        ...     user="admin@company.com",
        ...     old_value={"timeout": 30},
        ...     new_value={"timeout": 60}
        ... )
    """
    
    def __init__(
        self,
        name: str,
        *,
        compliance_mode: bool = False,
        retain_in_memory: bool = True,
        default_retention_days: int = 2555,  # 7 years default
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        """Initialize audit logger.
        
        Args:
            name: Logger name
            compliance_mode: Enable compliance-specific features
            retain_in_memory: Whether to retain events in memory
            default_retention_days: Default retention period in days
            logger: Custom structured logger instance
        """
        self.name = name
        self.compliance_mode = compliance_mode
        self.retain_in_memory = retain_in_memory
        self.default_retention_days = default_retention_days
        self.logger = logger or StructuredLogger(f"audit.{name}")
        
        # In-memory event storage
        self._events: List[AuditEvent] = []
        
        # Compliance settings
        self._compliance_frameworks: List[str] = []
        
        # Event handlers
        self._event_handlers: List[callable] = []
    
    def add_compliance_framework(self, framework: str) -> None:
        """Add compliance framework to audit logger.
        
        Args:
            framework: Compliance framework name (e.g., "SOX", "GDPR")
        """
        if framework not in self._compliance_frameworks:
            self._compliance_frameworks.append(framework)
            self.logger.info(
                "Compliance framework added",
                framework=framework,
                total_frameworks=len(self._compliance_frameworks)
            )
    
    def add_event_handler(self, handler: callable) -> None:
        """Add custom event handler.
        
        Args:
            handler: Function to call for each audit event
        """
        self._event_handlers.append(handler)
    
    def log_event(
        self,
        event_type: Union[AuditEventType, str],
        *,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        outcome: str = "success",
        severity: Union[AuditSeverity, str] = AuditSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        compliance_tags: Optional[List[str]] = None,
        retention_days: Optional[int] = None,
    ) -> AuditEvent:
        """Log a general audit event.
        
        Args:
            event_type: Type of audit event
            actor: User or system that triggered the event
            resource: Resource being acted upon
            action: Specific action performed
            outcome: Result of the action
            severity: Event severity level
            details: Additional event details
            context: Environmental context
            compliance_tags: Tags for compliance frameworks
            retention_days: Custom retention period
            
        Returns:
            Created AuditEvent instance
        """
        # Create audit event
        event = AuditEvent(
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            outcome=outcome,
            severity=severity,
            details=details or {},
            context=context or {},
            compliance_tags=compliance_tags or [],
            retention_period=retention_days or self.default_retention_days,
        )
        
        # Add compliance framework tags if in compliance mode
        if self.compliance_mode:
            for framework in self._compliance_frameworks:
                event.add_compliance_tag(framework)
        
        # Store event in memory if enabled
        if self.retain_in_memory:
            self._events.append(event)
        
        # Log event with structured logger
        self.logger.info(
            f"Audit event: {action or event_type}",
            **event.to_dict()
        )
        
        # Call custom event handlers
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(
                    "Audit event handler failed",
                    handler=handler.__name__,
                    error=str(e)
                )
        
        return event
    
    def log_authentication(
        self,
        action: str,
        user: str,
        *,
        outcome: str = "success",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs: Any
    ) -> AuditEvent:
        """Log authentication event.
        
        Args:
            action: Authentication action (login, logout, etc.)
            user: Username or user identifier
            outcome: Authentication outcome
            ip_address: Client IP address
            user_agent: Client user agent
            session_id: Session identifier
            **kwargs: Additional context
            
        Returns:
            Created AuditEvent instance
        """
        event_type_map = {
            "login": AuditEventType.USER_LOGIN,
            "logout": AuditEventType.USER_LOGOUT,
            "failure": AuditEventType.AUTH_FAILURE,
            "expired": AuditEventType.SESSION_EXPIRED,
        }
        
        context = {
            "ip_address": ip_address,
            "user_agent": user_agent,
            "session_id": session_id,
            **kwargs
        }
        
        severity = AuditSeverity.HIGH if outcome != "success" else AuditSeverity.MEDIUM
        
        return self.log_event(
            event_type=event_type_map.get(action, AuditEventType.USER_LOGIN),
            actor=user,
            resource=f"auth.{user}",
            action=action,
            outcome=outcome,
            severity=severity,
            context=context,
            compliance_tags=["SOX", "GDPR"] if self.compliance_mode else None,
        )
    
    def log_change(
        self,
        action: str,
        resource: str,
        *,
        actor: Optional[str] = None,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        change_reason: Optional[str] = None,
        **kwargs: Any
    ) -> AuditEvent:
        """Log configuration or data change event.
        
        Args:
            action: Change action performed
            resource: Resource that was changed
            actor: User who made the change
            old_value: Previous value
            new_value: New value
            change_reason: Reason for the change
            **kwargs: Additional context
            
        Returns:
            Created AuditEvent instance
        """
        details = {
            "old_value": old_value,
            "new_value": new_value,
            "change_reason": change_reason,
        }
        
        # Remove None values
        details = {k: v for k, v in details.items() if v is not None}
        
        return self.log_event(
            event_type=AuditEventType.CONFIG_UPDATE,
            actor=actor,
            resource=resource,
            action=action,
            outcome="success",
            severity=AuditSeverity.MEDIUM,
            details=details,
            context=kwargs,
            compliance_tags=["SOX", "CHANGE_CONTROL"] if self.compliance_mode else None,
        )
    
    def log_access(
        self,
        resource: str,
        actor: str,
        *,
        action: str = "access",
        outcome: str = "success",
        permission_level: Optional[str] = None,
        **kwargs: Any
    ) -> AuditEvent:
        """Log resource access event.
        
        Args:
            resource: Resource being accessed
            actor: User accessing the resource
            action: Access action
            outcome: Access outcome
            permission_level: Permission level used
            **kwargs: Additional context
            
        Returns:
            Created AuditEvent instance
        """
        event_type = (
            AuditEventType.PERMISSION_DENIED 
            if outcome != "success" 
            else AuditEventType.RESOURCE_ACCESS
        )
        
        severity = AuditSeverity.HIGH if outcome != "success" else AuditSeverity.LOW
        
        details = {"permission_level": permission_level} if permission_level else {}
        
        return self.log_event(
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            outcome=outcome,
            severity=severity,
            details=details,
            context=kwargs,
            compliance_tags=["GDPR", "ACCESS_CONTROL"] if self.compliance_mode else None,
        )
    
    def log_security_event(
        self,
        event_description: str,
        *,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        threat_level: str = "medium",
        indicators: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AuditEvent:
        """Log security-related event.
        
        Args:
            event_description: Description of security event
            actor: Actor involved in the event
            resource: Resource affected
            threat_level: Threat level assessment
            indicators: List of threat indicators
            **kwargs: Additional context
            
        Returns:
            Created AuditEvent instance
        """
        severity_map = {
            "low": AuditSeverity.LOW,
            "medium": AuditSeverity.MEDIUM,
            "high": AuditSeverity.HIGH,
            "critical": AuditSeverity.CRITICAL,
        }
        
        details = {
            "threat_level": threat_level,
            "indicators": indicators or [],
        }
        
        return self.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            actor=actor,
            resource=resource,
            action="security_event",
            outcome="detected",
            severity=severity_map.get(threat_level, AuditSeverity.MEDIUM),
            details=details,
            context=kwargs,
            compliance_tags=["SECURITY", "INCIDENT_RESPONSE"] if self.compliance_mode else None,
        )
    
    def get_events(
        self,
        *,
        event_type: Optional[Union[AuditEventType, str]] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        severity: Optional[Union[AuditSeverity, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """Query audit events with filtering.
        
        Args:
            event_type: Filter by event type
            actor: Filter by actor
            resource: Filter by resource
            severity: Filter by severity level
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum number of events to return
            
        Returns:
            List of matching audit events
        """
        if not self.retain_in_memory:
            self.logger.warning("Event querying not available - in-memory retention disabled")
            return []
        
        filtered_events = self._events[:]
        
        # Apply filters
        if event_type:
            if isinstance(event_type, str):
                try:
                    event_type = AuditEventType(event_type)
                except ValueError:
                    pass
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if actor:
            filtered_events = [e for e in filtered_events if e.actor == actor]
        
        if resource:
            filtered_events = [e for e in filtered_events if e.resource == resource]
        
        if severity:
            if isinstance(severity, str):
                try:
                    severity = AuditSeverity(severity)
                except ValueError:
                    pass
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        if start_time:
            filtered_events = [
                e for e in filtered_events 
                if datetime.fromisoformat(e.timestamp.replace('Z', '+00:00')) >= start_time
            ]
        
        if end_time:
            filtered_events = [
                e for e in filtered_events 
                if datetime.fromisoformat(e.timestamp.replace('Z', '+00:00')) <= end_time
            ]
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            filtered_events = filtered_events[:limit]
        
        return filtered_events
    
    def get_audit_trail(
        self,
        resource: str,
        *,
        limit: Optional[int] = None
    ) -> List[AuditEvent]:
        """Get complete audit trail for a specific resource.
        
        Args:
            resource: Resource to get audit trail for
            limit: Maximum number of events to return
            
        Returns:
            List of audit events for the resource
        """
        return self.get_events(resource=resource, limit=limit)
    
    def cleanup_expired_events(self) -> int:
        """Remove expired events based on retention periods.
        
        Returns:
            Number of events removed
        """
        if not self.retain_in_memory:
            return 0
        
        initial_count = len(self._events)
        self._events = [e for e in self._events if not e.is_expired]
        removed_count = initial_count - len(self._events)
        
        if removed_count > 0:
            self.logger.info(
                "Expired audit events cleaned up",
                events_removed=removed_count,
                events_remaining=len(self._events)
            )
        
        return removed_count
    
    def export_events(
        self,
        format: str = "json",
        *,
        file_path: Optional[Path] = None,
        **filter_kwargs: Any
    ) -> Union[str, None]:
        """Export audit events in specified format.
        
        Args:
            format: Export format ("json", "csv", "xml")
            file_path: Optional file path to save export
            **filter_kwargs: Filtering parameters for get_events
            
        Returns:
            Formatted audit data as string, or None if saved to file
        """
        events = self.get_events(**filter_kwargs)
        
        if format.lower() == "json":
            import json
            data = [event.to_dict() for event in events]
            output = json.dumps(data, indent=2, default=str)
        
        elif format.lower() == "csv":
            import csv
            import io
            
            output_io = io.StringIO()
            if events:
                writer = csv.DictWriter(output_io, fieldnames=events[0].to_dict().keys())
                writer.writeheader()
                for event in events:
                    writer.writerow(event.to_dict())
            output = output_io.getvalue()
        
        elif format.lower() == "xml":
            output = '<?xml version="1.0" encoding="UTF-8"?>\n<audit_events>\n'
            for event in events:
                output += "  <event>\n"
                for key, value in event.to_dict().items():
                    output += f"    <{key}>{value}</{key}>\n"
                output += "  </event>\n"
            output += "</audit_events>\n"
        
        else:
            raise ValidationError(f"Unsupported export format: {format}")
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(output)
            
            self.log_event(
                event_type=AuditEventType.DATA_EXPORT,
                action="export_audit_events",
                resource=str(file_path),
                outcome="success",
                details={
                    "format": format,
                    "event_count": len(events),
                    "file_size": len(output)
                }
            )
            return None
        
        return output
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics.
        
        Returns:
            Dictionary containing audit statistics
        """
        if not self.retain_in_memory:
            return {"error": "Statistics not available - in-memory retention disabled"}
        
        total_events = len(self._events)
        if total_events == 0:
            return {"total_events": 0}
        
        # Count by event type
        type_counts = {}
        for event in self._events:
            type_name = event.event_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for event in self._events:
            severity_name = event.severity.value
            severity_counts[severity_name] = severity_counts.get(severity_name, 0) + 1
        
        # Count by actor
        actor_counts = {}
        for event in self._events:
            if event.actor:
                actor_counts[event.actor] = actor_counts.get(event.actor, 0) + 1
        
        # Time range
        timestamps = [
            datetime.fromisoformat(e.timestamp.replace('Z', '+00:00')) 
            for e in self._events
        ]
        
        return {
            "total_events": total_events,
            "time_range": {
                "earliest": min(timestamps).isoformat(),
                "latest": max(timestamps).isoformat(),
            },
            "event_types": type_counts,
            "severity_levels": severity_counts,
            "top_actors": dict(sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "compliance_frameworks": self._compliance_frameworks,
            "compliance_mode": self.compliance_mode,
        }
    
    def generate_compliance_report(
        self,
        framework: str,
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate compliance report for specific framework.
        
        Args:
            framework: Compliance framework name
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report dictionary
        """
        # Filter events for compliance framework
        events = self.get_events(start_time=start_date, end_time=end_date)
        compliance_events = [
            e for e in events 
            if framework in e.compliance_tags
        ]
        
        # Framework-specific analysis
        report = {
            "framework": framework,
            "report_period": {
                "start": start_date.isoformat() if start_date else "inception",
                "end": end_date.isoformat() if end_date else "present",
            },
            "total_events": len(compliance_events),
            "event_breakdown": {},
            "security_events": 0,
            "access_violations": 0,
            "configuration_changes": 0,
            "recommendations": [],
        }
        
        # Analyze events by type
        for event in compliance_events:
            event_type = event.event_type.value
            report["event_breakdown"][event_type] = report["event_breakdown"].get(event_type, 0) + 1
            
            # Count specific event categories
            if event.event_type in [AuditEventType.SECURITY_VIOLATION, AuditEventType.INTRUSION_ATTEMPT]:
                report["security_events"] += 1
            
            if event.outcome != "success" and "access" in event_type:
                report["access_violations"] += 1
            
            if event.event_type in [AuditEventType.CONFIG_UPDATE, AuditEventType.CONFIG_CREATE]:
                report["configuration_changes"] += 1
        
        # Generate framework-specific recommendations
        if framework.upper() == "SOX":
            if report["configuration_changes"] > 100:
                report["recommendations"].append(
                    "High number of configuration changes detected. Consider implementing change approval workflow."
                )
            if report["access_violations"] > 0:
                report["recommendations"].append(
                    "Access violations detected. Review user permissions and access controls."
                )
        
        elif framework.upper() == "GDPR":
            data_events = [e for e in compliance_events if "data" in e.event_type.value]
            if len(data_events) > 0:
                report["data_processing_events"] = len(data_events)
                report["recommendations"].append(
                    "Data processing events logged. Ensure data subject rights are protected."
                )
        
        return report
    
    def __len__(self) -> int:
        """Get number of audit events in memory."""
        return len(self._events) if self.retain_in_memory else 0
    
    def __repr__(self) -> str:
        """Return string representation of audit logger."""
        return (
            f"AuditLogger("
            f"name={self.name!r}, "
            f"events={len(self._events) if self.retain_in_memory else 'N/A'}, "
            f"compliance_mode={self.compliance_mode})"
        )