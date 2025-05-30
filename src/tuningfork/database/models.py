# src/tuningfork/database/models.py
"""Database models for TuningFork Phase 2."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class QueryResult:
    """Standardized query result across all database platforms."""
    rows: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    execution_time: float
    query_plan: Optional[Dict[str, Any]] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class ColumnInfo:
    """Database column information."""
    name: str
    data_type: str
    is_nullable: bool
    default_value: Optional[Any]
    max_length: Optional[int]
    precision: Optional[int]
    scale: Optional[int]
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_unique: bool = False
    comment: Optional[str] = None


@dataclass
class IndexInfo:
    """Database index information."""
    schema_name: str
    table_name: str
    index_name: str
    index_type: str  # 'btree', 'hash', 'gist', etc.
    is_unique: bool
    is_primary: bool
    columns: List[str]
    included_columns: List[str] = None
    filter_condition: Optional[str] = None
    size_bytes: Optional[int] = None
    usage_stats: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.included_columns is None:
            self.included_columns = []


@dataclass
class TableInfo:
    """Database table information."""
    schema_name: str
    table_name: str
    table_type: str  # 'table', 'view', 'materialized_view'
    columns: List[ColumnInfo]
    indexes: List[IndexInfo]
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    data_size_bytes: Optional[int] = None
    index_size_bytes: Optional[int] = None
    comment: Optional[str] = None
    created_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None


@dataclass
class DatabaseMetadata:
    """Complete database metadata."""
    platform: str
    version: str
    server_edition: Optional[str]
    current_database: str
    
    # Schema objects
    schemas: List[str]
    tables: List[TableInfo] 
    views: List[TableInfo]
    indexes: List[IndexInfo]
    
    # Configuration
    configuration: Dict[str, Any]
    
    # Extraction metadata
    extraction_timestamp: datetime
    extraction_duration: float
    extraction_warnings: List[str] = None
    
    def __post_init__(self):
        if self.extraction_warnings is None:
            self.extraction_warnings = []


@dataclass
class PerformanceStats:
    """Database performance statistics."""
    timestamp: datetime
    connections_active: int
    connections_total: int
    queries_per_second: float
    transactions_per_second: float
    cache_hit_ratio: float
    buffer_pool_hit_ratio: Optional[float] = None
    lock_waits: int = 0
    deadlocks: int = 0
    
    # Resource utilization
    cpu_usage_percent: Optional[float] = None
    memory_usage_bytes: Optional[int] = None
    disk_io_read_mbps: Optional[float] = None
    disk_io_write_mbps: Optional[float] = None