"""Configuration models for TuningFork system.

This module defines Pydantic models for all configuration objects used
throughout the TuningFork system. These models provide validation, type
safety, and serialization capabilities.

Classes:
    BaseConfig: Base configuration class
    DatabaseConfig: Database connection configuration
    SSLConfig: SSL/TLS configuration
    PoolConfig: Connection pool configuration
    SystemConfig: System-wide configuration
    LoggingConfig: Logging configuration
    SecurityConfig: Security configuration
    PerformanceConfig: Performance tuning configuration

Example:
    >>> database_config = DatabaseConfig(
    ...     id="prod_db",
    ...     host="localhost",
    ...     port=5432,
    ...     database="production",
    ...     username="app_user",
    ...     password="secure_password"
    ... )
    >>> print(database_config.connection_string)
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, SecretStr, validator, root_validator
from pydantic import AnyUrl, EmailStr, PositiveInt, conint, constr

from ..core.exceptions import ValidationError
from ..core.utils import ValidationUtils


class BaseConfig(BaseModel):
    """Base configuration class with common functionality.
    
    This class provides the foundation for all configuration objects
    including validation, environment variable resolution, and serialization.
    
    Example:
        >>> class MyConfig(BaseConfig):
        ...     name: str
        ...     value: int = 42
    """
    
    class Config:
        # Pydantic configuration
        validate_assignment = True
        extra = "forbid"
        use_enum_values = True
        validate_all = True
        allow_population_by_field_name = True
        
        # JSON serialization
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None,
            Path: str,
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        }
    
    @root_validator(pre=True)
    @classmethod
    def resolve_environment_variables(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve environment variables in configuration values.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        
        Args:
            values: Configuration values
            
        Returns:
            Values with environment variables resolved
        """
        def resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                # Handle ${VAR_NAME} and ${VAR_NAME:default} patterns
                import re
                
                def replace_env_var(match):
                    var_spec = match.group(1)
                    if ":" in var_spec:
                        var_name, default = var_spec.split(":", 1)
                    else:
                        var_name, default = var_spec, ""
                    
                    return os.getenv(var_name, default)
                
                return re.sub(r'\$\{([^}]+)\}', replace_env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value
        
        return {key: resolve_value(value) for key, value in values.items()}
    
    def to_dict(self, *, mask_secrets: bool = True) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Args:
            mask_secrets: Whether to mask secret values
            
        Returns:
            Dictionary representation of configuration
        """
        data = self.dict()
        
        if mask_secrets:
            def mask_value(value: Any) -> Any:
                if isinstance(value, dict):
                    return {k: mask_value(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [mask_value(item) for item in value]
                elif isinstance(value, SecretStr):
                    return "***MASKED***"
                else:
                    return value
            
            data = mask_value(data)
        
        return data
    
    def update_from_dict(self, data: Dict[str, Any]) -> "BaseConfig":
        """Update configuration from dictionary.
        
        Args:
            data: Dictionary with updated values
            
        Returns:
            New configuration instance with updated values
        """
        current_data = self.dict()
        current_data.update(data)
        return self.__class__(**current_data)


class SSLConfig(BaseConfig):
    """SSL/TLS configuration.
    
    Defines SSL settings for secure database connections and API endpoints.
    
    Attributes:
        enabled: Whether SSL is enabled
        cert_file: Path to SSL certificate file
        key_file: Path to SSL private key file
        ca_file: Path to CA certificate file
        verify_mode: SSL certificate verification mode
        check_hostname: Whether to verify hostname
    """
    
    enabled: bool = Field(True, description="Enable SSL/TLS connections")
    cert_file: Optional[Path] = Field(None, description="SSL certificate file path")
    key_file: Optional[Path] = Field(None, description="SSL private key file path")
    ca_file: Optional[Path] = Field(None, description="CA certificate file path")
    verify_mode: Literal["none", "optional", "required"] = Field(
        "required", description="Certificate verification mode"
    )
    check_hostname: bool = Field(True, description="Verify SSL hostname")
    
    @validator("cert_file", "key_file", "ca_file")
    @classmethod
    def validate_ssl_files(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate SSL file paths exist if specified.
        
        Args:
            v: File path to validate
            
        Returns:
            Validated file path
            
        Raises:
            ValidationError: If file doesn't exist
        """
        if v is not None and not ValidationUtils.validate_file_path(v):
            raise ValidationError(f"SSL file not found: {v}")
        return v
    
    @root_validator
    @classmethod
    def validate_ssl_configuration(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SSL configuration consistency.
        
        Args:
            values: Configuration values
            
        Returns:
            Validated configuration values
            
        Raises:
            ValidationError: If configuration is inconsistent
        """
        enabled = values.get("enabled", True)
        cert_file = values.get("cert_file")
        key_file = values.get("key_file")
        
        if enabled and cert_file and not key_file:
            raise ValidationError("SSL key file required when certificate is provided")
        
        if enabled and key_file and not cert_file:
            raise ValidationError("SSL certificate file required when key is provided")
        
        return values


class PoolConfig(BaseConfig):
    """Connection pool configuration.
    
    Defines connection pooling settings for database connections.
    
    Attributes:
        min_size: Minimum number of connections in pool
        max_size: Maximum number of connections in pool
        max_overflow: Maximum overflow connections beyond max_size
        pool_timeout: Timeout for getting connection from pool
        pool_recycle: Time before connection is recycled
        pool_pre_ping: Whether to ping connections before use
    """
    
    min_size: PositiveInt = Field(1, description="Minimum pool size")
    max_size: PositiveInt = Field(10, description="Maximum pool size")
    max_overflow: conint(ge=0) = Field(5, description="Maximum overflow connections")
    pool_timeout: PositiveInt = Field(30, description="Pool timeout in seconds")
    pool_recycle: PositiveInt = Field(3600, description="Pool recycle time in seconds")
    pool_pre_ping: bool = Field(True, description="Enable pool pre-ping")
    
    @validator("max_size")
    @classmethod
    def validate_max_size(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure max_size >= min_size.
        
        Args:
            v: max_size value
            values: All configuration values
            
        Returns:
            Validated max_size
            
        Raises:
            ValidationError: If max_size < min_size
        """
        min_size = values.get("min_size", 1)
        if v < min_size:
            raise ValidationError(f"max_size ({v}) must be >= min_size ({min_size})")
        return v


class CredentialConfig(BaseConfig):
    """Credential configuration with secure handling.
    
    Handles sensitive credential information with proper masking.
    
    Attributes:
        username: Database username
        password: Database password (stored securely)
        auth_source: Authentication source/database
        auth_mechanism: Authentication mechanism
    """
    
    username: str = Field(..., min_length=1, description="Database username")
    password: SecretStr = Field(..., description="Database password")
    auth_source: Optional[str] = Field(None, description="Authentication source")
    auth_mechanism: Optional[str] = Field(None, description="Authentication mechanism")
    
    @validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format.
        
        Args:
            v: Username to validate
            
        Returns:
            Validated username
            
        Raises:
            ValidationError: If username format is invalid
        """
        if not v.strip():
            raise ValidationError("Username cannot be empty or whitespace")
        return v.strip()


class DatabaseConfig(BaseConfig):
    """Database connection configuration.
    
    Comprehensive configuration for database connections including
    connection parameters, SSL settings, and pool configuration.
    
    Attributes:
        id: Unique database identifier
        platform: Database platform (postgresql, mysql, etc.)
        host: Database host
        port: Database port
        database: Database name
        credentials: Database credentials
        ssl_config: SSL configuration
        pool_config: Connection pool configuration
        connection_timeout: Connection timeout in seconds
        query_timeout: Query timeout in seconds
        options: Additional connection options
        
    Example:
        >>> config = DatabaseConfig(
        ...     id="prod_postgres",
        ...     platform="postgresql",
        ...     host="db.example.com",
        ...     port=5432,
        ...     database="production",
        ...     credentials=CredentialConfig(
        ...         username="app_user",
        ...         password=SecretStr("secure_password")
        ...     )
        ... )
    """
    
    id: constr(min_length=1) = Field(..., description="Unique database identifier")
    platform: Literal["postgresql", "mysql", "mssql", "oracle", "sqlite"] = Field(
        ..., description="Database platform"
    )
    host: constr(min_length=1) = Field(..., description="Database host")
    port: PositiveInt = Field(..., description="Database port")
    database: constr(min_length=1) = Field(..., description="Database name")
    credentials: CredentialConfig = Field(..., description="Database credentials")
    ssl_config: SSLConfig = Field(default_factory=SSLConfig, description="SSL configuration")
    pool_config: PoolConfig = Field(default_factory=PoolConfig, description="Pool configuration")
    connection_timeout: PositiveInt = Field(30, description="Connection timeout in seconds")
    query_timeout: PositiveInt = Field(300, description="Query timeout in seconds")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")
    
    @validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate database ID format.
        
        Args:
            v: Database ID to validate
            
        Returns:
            Validated database ID
            
        Raises:
            ValidationError: If ID format is invalid
        """
        if not ValidationUtils.validate_identifier(v):
            raise ValidationError(f"Invalid database ID format: {v}")
        return v
    
    @validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host format.
        
        Args:
            v: Host to validate
            
        Returns:
            Validated host
            
        Raises:
            ValidationError: If host format is invalid
        """
        if not v.strip():
            raise ValidationError("Host cannot be empty")
        return v.strip()
    
    @validator("database")
    @classmethod
    def validate_database(cls, v: str) -> str:
        """Validate database name format.
        
        Args:
            v: Database name to validate
            
        Returns:
            Validated database name
            
        Raises:
            ValidationError: If database name format is invalid
        """
        if not ValidationUtils.validate_sql_identifier(v):
            raise ValidationError(f"Invalid database name format: {v}")
        return v
    
    @property
    def connection_string(self) -> str:
        """Generate connection string for database.
        
        Returns:
            Database connection string with masked password
        """
        if self.platform == "postgresql":
            return (
                f"postgresql://{self.credentials.username}:***@"
                f"{self.host}:{self.port}/{self.database}"
            )
        elif self.platform == "mysql":
            return (
                f"mysql://{self.credentials.username}:***@"
                f"{self.host}:{self.port}/{self.database}"
            )
        elif self.platform == "mssql":
            return (
                f"mssql://{self.credentials.username}:***@"
                f"{self.host}:{self.port}/{self.database}"
            )
        elif self.platform == "sqlite":
            return f"sqlite:///{self.database}"
        else:
            return f"{self.platform}://{self.host}:{self.port}/{self.database}"


class LoggingConfig(BaseConfig):
    """Logging configuration.
    
    Defines logging behavior including levels, formats, and output destinations.
    
    Attributes:
        level: Log level
        format: Log format (json, text)
        file_path: Log file path
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        console_output: Enable console output
        structured: Enable structured logging
        filters: Log filters configuration
    """
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level"
    )
    format: Literal["json", "text"] = Field("json", description="Log format")
    file_path: Optional[Path] = Field(None, description="Log file path")
    max_file_size: PositiveInt = Field(10485760, description="Max file size in bytes (10MB)")
    backup_count: conint(ge=0) = Field(5, description="Number of backup files")
    console_output: bool = Field(True, description="Enable console output")
    structured: bool = Field(True, description="Enable structured logging")
    filters: Dict[str, str] = Field(default_factory=dict, description="Log filters")
    
    @validator("file_path")
    @classmethod
    def validate_log_file_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate log file path is writable.
        
        Args:
            v: Log file path
            
        Returns:
            Validated log file path
            
        Raises:
            ValidationError: If path is not writable
        """
        if v is not None:
            # Ensure parent directory exists
            parent_dir = v.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise ValidationError(f"Cannot create log directory: {e}")
        
        return v


class SecurityConfig(BaseConfig):
    """Security configuration.
    
    Defines security settings including authentication, authorization,
    and encryption parameters.
    
    Attributes:
        secret_key: Application secret key
        api_key_length: API key length
        session_timeout: Session timeout in seconds
        password_min_length: Minimum password length
        require_ssl: Require SSL for connections
        allowed_hosts: List of allowed hosts
        cors_origins: CORS allowed origins
    """
    
    secret_key: SecretStr = Field(..., description="Application secret key")
    api_key_length: conint(ge=16, le=128) = Field(32, description="API key length")
    session_timeout: PositiveInt = Field(3600, description="Session timeout in seconds")
    password_min_length: conint(ge=8, le=128) = Field(12, description="Minimum password length")
    require_ssl: bool = Field(True, description="Require SSL connections")
    allowed_hosts: List[str] = Field(default_factory=list, description="Allowed hosts")
    cors_origins: List[str] = Field(default_factory=list, description="CORS origins")
    
    @validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: SecretStr) -> SecretStr:
        """Validate secret key strength.
        
        Args:
            v: Secret key to validate
            
        Returns:
            Validated secret key
            
        Raises:
            ValidationError: If secret key is too weak
        """
        key_value = v.get_secret_value()
        if len(key_value) < 32:
            raise ValidationError("Secret key must be at least 32 characters long")
        return v


class PerformanceConfig(BaseConfig):
    """Performance configuration.
    
    Defines performance-related settings for the TuningFork system.
    
    Attributes:
        max_concurrent_operations: Maximum concurrent operations
        operation_timeout: Default operation timeout
        cache_ttl: Default cache TTL
        batch_size: Default batch size for operations
        memory_limit: Memory limit in bytes
        cpu_limit: CPU limit percentage
    """
    
    max_concurrent_operations: PositiveInt = Field(
        10, description="Maximum concurrent operations"
    )
    operation_timeout: PositiveInt = Field(300, description="Operation timeout in seconds")
    cache_ttl: PositiveInt = Field(3600, description="Cache TTL in seconds")
    batch_size: PositiveInt = Field(100, description="Default batch size")
    memory_limit: Optional[PositiveInt] = Field(None, description="Memory limit in bytes")
    cpu_limit: Optional[conint(ge=1, le=100)] = Field(None, description="CPU limit percentage")


class SystemConfig(BaseConfig):
    """System-wide configuration.
    
    Main configuration object that combines all other configuration types
    for comprehensive system configuration.
    
    Attributes:
        app_name: Application name
        version: Application version
        environment: Deployment environment
        debug: Debug mode flag
        logging: Logging configuration
        security: Security configuration
        performance: Performance configuration
        databases: Database configurations
        plugins: Plugin configuration
        
    Example:
        >>> system_config = SystemConfig(
        ...     app_name="TuningFork",
        ...     version="1.0.0",
        ...     environment="production",
        ...     databases={
        ...         "primary": DatabaseConfig(...)
        ...     }
        ... )
    """
    
    app_name: str = Field("TuningFork", description="Application name")
    version: constr(regex=r"^\d+\.\d+\.\d+") = Field(..., description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        "development", description="Deployment environment"
    )
    debug: bool = Field(False, description="Enable debug mode")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging config")
    security: SecurityConfig = Field(..., description="Security configuration")
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance config"
    )
    databases: Dict[str, DatabaseConfig] = Field(
        default_factory=dict, description="Database configurations"
    )
    plugins: Dict[str, Any] = Field(
        default_factory=dict, description="Plugin configurations"
    )
    
    @validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting.
        
        Args:
            v: Environment value
            
        Returns:
            Validated environment
        """
        return v.lower()
    
    @validator("databases")
    @classmethod
    def validate_databases(cls, v: Dict[str, DatabaseConfig]) -> Dict[str, DatabaseConfig]:
        """Validate database configurations.
        
        Args:
            v: Database configurations
            
        Returns:
            Validated database configurations
            
        Raises:
            ValidationError: If database IDs don't match keys
        """
        for key, db_config in v.items():
            if db_config.id != key:
                raise ValidationError(
                    f"Database config ID '{db_config.id}' doesn't match key '{key}'"
                )
        
        return v
    
    @root_validator
    @classmethod
    def validate_environment_specific_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate environment-specific configuration.
        
        Args:
            values: Configuration values
            
        Returns:
            Validated configuration values
            
        Raises:
            ValidationError: If configuration is invalid for environment
        """
        environment = values.get("environment", "development")
        debug = values.get("debug", False)
        security = values.get("security")
        
        # Production environment validations
        if environment == "production":
            if debug:
                raise ValidationError("Debug mode cannot be enabled in production")
            
            if security and not security.require_ssl:
                raise ValidationError("SSL must be required in production environment")
        
        return values
    
    def get_database_config(self, database_id: str) -> Optional[DatabaseConfig]:
        """Get database configuration by ID.
        
        Args:
            database_id: Database identifier
            
        Returns:
            Database configuration or None if not found
        """
        return self.databases.get(database_id)
    
    def add_database_config(self, config: DatabaseConfig) -> None:
        """Add database configuration.
        
        Args:
            config: Database configuration to add
            
        Raises:
            ValidationError: If database ID already exists
        """
        if config.id in self.databases:
            raise ValidationError(f"Database configuration already exists: {config.id}")
        
        self.databases[config.id] = config
    
    def remove_database_config(self, database_id: str) -> bool:
        """Remove database configuration.
        
        Args:
            database_id: Database identifier to remove
            
        Returns:
            True if configuration was removed
        """
        if database_id in self.databases:
            del self.databases[database_id]
            return True
        return False
    
    def get_plugin_config(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get plugin configuration.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin configuration or None if not found
        """
        return self.plugins.get(plugin_id)
    
    def set_plugin_config(self, plugin_id: str, config: Dict[str, Any]) -> None:
        """Set plugin configuration.
        
        Args:
            plugin_id: Plugin identifier
            config: Plugin configuration
        """
        self.plugins[plugin_id] = config