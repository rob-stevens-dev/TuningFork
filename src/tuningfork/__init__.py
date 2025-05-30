"""TuningFork - Enterprise Database Performance Optimization Platform.

TuningFork is a comprehensive database performance optimization platform that
combines rule-based heuristics with machine learning to automatically identify
and resolve database performance issues across multiple database platforms.

Modules:
    core: Core infrastructure and base classes
    config: Configuration management
    logging: Structured logging framework

Example:
    Basic usage of TuningFork components:
    
    >>> from tuningfork.core import BaseComponent, PluginManager
    >>> from tuningfork.config import ConfigManager
    >>> from tuningfork.logging import get_logger
    >>> 
    >>> # Initialize logging
    >>> logger = get_logger(__name__)
    >>> logger.info("TuningFork initialized", version="0.1.0")
    >>> 
    >>> # Initialize configuration
    >>> config_manager = ConfigManager()
    >>> await config_manager.load_from_file("config.yaml")
    >>> 
    >>> # Initialize plugin manager
    >>> plugin_manager = PluginManager()
    >>> await plugin_manager.load_plugins_from_directory("plugins/")
    >>> await plugin_manager.initialize_all()
"""

from . import core, config, logging

__version__ = "0.1.0"
__title__ = "TuningFork"
__description__ = "Enterprise Database Performance Optimization Platform"
__author__ = "TuningFork Team"
__license__ = "MIT"

__all__ = [
    "core",
    "config", 
    "logging",
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    "__license__",
]