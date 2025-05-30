"""Unit tests for TuningFork base classes.

This module tests the base component classes including lifecycle management,
configuration handling, and component registry functionality.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any

from tuningfork.core.base import (
    BaseComponent,
    AsyncComponent,
    ConfigurableComponent,
    LifecycleComponent,
    SingletonComponent,
    ComponentRegistry,
    get_global_registry,
    register_component,
    get_component,
)
from tuningfork.core.exceptions import (
    TuningForkException,
    ValidationError,
    ConfigurationError,
)


# Test configuration class - NOT a test class (no Test prefix)
class ComponentTestConfig:
    """Test configuration for component testing."""
    def __init__(self, name: str = "test", value: int = 42):
        self.name = name
        self.value = value


# Test component implementations - avoid pytest collection with underscore prefix
class _TestableBaseComponent(BaseComponent[ComponentTestConfig]):
    """Test implementation of BaseComponent."""
    component_name = "TestComponent"
    version = "1.0.0"


class _TestableAsyncComponent(AsyncComponent[ComponentTestConfig]):
    """Test implementation of AsyncComponent."""
    component_name = "TestAsyncComponent"
    version = "1.0.0"
    
    def __init__(self, config: ComponentTestConfig):
        super().__init__(config)
        self.initialized_called = False
        self.cleanup_called = False
    
    async def _async_initialize(self) -> None:
        self.initialized_called = True
    
    async def _async_cleanup(self) -> None:
        self.cleanup_called = True


class _TestableLifecycleComponent(LifecycleComponent[ComponentTestConfig]):
    """Test implementation of LifecycleComponent."""
    component_name = "TestLifecycleComponent"
    version = "1.0.0"
    
    def __init__(self, config: ComponentTestConfig):
        super().__init__(config)
        self.paused_called = False
        self.resumed_called = False
    
    async def _async_initialize(self) -> None:
        pass
    
    async def _async_pause(self) -> None:
        self.paused_called = True
    
    async def _async_resume(self) -> None:
        self.resumed_called = True


class _TestableSingletonComponent(SingletonComponent[ComponentTestConfig]):
    """Test implementation of SingletonComponent."""
    component_name = "TestSingletonComponent"
    version = "1.0.0"


class TestBaseComponent:
    """Test BaseComponent functionality."""
    
    def test_component_initialization(self):
        """Test basic component initialization."""
        config = ComponentTestConfig(name="test_component")
        component = _TestableBaseComponent(config)
        
        assert component.config == config
        assert not component.is_initialized
        assert component.component_name == "TestComponent"
        assert component.version == "1.0.0"
        assert component.uptime >= 0
    
    def test_component_initialization_with_none_config(self):
        """Test component initialization with None config raises error."""
        with pytest.raises(ValidationError) as exc_info:
            _TestableBaseComponent(None)
        
        assert exc_info.value.code == "CONFIG_NULL"
    
    def test_component_validation(self):
        """Test component configuration validation."""
        config = ComponentTestConfig()
        component = _TestableBaseComponent(config)
        
        assert component.validate_config()
    
    def test_component_health_status(self):
        """Test component health status reporting."""
        config = ComponentTestConfig()
        component = _TestableBaseComponent(config)
        
        health = component.get_health_status()
        
        assert health["component"] == "TestComponent"
        assert health["version"] == "1.0.0"
        assert health["initialized"] is False
        assert "uptime_seconds" in health
        assert health["status"] == "not_initialized"
    
    def test_component_metrics(self):
        """Test component metrics reporting."""
        config = ComponentTestConfig()
        component = _TestableBaseComponent(config)
        
        metrics = component.get_metrics()
        
        assert metrics["component"] == "TestComponent"
        assert "uptime_seconds" in metrics
        assert metrics["initialized"] is False
    
    def test_component_repr(self):
        """Test component string representation."""
        config = ComponentTestConfig()
        component = _TestableBaseComponent(config)
        
        repr_str = repr(component)
        
        assert "_TestableBaseComponent" in repr_str
        assert "TestComponent" in repr_str
        assert "initialized=False" in repr_str


class TestConfigurableComponent:
    """Test ConfigurableComponent functionality."""
    
    def test_configurable_component_initialization(self):
        """Test configurable component initialization."""
        config = ComponentTestConfig(name="configurable")
        component = ConfigurableComponent(config)
        
        assert component.config == config
        assert component.config_version == 1
        assert len(component._config_history) == 1
    
    def test_config_update(self):
        """Test configuration update."""
        config = ComponentTestConfig(name="original")
        component = ConfigurableComponent(config)
        
        new_config = ComponentTestConfig(name="updated")
        component.update_config(new_config)
        
        assert component.config == new_config
        assert component.config_version == 2
        assert len(component._config_history) == 2
    
    def test_config_update_with_validation_disabled(self):
        """Test configuration update without validation."""
        config = ComponentTestConfig(name="original")
        component = ConfigurableComponent(config)
        
        new_config = ComponentTestConfig(name="updated")
        component.update_config(new_config, validate=False)
        
        assert component.config == new_config
    
    def test_config_rollback(self):
        """Test configuration rollback."""
        config = ComponentTestConfig(name="original")
        component = ConfigurableComponent(config)
        
        # Update config
        new_config = ComponentTestConfig(name="updated")
        component.update_config(new_config)
        
        # Rollback
        success = component.rollback_config()
        
        assert success
        assert component.config == config
        assert component.config_version == 3
    
    def test_config_rollback_without_history(self):
        """Test configuration rollback without sufficient history."""
        config = ComponentTestConfig(name="original")
        component = ConfigurableComponent(config)
        
        # Try to rollback without updates
        success = component.rollback_config()
        
        assert not success
        assert component.config == config


class TestAsyncComponent:
    """Test AsyncComponent functionality."""
    
    @pytest.mark.asyncio
    async def test_async_component_initialization(self):
        """Test async component initialization."""
        config = ComponentTestConfig(name="async_test")
        component = _TestableAsyncComponent(config)
        
        assert not component.is_initialized
        assert not component.initialized_called
        
        await component.initialize()
        
        assert component.is_initialized
        assert component.initialized_called
    
    @pytest.mark.asyncio
    async def test_async_component_double_initialization(self):
        """Test async component handles double initialization."""
        config = ComponentTestConfig()
        component = _TestableAsyncComponent(config)
        
        await component.initialize()
        await component.initialize()  # Should not reinitialize
        
        assert component.is_initialized
        assert component.initialized_called
    
    @pytest.mark.asyncio
    async def test_async_component_cleanup(self):
        """Test async component cleanup."""
        config = ComponentTestConfig()
        component = _TestableAsyncComponent(config)
        
        await component.initialize()
        await component.cleanup()
        
        assert not component.is_initialized
        assert component.cleanup_called
    
    @pytest.mark.asyncio
    async def test_async_component_cleanup_without_init(self):
        """Test async component cleanup without initialization."""
        config = ComponentTestConfig()
        component = _TestableAsyncComponent(config)
        
        await component.cleanup()  # Should not fail
        
        assert not component.is_initialized
        assert not component.cleanup_called
    
    @pytest.mark.asyncio
    async def test_async_component_context_manager(self):
        """Test async component as context manager."""
        config = ComponentTestConfig()
        component = _TestableAsyncComponent(config)
        
        async with component as ctx_component:
            assert ctx_component is component
            assert component.is_initialized
            assert component.initialized_called
        
        assert not component.is_initialized
        assert component.cleanup_called
    
    @pytest.mark.asyncio
    async def test_async_component_managed_lifecycle(self):
        """Test async component managed lifecycle."""
        config = ComponentTestConfig()
        component = _TestableAsyncComponent(config)
        
        async with component.managed_lifecycle() as managed_component:
            assert managed_component is component
            assert component.is_initialized
        
        assert not component.is_initialized
        assert component.cleanup_called
    
    @pytest.mark.asyncio
    async def test_async_component_initialization_error(self):
        """Test async component initialization error handling."""
        class FailingAsyncComponent(AsyncComponent[ComponentTestConfig]):
            component_name = "FailingComponent"
            
            async def _async_initialize(self) -> None:
                raise RuntimeError("Initialization failed")
        
        config = ComponentTestConfig()
        component = FailingAsyncComponent(config)
        
        with pytest.raises(TuningForkException) as exc_info:
            await component.initialize()
        
        assert exc_info.value.code == "INIT_FAILED"
        assert not component.is_initialized


class TestLifecycleComponent:
    """Test LifecycleComponent functionality."""
    
    @pytest.mark.asyncio
    async def test_lifecycle_component_states(self):
        """Test lifecycle component state transitions."""
        config = ComponentTestConfig()
        component = _TestableLifecycleComponent(config)
        
        assert component.state == "created"
        
        await component.initialize()
        assert component.state == "running"
        
        await component.pause()
        assert component.state == "paused"
        assert component.paused_called
        
        await component.resume()
        assert component.state == "running"
        assert component.resumed_called
        
        await component.cleanup()
        assert component.state == "stopped"
    
    @pytest.mark.asyncio
    async def test_lifecycle_component_invalid_state_transitions(self):
        """Test invalid state transitions raise errors."""
        config = ComponentTestConfig()
        component = _TestableLifecycleComponent(config)
        
        # Try to pause before initialization
        with pytest.raises(TuningForkException) as exc_info:
            await component.pause()
        
        assert exc_info.value.code == "INVALID_STATE"
        
        await component.initialize()
        
        # Try to resume without pause
        with pytest.raises(TuningForkException) as exc_info:
            await component.resume()
        
        assert exc_info.value.code == "INVALID_STATE"
    
    @pytest.mark.asyncio
    async def test_lifecycle_component_state_history(self):
        """Test lifecycle component state history tracking."""
        config = ComponentTestConfig()
        component = _TestableLifecycleComponent(config)
        
        initial_history = component.state_history
        assert len(initial_history) == 1
        assert initial_history[0][0] == "created"
        
        await component.initialize()
        
        history = component.state_history
        assert len(history) >= 3  # created, initializing, running
        assert history[-1][0] == "running"
    
    def test_lifecycle_component_health_status_with_state(self):
        """Test lifecycle component health status includes state."""
        config = ComponentTestConfig()
        component = _TestableLifecycleComponent(config)
        
        health = component.get_health_status()
        
        assert "state" in health
        assert health["state"] == "created"
        assert "state_history" in health


class TestSingletonComponent:
    """Test SingletonComponent functionality."""
    
    def test_singleton_same_instance(self):
        """Test singleton returns same instance for same config."""
        config = ComponentTestConfig(name="singleton")
        
        component1 = _TestableSingletonComponent(config)
        component2 = _TestableSingletonComponent(config)
        
        assert component1 is component2
    
    def test_singleton_different_configs(self):
        """Test singleton creates different instances for different configs."""
        config1 = ComponentTestConfig(name="singleton1")
        config2 = ComponentTestConfig(name="singleton2")
        
        component1 = _TestableSingletonComponent(config1)
        component2 = _TestableSingletonComponent(config2)
        
        assert component1 is not component2
    
    def test_singleton_get_instance(self):
        """Test singleton get_instance method."""
        config = ComponentTestConfig(name="singleton")
        
        # No instance exists yet
        instance = _TestableSingletonComponent.get_instance(config)
        assert instance is None
        
        # Create instance
        component = _TestableSingletonComponent(config)
        
        # Get existing instance
        instance = _TestableSingletonComponent.get_instance(config)
        assert instance is component
    
    def test_singleton_clear_instances(self):
        """Test singleton clear_instances method."""
        config = ComponentTestConfig(name="singleton")
        component = _TestableSingletonComponent(config)
        
        # Clear all instances
        _TestableSingletonComponent.clear_instances()
        
        # Get instance should return None
        instance = _TestableSingletonComponent.get_instance(config)
        assert instance is None


class TestComponentRegistry:
    """Test ComponentRegistry functionality."""
    
    def test_registry_register_component(self):
        """Test component registration."""
        registry = ComponentRegistry()
        config = ComponentTestConfig(name="registry_test")
        component = _TestableBaseComponent(config)
        
        registry.register("test_component", component)
        
        retrieved = registry.get("test_component")
        assert retrieved is component
    
    def test_registry_register_duplicate_name(self):
        """Test registering component with duplicate name raises error."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        component1 = _TestableBaseComponent(config)
        component2 = _TestableBaseComponent(config)
        
        registry.register("test_component", component1)
        
        with pytest.raises(ValidationError) as exc_info:
            registry.register("test_component", component2)
        
        assert exc_info.value.code == "COMPONENT_EXISTS"
    
    def test_registry_unregister_component(self):
        """Test component unregistration."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        component = _TestableBaseComponent(config)
        
        registry.register("test_component", component)
        registry.unregister("test_component")
        
        with pytest.raises(ValidationError):
            registry.get("test_component")
    
    def test_registry_unregister_nonexistent_component(self):
        """Test unregistering non-existent component raises error."""
        registry = ComponentRegistry()
        
        with pytest.raises(ValidationError) as exc_info:
            registry.unregister("nonexistent")
        
        assert exc_info.value.code == "COMPONENT_NOT_FOUND"
    
    def test_registry_get_all_components(self):
        """Test getting all registered components."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        component1 = _TestableBaseComponent(config)
        component2 = _TestableBaseComponent(config)
        
        registry.register("component1", component1)
        registry.register("component2", component2)
        
        all_components = registry.get_all()
        
        assert len(all_components) == 2
        assert all_components["component1"] is component1
        assert all_components["component2"] is component2
    
    def test_registry_initialization_order_simple(self):
        """Test initialization order calculation without dependencies."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        component1 = _TestableBaseComponent(config)
        component2 = _TestableBaseComponent(config)
        
        registry.register("component1", component1)
        registry.register("component2", component2)
        
        order = registry.get_initialization_order()
        
        assert len(order) == 2
        assert set(order) == {"component1", "component2"}
    
    def test_registry_initialization_order_with_dependencies(self):
        """Test initialization order calculation with dependencies."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        component1 = _TestableBaseComponent(config)
        component2 = _TestableBaseComponent(config)
        component3 = _TestableBaseComponent(config)
        
        # component2 depends on component1
        # component3 depends on component2
        registry.register("component1", component1, dependencies=[])
        registry.register("component2", component2, dependencies=["component1"])
        registry.register("component3", component3, dependencies=["component2"])
        
        order = registry.get_initialization_order()
        
        assert order.index("component1") < order.index("component2")
        assert order.index("component2") < order.index("component3")
    
    def test_registry_circular_dependency_detection(self):
        """Test circular dependency detection."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        component1 = _TestableBaseComponent(config)
        component2 = _TestableBaseComponent(config)
        
        # Create circular dependency
        registry.register("component1", component1, dependencies=["component2"])
        registry.register("component2", component2, dependencies=["component1"])
        
        with pytest.raises(ValidationError) as exc_info:
            registry.get_initialization_order()
        
        assert exc_info.value.code == "CIRCULAR_DEPENDENCY"
    
    @pytest.mark.asyncio
    async def test_registry_initialize_all(self):
        """Test initializing all components."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        component1 = _TestableAsyncComponent(config)
        component2 = _TestableAsyncComponent(config)
        
        registry.register("async1", component1)
        registry.register("async2", component2)
        
        await registry.initialize_all()
        
        assert component1.is_initialized
        assert component2.is_initialized
    
    @pytest.mark.asyncio
    async def test_registry_cleanup_all(self):
        """Test cleaning up all components."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        component1 = _TestableAsyncComponent(config)
        component2 = _TestableAsyncComponent(config)
        
        registry.register("async1", component1)
        registry.register("async2", component2)
        
        await registry.initialize_all()
        await registry.cleanup_all()
        
        assert not component1.is_initialized
        assert not component2.is_initialized
        assert component1.cleanup_called
        assert component2.cleanup_called
    
    def test_registry_health_status(self):
        """Test registry health status reporting."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        component1 = _TestableBaseComponent(config)
        component2 = _TestableBaseComponent(config)
        
        registry.register("component1", component1)
        registry.register("component2", component2)
        
        health = registry.get_health_status()
        
        assert "component1" in health
        assert "component2" in health
        assert health["component1"]["component"] == "TestComponent"
        assert health["component2"]["component"] == "TestComponent"


class TestGlobalRegistry:
    """Test global registry functions."""
    
    def test_get_global_registry(self):
        """Test getting global registry instance."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        
        assert registry1 is registry2
    
    def test_register_component_global(self):
        """Test registering component with global registry."""
        config = ComponentTestConfig()
        component = _TestableBaseComponent(config)
        
        register_component("global_test", component)
        
        retrieved = get_component("global_test")
        assert retrieved is component
        
        # Clean up
        global_registry = get_global_registry()
        try:
            global_registry.unregister("global_test")
        except ValidationError:
            pass  # May not exist in clean test environment
    
    def test_get_component_global(self):
        """Test getting component from global registry."""
        config = ComponentTestConfig()
        component = _TestableBaseComponent(config)
        
        register_component("global_get_test", component)
        
        retrieved = get_component("global_get_test")
        assert retrieved is component
        
        # Clean up
        global_registry = get_global_registry()
        try:
            global_registry.unregister("global_get_test")
        except ValidationError:
            pass


class TestComponentIntegration:
    """Test integration between different component types."""
    
    @pytest.mark.asyncio
    async def test_mixed_component_types_in_registry(self):
        """Test registry with mixed component types."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        base_component = _TestableBaseComponent(config)
        async_component = _TestableAsyncComponent(config)
        lifecycle_component = _TestableLifecycleComponent(config)
        
        registry.register("base", base_component)
        registry.register("async", async_component)
        registry.register("lifecycle", lifecycle_component)
        
        # Initialize all - should handle different types correctly
        await registry.initialize_all()
        
        assert async_component.is_initialized
        assert lifecycle_component.is_initialized
        assert lifecycle_component.state == "running"
        
        # Cleanup all
        await registry.cleanup_all()
        
        assert not async_component.is_initialized
        assert not lifecycle_component.is_initialized
    
    @pytest.mark.asyncio
    async def test_component_dependency_chain(self):
        """Test complex dependency chain initialization."""
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        # Create dependency chain: comp1 -> comp2 -> comp3
        comp1 = _TestableAsyncComponent(config)
        comp2 = _TestableAsyncComponent(config)
        comp3 = _TestableAsyncComponent(config)
        
        registry.register("comp1", comp1)
        registry.register("comp2", comp2, dependencies=["comp1"])
        registry.register("comp3", comp3, dependencies=["comp2"])
        
        await registry.initialize_all()
        
        # All should be initialized
        assert comp1.is_initialized
        assert comp2.is_initialized
        assert comp3.is_initialized
        
        await registry.cleanup_all()
        
        # All should be cleaned up
        assert not comp1.is_initialized
        assert not comp2.is_initialized
        assert not comp3.is_initialized


class TestComponentErrorHandling:
    """Test component error handling scenarios."""
    
    def test_component_validation_error(self):
        """Test component with validation error."""
        class ValidatingComponent(BaseComponent[ComponentTestConfig]):
            component_name = "ValidatingComponent"
            
            def validate_config(self) -> bool:
                return self._config.name != "invalid"
        
        # Valid config should work
        valid_config = ComponentTestConfig(name="valid")
        component = ValidatingComponent(valid_config)
        assert component.validate_config()
        
        # Invalid config should raise error
        invalid_config = ComponentTestConfig(name="invalid")
        with pytest.raises(ConfigurationError):
            ValidatingComponent(invalid_config)
    
    @pytest.mark.asyncio
    async def test_async_component_cleanup_error(self):
        """Test async component handles cleanup errors gracefully."""
        class FailingCleanupComponent(AsyncComponent[ComponentTestConfig]):
            component_name = "FailingCleanupComponent"
            
            async def _async_initialize(self) -> None:
                pass
            
            async def _async_cleanup(self) -> None:
                raise RuntimeError("Cleanup failed")
        
        config = ComponentTestConfig()
        component = FailingCleanupComponent(config)
        
        await component.initialize()
        
        # Cleanup should not raise exception but should handle error gracefully
        await component.cleanup()
        
        # Component should still be marked as not initialized despite cleanup error
        # This is the expected behavior - cleanup errors don't prevent state change
        assert not component.is_initialized
    
    @pytest.mark.asyncio
    async def test_registry_partial_initialization_failure(self):
        """Test registry handles partial initialization failure."""
        class FailingInitComponent(AsyncComponent[ComponentTestConfig]):
            component_name = "FailingInitComponent"
            
            async def _async_initialize(self) -> None:
                raise RuntimeError("Init failed")
        
        registry = ComponentRegistry()
        config = ComponentTestConfig()
        
        good_component = _TestableAsyncComponent(config)
        bad_component = FailingInitComponent(config)
        
        registry.register("good", good_component)
        registry.register("bad", bad_component)
        
        # Should raise exception for failing component
        with pytest.raises(TuningForkException):
            await registry.initialize_all()
        
        # Good component might be initialized depending on order
        # Bad component should not be initialized
        assert not bad_component.is_initialized


@pytest.mark.parametrize("component_class", [
    _TestableBaseComponent,
    _TestableAsyncComponent,
    _TestableLifecycleComponent,
])
def test_component_uptime_tracking(component_class):
    """Test that all component types track uptime."""
    config = ComponentTestConfig()
    component = component_class(config)
    
    import time
    time.sleep(0.01)  # Small delay
    
    assert component.uptime > 0


@pytest.mark.parametrize("component_class,expected_name", [
    (_TestableBaseComponent, "TestComponent"),
    (_TestableAsyncComponent, "TestAsyncComponent"),
    (_TestableLifecycleComponent, "TestLifecycleComponent"),
])
def test_component_metadata(component_class, expected_name):
    """Test component metadata is correctly set."""
    config = ComponentTestConfig()
    component = component_class(config)
    
    assert component.component_name == expected_name
    assert component.version == "1.0.0"