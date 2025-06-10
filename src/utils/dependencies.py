"""
Dependency Management Module.

This module centralizes all optional dependency imports and availability checks.
It provides a consistent interface for managing both required and optional dependencies.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import importlib
import pkg_resources

# Configure logging
logger = logging.getLogger(__name__)

class DependencyError(Exception):
    """Raised when a required dependency is missing."""
    pass

class VersionError(Exception):
    """Raised when a dependency version is incompatible."""
    pass

class DependencyManager:
    """Manages all project dependencies with version requirements."""
    
    # Map PyPI package name to (import name, version)
    REQUIRED_DEPENDENCIES = {
        'pandas':    ('pandas', '>=1.3.0'),
        'numpy':     ('numpy', '>=1.20.0'),
        'scikit-learn': ('sklearn', '>=0.24.0'),  # PyPI name: scikit-learn, import: sklearn
    }
    
    OPTIONAL_DEPENDENCIES = {
        'transformers': {
            'version': '>=4.0.0',
            'components': ['pipeline'],
            'purpose': 'Multilingual sentiment analysis'
        },
        'vaderSentiment': {
            'version': '>=3.3.0',
            'components': ['SentimentIntensityAnalyzer'],
            'purpose': 'English sentiment analysis',
            'import_path': 'vaderSentiment.vaderSentiment'
        },
        'textblob': {
            'version': '>=0.15.0',
            'components': ['TextBlob'],
            'purpose': 'Fallback sentiment analysis'
        },
        'nrclex': {
            'version': '>=0.0.6',
            'components': ['NRCLex'],
            'purpose': 'Emotion analysis'
        }
    }
    
    def __init__(self):
        """Initialize the dependency manager."""
        self._imported_modules: Dict[str, Any] = {}
        self._import_errors: Dict[str, str] = {}
        self._check_required_dependencies()
        self._init_optional_dependencies()
    
    def _check_required_dependencies(self):
        """Verify all required dependencies are available with correct versions."""
        for pypi_name, (import_name, version) in self.REQUIRED_DEPENDENCIES.items():
            try:
                pkg_resources.require(f"{pypi_name}{version}")
                self._imported_modules[import_name] = importlib.import_module(import_name)
            except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound) as e:
                raise DependencyError(f"Required dependency {pypi_name} {version} not satisfied: {str(e)}")
    
    def _init_optional_dependencies(self):
        """Initialize optional dependencies."""
        for package, info in self.OPTIONAL_DEPENDENCIES.items():
            try:
                pkg_resources.require(f"{package}{info['version']}")
                self._try_import(package, info['components'], f"{info['purpose']} will not be available", info.get('import_path'))
            except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound) as e:
                logger.warning(f"Optional dependency {package} {info['version']} not satisfied: {str(e)}")
                self._import_errors[package] = str(e)
    
    def _try_import(self, package: str, components: List[str], warning_msg: str, import_path: str = None):
        """Attempt to import a package and its components."""
        try:
            if import_path:
                module = importlib.import_module(import_path)
            else:
                module = importlib.import_module(package)
            self._imported_modules[package] = module
            
            # Verify components are available
            for component in components:
                if not hasattr(module, component):
                    raise ImportError(f"Missing required component: {component}")
                    
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not import {package}: {str(e)}. {warning_msg}")
            self._import_errors[package] = str(e)
    
    @property
    def transformers_available(self) -> bool:
        """Check if transformers is available."""
        return 'transformers' in self._imported_modules
    
    @property
    def vader_available(self) -> bool:
        """Check if VADER is available."""
        return 'vaderSentiment' in self._imported_modules
    
    @property
    def textblob_available(self) -> bool:
        """Check if TextBlob is available."""
        return 'textblob' in self._imported_modules
    
    @property
    def nrclex_available(self) -> bool:
        """Check if NRCLex is available."""
        return 'nrclex' in self._imported_modules
    
    def get_module(self, package: str) -> Optional[Any]:
        """Get the imported module if available."""
        return self._imported_modules.get(package)
    
    def get_error(self, package: str) -> Optional[str]:
        """Get the error message if import failed."""
        return self._import_errors.get(package)
    
    def get_component(self, package: str, component: str) -> Any:
        """Get a component from an imported package.
        
        Args:
            package: The package name
            component: The component name
            
        Returns:
            The requested component
            
        Raises:
            DependencyError: If the package or component is not available
        """
        if package not in self._imported_modules:
            raise DependencyError(f"Package {package} not available")
        
        module = self._imported_modules[package]
        if not hasattr(module, component):
            raise DependencyError(f"Component {component} not found in {package}")
        
        return getattr(module, component)

# Create singleton instance
dependency_manager = DependencyManager()
