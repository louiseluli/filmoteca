"""
============================================================================
FILMOTECA - Configuration Manager
============================================================================
This module loads and validates all configuration from .env file.
Provides type-safe access to settings throughout the application.

🔧 USAGE:
    from config import config
    
    # Access settings with autocomplete and type checking
    api_key = config.tmdb_api_key
    data_dir = config.data_dir
    weights = config.recommendation_weights

🔧 CUSTOMIZE: 
    - Add new settings in the appropriate Config class section
    - Update validation logic in validators as needed
    - Modify default values to match your preferences

📝 FEATURES:
    - Automatic .env loading
    - Type validation with Pydantic
    - Helpful error messages for missing/invalid settings
    - Organized by functional area
    - Environment-aware (development/production)
============================================================================
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv
import sys


# ============================================================================
# FIND AND LOAD .env FILE
# ============================================================================
# This searches for .env file starting from current directory up to project root

def find_dotenv() -> Optional[Path]:
    """
    Find .env file by searching up the directory tree.
    
    Returns:
        Path to .env file if found, None otherwise
    
    🔧 CUSTOMIZE: Add additional search locations if needed
    """
    current = Path.cwd()
    
    # Search up to 5 levels up
    for _ in range(5):
        env_file = current / ".env"
        if env_file.exists():
            return env_file
        
        # Stop at root directory
        if current.parent == current:
            break
            
        current = current.parent
    
    return None


# Load environment variables
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    print("⚠️  No .env file found. Using environment variables or defaults.")


# ============================================================================
# API CONFIGURATION
# ============================================================================
# Settings for external API services (TMDB, OMDB, Does the Dog Die)

class APIConfig(BaseModel):
    """
    API authentication and rate limiting configuration.
    
    🔧 CUSTOMIZE: Add new API services here
    """
    
    # TMDB (The Movie Database)
    tmdb_api_key: str = Field(
        default="",
        description="TMDB API key for v3 endpoints"
    )
    tmdb_read_token: str = Field(
        default="",
        description="TMDB Read Access Token for v4 endpoints"
    )
    tmdb_rate_limit: int = Field(
        default=40,
        description="TMDB requests per 10 seconds"
    )
    tmdb_base_url: str = Field(
        default="https://api.themoviedb.org/3",
        description="TMDB API base URL"
    )
    
    # OMDB (Open Movie Database)
    omdb_api_key: str = Field(
        default="",
        description="OMDB API key"
    )
    omdb_rate_limit: int = Field(
        default=1000,
        description="OMDB requests per day"
    )
    omdb_base_url: str = Field(
        default="http://www.omdbapi.com",
        description="OMDB API base URL"
    )
    
    # Does the Dog Die
    ddd_api_key: str = Field(
        default="",
        description="Does the Dog Die API key"
    )
    ddd_base_url: str = Field(
        default="https://www.doesthedogdie.com/api",
        description="DDD API base URL"
    )
    
    # General API settings
    parallel_workers: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of concurrent API requests"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed requests"
    )
    retry_delay: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
        description="Delay in seconds between retries"
    )
    timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    
    @field_validator('tmdb_api_key', 'omdb_api_key')
    @classmethod
    def validate_api_key_format(cls, v: str, info) -> str:
        """
        Validate API key is not the placeholder value.
        
        🔧 CUSTOMIZE: Add more specific validation if needed
        """
        if v and v.startswith('your_') and v.endswith('_here'):
            field_name = info.field_name.upper()
            print(f"⚠️  {field_name} is placeholder value. Enrichment will be skipped.")
        return v
    
    def has_tmdb(self) -> bool:
        """Check if TMDB API is configured."""
        return bool(self.tmdb_api_key and not self.tmdb_api_key.startswith('your_'))
    
    def has_omdb(self) -> bool:
        """Check if OMDB API is configured."""
        return bool(self.omdb_api_key and not self.omdb_api_key.startswith('your_'))
    
    def has_ddd(self) -> bool:
        """Check if Does the Dog Die API is configured."""
        return bool(self.ddd_api_key and not self.ddd_api_key.startswith('your_'))
    
    class Config:
        """Pydantic configuration."""
        extra = 'ignore'  # Ignore extra fields


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
# Settings for database connections (SQLite, PostgreSQL)

class DatabaseConfig(BaseModel):
    """
    Database connection and configuration.
    
    🔧 CUSTOMIZE: Modify for different database backends
    """
    
    database_url: str = Field(
        default="sqlite:///data/raw/imdb.db",
        description="Database connection URL"
    )
    
    echo: bool = Field(
        default=False,
        description="Echo SQL queries (useful for debugging)"
    )
    
    pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Connection pool size (PostgreSQL only)"
    )
    
    max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Max overflow connections (PostgreSQL only)"
    )
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self.database_url.startswith('sqlite')
    
    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return self.database_url.startswith('postgresql')
    
    @property
    def database_path(self) -> Optional[Path]:
        """Get database file path for SQLite."""
        if self.is_sqlite:
            # Extract path from sqlite:///path/to/db.db
            path_str = self.database_url.replace('sqlite:///', '')
            return Path(path_str)
        return None
    
    class Config:
        """Pydantic configuration."""
        extra = 'ignore'


# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
# File and directory paths for data, logs, reports

class PathsConfig(BaseModel):
    """
    Project directory structure and file paths.
    
    🔧 CUSTOMIZE: Adjust paths to match your preferred structure
    """
    
    # Base directories
    data_dir: Path = Field(
        default=Path("./data"),
        description="Main data directory"
    )
    raw_data_dir: Path = Field(
        default=Path("./data/raw"),
        description="Raw data (IMDb, Watched.csv)"
    )
    processed_data_dir: Path = Field(
        default=Path("./data/processed"),
        description="Processed and enriched data"
    )
    reports_dir: Path = Field(
        default=Path("./data/reports"),
        description="Analysis reports and visualizations"
    )
    logs_dir: Path = Field(
        default=Path("./data/logs"),
        description="Application logs"
    )
    
    # Input files
    watchlist_file: Path = Field(
        default=Path("./data/raw/Watched.csv"),
        description="Your IMDb watched history"
    )
    ground_truth_file: Optional[Path] = Field(
        default=Path("./data/ground_truth_verified.csv"),
        description="Ground truth for validation (optional)"
    )
    imdb_database: Path = Field(
        default=Path("./data/raw/imdb.db"),
        description="SQLite database for IMDb data"
    )
    
    # IMDb dataset files
    imdb_dir: Path = Field(
        default=Path("./data/raw/imdb"),
        description="Directory for IMDb TSV files"
    )
    
    @model_validator(mode='after')
    def create_directories(self) -> 'PathsConfig':
        """
        Create directories if they don't exist.
        
        🔧 CUSTOMIZE: Add any additional directories to create
        """
        directories = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.reports_dir,
            self.reports_dir / "eda",
            self.logs_dir,
            self.imdb_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        return self
    
    def get_imdb_file(self, filename: str) -> Path:
        """
        Get path to IMDb dataset file.
        
        Args:
            filename: IMDb dataset filename (e.g., 'title.basics.tsv.gz')
        
        Returns:
            Full path to the file
        
        🔧 USAGE:
            path = config.paths.get_imdb_file('title.basics.tsv.gz')
        """
        return self.imdb_dir / filename
    
    class Config:
        """Pydantic configuration."""
        extra = 'ignore'


# ============================================================================
# RECOMMENDATION ENGINE CONFIGURATION
# ============================================================================
# Settings for movie recommendation algorithms

class RecommendationConfig(BaseModel):
    """
    Recommendation engine parameters and weights.
    
    🔧 CUSTOMIZE: Adjust these to tune recommendations to your taste
    """
    
    # Filtering thresholds
    min_rating_threshold: float = Field(
        default=6.0,
        ge=0.0,
        le=10.0,
        description="Minimum rating to consider (1-10 scale)"
    )
    
    # Output settings
    top_n_recommendations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of recommendations to generate"
    )
    
    # Similarity settings
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1)"
    )
    
    # Feature weights (must sum to 1.0)
    weight_genre: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Genre similarity weight"
    )
    weight_director: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Director similarity weight"
    )
    weight_actors: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Actor similarity weight"
    )
    weight_keywords: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Plot keywords similarity weight"
    )
    weight_year: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Release year proximity weight"
    )
    weight_rating: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Rating similarity weight"
    )
    weight_embedding: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Semantic embedding similarity weight"
    )
    
    @model_validator(mode='after')
    def validate_weights_sum(self) -> 'RecommendationConfig':
        """
        Ensure all weights sum to 1.0.
        
        🔧 IMPORTANT: Weights must sum to 1.0 for proper normalization
        """
        total = (
            self.weight_genre +
            self.weight_director +
            self.weight_actors +
            self.weight_keywords +
            self.weight_year +
            self.weight_rating +
            self.weight_embedding
        )
        
        if abs(total - 1.0) > 0.01:  # Allow 1% tolerance
            raise ValueError(
                f"Recommendation weights must sum to 1.0, got {total:.3f}. "
                f"Please adjust WEIGHT_* values in .env file."
            )
        
        return self
    
    @property
    def weights_dict(self) -> Dict[str, float]:
        """Get all weights as a dictionary."""
        return {
            'genre': self.weight_genre,
            'director': self.weight_director,
            'actors': self.weight_actors,
            'keywords': self.weight_keywords,
            'year': self.weight_year,
            'rating': self.weight_rating,
            'embedding': self.weight_embedding,
        }
    
    class Config:
        """Pydantic configuration."""
        extra = 'ignore'


# ============================================================================
# PROCESSING CONFIGURATION
# ============================================================================
# Settings for data processing and performance

class ProcessingConfig(BaseModel):
    """
    Data processing and performance settings.
    
    🔧 CUSTOMIZE: Adjust for your hardware and needs
    """
    
    # Batch processing
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Movies per batch"
    )
    
    # Caching
    enable_cache: bool = Field(
        default=True,
        description="Enable result caching"
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=0,
        le=8760,  # 1 year
        description="Cache time-to-live in hours"
    )
    
    # Export settings
    export_format: str = Field(
        default="json",
        description="Export format: json, parquet, or csv"
    )
    include_raw_data: bool = Field(
        default=False,
        description="Include raw IMDb data in exports"
    )
    compress_exports: bool = Field(
        default=True,
        description="Compress exported files"
    )
    
    @field_validator('export_format')
    @classmethod
    def validate_export_format(cls, v: str) -> str:
        """Validate export format is supported."""
        allowed = ['json', 'parquet', 'csv']
        if v.lower() not in allowed:
            raise ValueError(f"export_format must be one of {allowed}, got '{v}'")
        return v.lower()
    
    class Config:
        """Pydantic configuration."""
        extra = 'ignore'


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Settings for application logging

class LoggingConfig(BaseModel):
    """
    Logging configuration.
    
    🔧 CUSTOMIZE: Adjust log levels and formats
    """
    
    log_level: str = Field(
        default="INFO",
        description="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    log_file: Path = Field(
        default=Path("./data/logs/filmoteca.log"),
        description="Log file path"
    )
    
    # Console logging
    console_output: bool = Field(
        default=True,
        description="Print logs to console"
    )
    
    # Log format
    log_format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Log message format"
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Log timestamp format"
    )
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v_upper
    
    class Config:
        """Pydantic configuration."""
        extra = 'ignore'


# ============================================================================
# MAIN CONFIGURATION
# ============================================================================
# Central configuration object combining all settings

class Config(BaseModel):
    """
    Main configuration class combining all settings.
    
    🔧 USAGE:
        from config import config
        
        # Access nested settings
        api_key = config.api.tmdb_api_key
        data_dir = config.paths.data_dir
        weights = config.recommendation.weights_dict
    
    🔧 CUSTOMIZE: Add new configuration sections here
    """
    
    # Configuration sections
    api: APIConfig
    database: DatabaseConfig
    paths: PathsConfig
    recommendation: RecommendationConfig
    processing: ProcessingConfig
    logging: LoggingConfig
    
    # Environment
    environment: str = Field(
        default="development",
        description="Environment: development, production, testing"
    )
    
    # Project metadata
    project_name: str = Field(
        default="Filmoteca",
        description="Project name"
    )
    version: str = Field(
        default="0.1.0",
        description="Project version"
    )
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"
    
    def print_summary(self):
        """
        Print configuration summary.
        
        🔧 USAGE: Call this to verify your configuration loaded correctly
            from config import config
            config.print_summary()
        """
        print("\n" + "="*70)
        print(f"🎬 {self.project_name} v{self.version} - Configuration Summary")
        print("="*70)
        
        print(f"\n📍 Environment: {self.environment.upper()}")
        print(f"📂 Data Directory: {self.paths.data_dir.absolute()}")
        print(f"📊 Watchlist File: {self.paths.watchlist_file.absolute()}")
        
        print("\n🔑 API Keys:")
        print(f"  • TMDB: {'✅ Configured' if self.api.has_tmdb() else '❌ Missing'}")
        print(f"  • OMDB: {'✅ Configured' if self.api.has_omdb() else '❌ Missing'}")
        print(f"  • Does the Dog Die: {'✅ Configured' if self.api.has_ddd() else '❌ Missing'}")
        
        print("\n💾 Database:")
        print(f"  • Type: {'SQLite' if self.database.is_sqlite else 'PostgreSQL'}")
        print(f"  • URL: {self.database.database_url}")
        
        print("\n🎯 Recommendation Settings:")
        print(f"  • Min Rating: {self.recommendation.min_rating_threshold}")
        print(f"  • Top N: {self.recommendation.top_n_recommendations}")
        print(f"  • Weights: Genre={self.recommendation.weight_genre}, "
              f"Director={self.recommendation.weight_director}, "
              f"Actors={self.recommendation.weight_actors}")
        
        print("\n⚡ Processing:")
        print(f"  • Batch Size: {self.processing.batch_size}")
        print(f"  • Cache: {'Enabled' if self.processing.enable_cache else 'Disabled'}")
        print(f"  • Export Format: {self.processing.export_format.upper()}")
        
        print("\n📝 Logging:")
        print(f"  • Level: {self.logging.log_level}")
        print(f"  • File: {self.logging.log_file.absolute()}")
        
        print("\n" + "="*70 + "\n")
    
    class Config:
        """Pydantic configuration."""
        extra = 'ignore'


# ============================================================================
# LOAD CONFIGURATION
# ============================================================================
# Load settings from environment variables

def load_config() -> Config:
    """
    Load configuration from environment variables.
    
    Returns:
        Configured Config object
    
    Raises:
        ValueError: If configuration is invalid
    
    🔧 CUSTOMIZE: Add custom validation or post-processing here
    """
    
    def get_env(key: str, default: Any = None) -> Any:
        """Get environment variable with fallback."""
        return os.getenv(key, default)
    
    try:
        # Load each configuration section
        config_obj = Config(
            api=APIConfig(
                tmdb_api_key=get_env('TMDB_API_KEY', ''),
                tmdb_read_token=get_env('TMDB_READ_TOKEN', ''),
                tmdb_rate_limit=int(get_env('TMDB_RATE_LIMIT', 40)),
                omdb_api_key=get_env('OMDB_API_KEY', ''),
                omdb_rate_limit=int(get_env('OMDB_RATE_LIMIT', 1000)),
                ddd_api_key=get_env('DDD_API_KEY', ''),
                parallel_workers=int(get_env('PARALLEL_WORKERS', 5)),
                max_retries=int(get_env('MAX_RETRIES', 3)),
                retry_delay=float(get_env('RETRY_DELAY', 2.0)),
            ),
            database=DatabaseConfig(
                database_url=get_env('DATABASE_URL', 'sqlite:///data/raw/imdb.db'),
            ),
            paths=PathsConfig(
                data_dir=Path(get_env('DATA_DIR', './data')),
                raw_data_dir=Path(get_env('RAW_DATA_DIR', './data/raw')),
                processed_data_dir=Path(get_env('PROCESSED_DATA_DIR', './data/processed')),
                reports_dir=Path(get_env('REPORTS_DIR', './data/reports')),
                logs_dir=Path(get_env('LOGS_DIR', './data/logs')),
                watchlist_file=Path(get_env('WATCHLIST_FILE', './data/raw/Watched.csv')),
                imdb_database=Path(get_env('IMDB_DATABASE', './data/raw/imdb.db')),
            ),
            recommendation=RecommendationConfig(
                min_rating_threshold=float(get_env('MIN_RATING_THRESHOLD', 6.0)),
                top_n_recommendations=int(get_env('TOP_N_RECOMMENDATIONS', 20)),
                similarity_threshold=float(get_env('SIMILARITY_THRESHOLD', 0.3)),
                weight_genre=float(get_env('WEIGHT_GENRE', 0.25)),
                weight_director=float(get_env('WEIGHT_DIRECTOR', 0.15)),
                weight_actors=float(get_env('WEIGHT_ACTORS', 0.15)),
                weight_keywords=float(get_env('WEIGHT_KEYWORDS', 0.20)),
                weight_year=float(get_env('WEIGHT_YEAR', 0.10)),
                weight_rating=float(get_env('WEIGHT_RATING', 0.10)),
                weight_embedding=float(get_env('WEIGHT_EMBEDDING', 0.05)),
            ),
            processing=ProcessingConfig(
                batch_size=int(get_env('BATCH_SIZE', 100)),
                enable_cache=get_env('ENABLE_CACHE', 'True').lower() == 'true',
                cache_ttl_hours=int(get_env('CACHE_TTL_HOURS', 24)),
                export_format=get_env('EXPORT_FORMAT', 'json'),
                include_raw_data=get_env('INCLUDE_RAW_DATA', 'False').lower() == 'true',
                compress_exports=get_env('COMPRESS_EXPORTS', 'True').lower() == 'true',
            ),
            logging=LoggingConfig(
                log_level=get_env('LOG_LEVEL', 'INFO'),
                log_file=Path(get_env('LOG_FILE', './data/logs/filmoteca.log')),
            ),
            environment=get_env('ENVIRONMENT', 'development'),
        )
        
        return config_obj
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("Please check your .env file and ensure all values are valid.")
        sys.exit(1)


# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================
# Single configuration instance used throughout the application

# Load configuration on module import
config = load_config()

# Print summary if running as main script
if __name__ == "__main__":
    config.print_summary()


# ============================================================================
# 🔧 USAGE EXAMPLES
# ============================================================================
"""
# Example 1: Import and use configuration
from config import config

# Access API keys
if config.api.has_tmdb():
    print(f"Using TMDB API: {config.api.tmdb_base_url}")

# Access paths
watchlist_path = config.paths.watchlist_file
print(f"Loading watchlist from: {watchlist_path}")

# Access recommendation settings
min_rating = config.recommendation.min_rating_threshold
weights = config.recommendation.weights_dict
print(f"Recommending movies with rating >= {min_rating}")
print(f"Using weights: {weights}")

# Check environment
if config.is_development:
    print("Running in development mode")


# Example 2: Print full configuration summary
from config import config
config.print_summary()


# Example 3: Access nested settings
from config import config

# Get all recommendation weights
for feature, weight in config.recommendation.weights_dict.items():
    print(f"{feature}: {weight}")

# Check which APIs are available
apis = []
if config.api.has_tmdb():
    apis.append("TMDB")
if config.api.has_omdb():
    apis.append("OMDB")
if config.api.has_ddd():
    apis.append("Does the Dog Die")

print(f"Available APIs: {', '.join(apis)}")


# Example 4: Validate configuration
from config import config

# Configuration is automatically validated on load
# If there are errors, the program will exit with a helpful message
# You can add custom validation:

if not config.paths.watchlist_file.exists():
    print(f"⚠️  Watchlist file not found: {config.paths.watchlist_file}")
    print("Please add your Watched.csv file to the data/raw directory")
"""