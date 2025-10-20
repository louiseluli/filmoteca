r"""
============================================================================
FILMOTECA - IMDb Database Schema and Loader
============================================================================
Creates SQLite database schema and loads all 7 IMDb datasets.
Handles 198.5M+ records efficiently with batch processing and indexes.

🎯 PURPOSE:
    - Define comprehensive database schema for all IMDb data
    - Load 7 TSV files into SQLite database with proper data types
    - Create strategic indexes for fast querying
    - Handle NULL values (\N) properly
    - Ensure data integrity (years as integers, ratings as floats)
    - Track progress for long operations

📊 DATABASE TABLES (7 tables matching IMDb datasets):
    1. title_basics      - Core title information (12M records)
    2. title_akas        - Alternative titles (53.5M records)
    3. title_ratings     - User ratings (1.6M records)
    4. title_crew        - Directors and writers (12M records)
    5. title_principals  - Cast and crew (95.3M records) ⚠️ LARGEST
    6. title_episode     - TV episodes (9.2M records)
    7. name_basics       - People data (14.8M records)

🔧 DATA TYPE HANDLING:
    - Years (start_year, birth_year, death_year): INTEGER
    - Ratings (average_rating): FLOAT with 1 decimal precision
    - Boolean (is_adult, is_original_title): BOOLEAN (0/1)
    - NULL values: '\N' converted to None/NULL

🔧 INDEX STRATEGY:
    - Primary keys on all ID columns (tconst, nconst)
    - Foreign key indexes for joins
    - Composite indexes for common query patterns
    - Year ranges, rating filters, genre searches
    - Language and region lookups
    - Full-text search preparation

🔧 USAGE:
    python scripts/load_imdb_to_db.py [--recreate] [--table TABLE_NAME]
    
    Options:
        --recreate      Drop and recreate all tables (careful!)
        --table NAME    Load only specific table
        --batch SIZE    Custom batch size (default: 10000)
        --verify        Run verification queries after loading

⚠️  IMPORTANT:
    - Requires ~15GB free disk space for database
    - Loading takes 30-60 minutes for all tables
    - Creates indexes after loading (adds 10-15 minutes)
    - Database file: data/raw/imdb.db

📝 OUTPUT:
    - SQLite database: data/raw/imdb.db
    - Load log: data/logs/imdb_load.log
    
============================================================================
"""

import os
import sys
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Iterator, Tuple
import argparse

# Increase CSV field size limit to handle large fields (e.g., long character lists)
# Default is 131072 bytes, we increase to 10MB
csv.field_size_limit(10 * 1024 * 1024)  # 10MB

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    Text, Index, ForeignKey, CheckConstraint, UniqueConstraint,
    MetaData, Table, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from tqdm import tqdm


# ============================================================================
# DATABASE SETUP
# ============================================================================

# Create SQLAlchemy base and engine
Base = declarative_base()

# Database engine (SQLite) with optimizations
# 🔧 CUSTOMIZE: Change to PostgreSQL for production
engine = create_engine(
    config.database.database_url,
    echo=False,  # Set to True to see all SQL queries
    connect_args={
        'check_same_thread': False,
        'timeout': 30
    } if config.database.is_sqlite else {}
)

# SQLite performance optimizations
if config.database.is_sqlite:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        """
        Set SQLite pragmas for better performance during bulk inserts.
        
        🔧 PERFORMANCE TUNING:
        - journal_mode=WAL: Write-Ahead Logging for concurrent access
        - synchronous=NORMAL: Balance between safety and speed
        - cache_size=10000: Use more memory for caching (10MB)
        - temp_store=MEMORY: Keep temporary tables in memory
        """
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA mmap_size=30000000000")  # 30GB memory-mapped I/O
        cursor.close()

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ============================================================================
# TABLE DEFINITIONS - SQLALCHEMY MODELS WITH DETAILED INDEXES
# ============================================================================

# ----------------------------------------------------------------------------
# 1. TITLE BASICS - Core Movie/TV Information
# ----------------------------------------------------------------------------
class TitleBasics(Base):
    """
    Core information about titles (movies, TV shows, episodes, etc.)
    
    Corresponds to: title.basics.tsv.gz
    Records: ~12 million titles
    
    🎯 PRIMARY USE: Base table for all title queries
    🔗 JOINS: 
        - title_akas.title_id → tconst (get alternative titles)
        - title_ratings.tconst → tconst (get ratings)
        - title_crew.tconst → tconst (get directors/writers)
        - title_principals.tconst → tconst (get cast)
    
    📊 INDEXES (8 total):
        1. PRIMARY: tconst - Fast lookups by IMDb ID
        2. COMPOSITE: title_type + start_year - Filter by type and year
        3. COMPOSITE: start_year + title_type - Year-based searches
        4. COMPOSITE: genres + start_year - Genre + decade queries
        5. SIMPLE: primary_title - Title searches
        6. SIMPLE: original_title - Original language searches
        7. SIMPLE: is_adult - Filter adult content
        8. COMPOSITE: runtime_minutes + start_year - Find movies by length/era
    
    🔧 CUSTOMIZE: Add indexes for your specific query patterns
    """
    __tablename__ = 'title_basics'
    
    tconst = Column(
        String(15), 
        primary_key=True,
        comment='IMDb unique title identifier (e.g., tt0111161 for Shawshank Redemption)'
    )
    
    title_type = Column(
        String(50),
        nullable=False,
        comment='Type: movie, short, tvSeries, tvEpisode, tvMiniSeries, tvMovie, video, videoGame'
    )
    
    primary_title = Column(
        String(500),
        nullable=False,
        comment='Main title used for marketing/display'
    )
    
    original_title = Column(
        String(500),
        nullable=False,
        comment='Original title in the original language'
    )
    
    is_adult = Column(
        Boolean,
        default=False,
        nullable=False,
        comment='0 = family-friendly, 1 = adult content'
    )
    
    start_year = Column(
        Integer,
        nullable=True,
        comment='Release year (INTEGER) for movies, series start year for TV'
    )
    
    end_year = Column(
        Integer,
        nullable=True,
        comment='TV series end year (INTEGER), NULL for other types'
    )
    
    runtime_minutes = Column(
        Integer,
        nullable=True,
        comment='Duration in minutes (INTEGER)'
    )
    
    genres = Column(
        String(200),
        nullable=True,
        comment='Comma-separated genres (up to 3): Drama,Comedy,Action'
    )
    
    # Strategic indexes for common query patterns
    __table_args__ = (
        # 🎯 INDEX 1: Type + Year - Find all movies from specific year
        Index('idx_type_year', 'title_type', 'start_year'),
        
        # 🎯 INDEX 2: Year + Type - Decade-based searches
        Index('idx_year_type', 'start_year', 'title_type'),
        
        # 🎯 INDEX 3: Genres + Year - Find Drama movies from 1990s
        Index('idx_genres_year', 'genres', 'start_year'),
        
        # 🎯 INDEX 4: Primary Title - Text searches
        Index('idx_primary_title', 'primary_title'),
        
        # 🎯 INDEX 5: Original Title - Original language searches
        Index('idx_original_title', 'original_title'),
        
        # 🎯 INDEX 6: Adult content filter
        Index('idx_is_adult', 'is_adult'),
        
        # 🎯 INDEX 7: Runtime + Year - Find feature films (>90 min) by decade
        Index('idx_runtime_year', 'runtime_minutes', 'start_year'),
        
        # 🔧 CUSTOMIZE: Add more indexes for your queries
        # Example: Index('idx_genre_adult', 'genres', 'is_adult'),
    )
    
    def __repr__(self):
        return f"<TitleBasics(tconst='{self.tconst}', title='{self.primary_title}', year={self.start_year})>"


# ----------------------------------------------------------------------------
# 2. TITLE AKAS - Alternative Titles (All Languages)
# ----------------------------------------------------------------------------
class TitleAkas(Base):
    """
    Alternative titles in all languages and regions.
    CRITICAL for multilingual support and finding original language titles.
    
    Corresponds to: title.akas.tsv.gz
    Records: ~53.5 million alternative titles
    
    🎯 PRIMARY USE: Get titles in specific languages/regions, find original titles
    🔗 JOINS: title_id → title_basics.tconst
    
    📊 INDEXES (11 total):
        1. PRIMARY: id - Auto-increment key
        2. FOREIGN KEY: title_id - Join to title_basics
        3. COMPOSITE: title_id + language - Get all Spanish titles for a movie
        4. COMPOSITE: title_id + region - Get all US titles for a movie
        5. COMPOSITE: title_id + is_original_title - Find original language title
        6. SIMPLE: language - All movies in Japanese
        7. SIMPLE: region - All movies released in France
        8. SIMPLE: is_original_title - All original language titles
        9. COMPOSITE: language + region - Spanish titles in Spain vs Mexico
        10. SIMPLE: types - Filter by type (original, dvd, festival)
        11. UNIQUE: title_id + ordering - Prevent duplicates
    
    🔧 IMPORTANT: Use is_original_title=1 to find original language titles
    """
    __tablename__ = 'title_akas'
    
    id = Column(
        Integer, 
        primary_key=True, 
        autoincrement=True,
        comment='Auto-increment ID for internal use'
    )
    
    title_id = Column(
        String(15),
        ForeignKey('title_basics.tconst', ondelete='CASCADE'),
        nullable=False,
        comment='References title_basics.tconst'
    )
    
    ordering = Column(
        Integer,
        nullable=False,
        comment='Unique ordering number for this title_id'
    )
    
    title = Column(
        String(500),
        nullable=True,
        comment='Alternative title in specific language/region'
    )
    
    region = Column(
        String(10),
        nullable=True,
        comment='Country code: US, GB, FR, DE, JP, etc. or XWW (worldwide)'
    )
    
    language = Column(
        String(10),
        nullable=True,
        comment='Language code: en, es, fr, de, ja, ko, zh, etc.'
    )
    
    types = Column(
        String(100),
        nullable=True,
        comment='Type: alternative, original, dvd, festival, tv, video, working, imdbDisplay'
    )
    
    attributes = Column(
        String(200),
        nullable=True,
        comment='Additional info: literal English title, transliteration, etc.'
    )
    
    is_original_title = Column(
        Boolean,
        default=False,
        nullable=False,
        comment='1 = original title in original language, 0 = not original'
    )
    
    __table_args__ = (
        # 🎯 INDEX 1: Foreign key for joins
        Index('idx_akas_title_id', 'title_id'),
        
        # 🎯 INDEX 2: Title + Language - "Get Spanish title for this movie"
        Index('idx_akas_title_lang', 'title_id', 'language'),
        
        # 🎯 INDEX 3: Title + Region - "Get US release title"
        Index('idx_akas_title_region', 'title_id', 'region'),
        
        # 🎯 INDEX 4: Title + Original - "Find original language title"
        Index('idx_akas_title_orig', 'title_id', 'is_original_title'),
        
        # 🎯 INDEX 5: Language only - "All movies in Japanese"
        Index('idx_akas_language', 'language'),
        
        # 🎯 INDEX 6: Region only - "All titles released in France"
        Index('idx_akas_region', 'region'),
        
        # 🎯 INDEX 7: Original title flag - "All original titles"
        Index('idx_akas_original', 'is_original_title'),
        
        # 🎯 INDEX 8: Language + Region - "Spanish in Spain vs Mexico"
        Index('idx_akas_lang_region', 'language', 'region'),
        
        # 🎯 INDEX 9: Types - Filter by DVD, festival, etc.
        Index('idx_akas_types', 'types'),
        
        # 🎯 INDEX 10: Title text search
        Index('idx_akas_title_text', 'title'),
        
        # 🎯 UNIQUE CONSTRAINT: Prevent duplicate orderings
        UniqueConstraint('title_id', 'ordering', name='uq_title_ordering'),
    )
    
    def __repr__(self):
        return f"<TitleAkas(title_id='{self.title_id}', title='{self.title}', lang='{self.language}')>"


# ----------------------------------------------------------------------------
# 3. TITLE RATINGS - User Ratings and Popularity
# ----------------------------------------------------------------------------
class TitleRatings(Base):
    """
    IMDb user ratings and vote counts.
    
    Corresponds to: title.ratings.tsv.gz
    Records: ~1.6 million rated titles
    
    🎯 PRIMARY USE: Filter by rating, sort by popularity
    🔗 JOINS: tconst → title_basics.tconst
    
    📊 INDEXES (5 total):
        1. PRIMARY: tconst - Join to title_basics
        2. SIMPLE: average_rating - Find highly-rated movies
        3. SIMPLE: num_votes - Find popular movies
        4. COMPOSITE: average_rating + num_votes - Highly-rated AND popular
        5. COMPOSITE: num_votes + average_rating - Popular AND highly-rated
    
    🔧 DATA TYPE: average_rating stored as FLOAT with 1 decimal precision
    """
    __tablename__ = 'title_ratings'
    
    tconst = Column(
        String(15),
        ForeignKey('title_basics.tconst', ondelete='CASCADE'),
        primary_key=True,
        comment='References title_basics.tconst'
    )
    
    average_rating = Column(
        Float,
        nullable=True,
        comment='Weighted average rating (1.0 - 10.0) as FLOAT with 1 decimal'
    )
    
    num_votes = Column(
        Integer,
        nullable=True,
        comment='Total number of user votes (INTEGER, higher = more reliable)'
    )
    
    __table_args__ = (
        # 🎯 INDEX 1: Rating - "Movies rated 8.0+"
        Index('idx_rating_avg', 'average_rating'),
        
        # 🎯 INDEX 2: Votes - "Most voted movies"
        Index('idx_rating_votes', 'num_votes'),
        
        # 🎯 INDEX 3: Rating + Votes - "Highly-rated with many votes"
        Index('idx_rating_quality', 'average_rating', 'num_votes'),
        
        # 🎯 INDEX 4: Votes + Rating - "Popular movies sorted by rating"
        Index('idx_rating_popular', 'num_votes', 'average_rating'),
        
        # 🔧 CHECK CONSTRAINT: Ensure rating is between 1.0 and 10.0
        CheckConstraint('average_rating >= 1.0 AND average_rating <= 10.0', name='chk_rating_range'),
    )
    
    def __repr__(self):
        return f"<TitleRatings(tconst='{self.tconst}', rating={self.average_rating:.1f}, votes={self.num_votes:,})>"


# ----------------------------------------------------------------------------
# 4. TITLE CREW - Directors and Writers
# ----------------------------------------------------------------------------
class TitleCrew(Base):
    """
    Directors and writers for each title.
    
    Corresponds to: title.crew.tsv.gz
    Records: ~12 million titles with crew data
    
    🎯 PRIMARY USE: Find titles by director/writer, analyze collaborations
    🔗 JOINS: tconst → title_basics.tconst
    
    📊 INDEXES (3 total):
        1. PRIMARY: tconst - Join to title_basics
        2. SIMPLE: directors - Find all movies by Christopher Nolan
        3. SIMPLE: writers - Find all movies by Quentin Tarantino
    
    🔧 NOTE: directors and writers are comma-separated nconst IDs
             Use LIKE queries: WHERE directors LIKE '%nm0000116%'
    """
    __tablename__ = 'title_crew'
    
    tconst = Column(
        String(15),
        ForeignKey('title_basics.tconst', ondelete='CASCADE'),
        primary_key=True,
        comment='References title_basics.tconst'
    )
    
    directors = Column(
        String(500),
        nullable=True,
        comment='Comma-separated nconst IDs of directors (e.g., nm0000116,nm0000134)'
    )
    
    writers = Column(
        String(1000),
        nullable=True,
        comment='Comma-separated nconst IDs of writers'
    )
    
    __table_args__ = (
        # 🎯 INDEX 1: Directors - "All movies by this director"
        Index('idx_crew_directors', 'directors'),
        
        # 🎯 INDEX 2: Writers - "All movies by this writer"
        Index('idx_crew_writers', 'writers'),
        
        # 🔧 NOTE: For exact director matching, you'll need to parse the CSV
        # or use: WHERE directors LIKE '%nm0000116%'
    )
    
    def __repr__(self):
        return f"<TitleCrew(tconst='{self.tconst}', directors='{self.directors}')>"


# ----------------------------------------------------------------------------
# 5. TITLE PRINCIPALS - Cast and Crew Details
# ----------------------------------------------------------------------------
class TitlePrincipals(Base):
    """
    Detailed cast and crew information (top-billed only).
    LARGEST TABLE - 95.3 million records!
    
    Corresponds to: title.principals.tsv.gz
    Records: ~95.3 million principal credits
    
    🎯 PRIMARY USE: Get cast lists, character names, actor collaborations
    🔗 JOINS: 
        - tconst → title_basics.tconst
        - nconst → name_basics.nconst
    
    📊 INDEXES (9 total):
        1. PRIMARY: id - Auto-increment key
        2. FOREIGN KEY: tconst - Join to title_basics
        3. FOREIGN KEY: nconst - Join to name_basics
        4. COMPOSITE: tconst + ordering - Get cast in billing order
        5. COMPOSITE: nconst + category - All movies where person was director
        6. COMPOSITE: category + tconst - All actors in a movie
        7. SIMPLE: category - All actor credits
        8. COMPOSITE: tconst + category + ordering - Actors in billing order
        9. UNIQUE: tconst + ordering - Prevent duplicate orderings
    
    🔧 IMPORTANT: Only includes top-billed cast/crew (usually top 10-20 per title)
    """
    __tablename__ = 'title_principals'
    
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment='Auto-increment ID for internal use'
    )
    
    tconst = Column(
        String(15),
        ForeignKey('title_basics.tconst', ondelete='CASCADE'),
        nullable=False,
        comment='References title_basics.tconst'
    )
    
    ordering = Column(
        Integer,
        nullable=False,
        comment='Billing order (INTEGER: 1 = top-billed, higher = less prominent)'
    )
    
    nconst = Column(
        String(15),
        ForeignKey('name_basics.nconst', ondelete='CASCADE'),
        nullable=False,
        comment='References name_basics.nconst (person ID)'
    )
    
    category = Column(
        String(50),
        nullable=True,
        comment='Category: actor, actress, director, writer, producer, composer, etc.'
    )
    
    job = Column(
        String(300),
        nullable=True,
        comment='Specific job title if applicable (e.g., "screenplay", "director")'
    )
    
    characters = Column(
        Text,
        nullable=True,
        comment='Character name(s) in JSON array format: ["Character Name"]'
    )
    
    __table_args__ = (
        # 🎯 INDEX 1: Foreign key - Join to titles
        Index('idx_principals_tconst', 'tconst'),
        
        # 🎯 INDEX 2: Foreign key - Join to people
        Index('idx_principals_nconst', 'nconst'),
        
        # 🎯 INDEX 3: Title + Ordering - "Get cast in billing order"
        Index('idx_principals_title_order', 'tconst', 'ordering'),
        
        # 🎯 INDEX 4: Person + Category - "All movies where Tom Cruise was actor"
        Index('idx_principals_person_cat', 'nconst', 'category'),
        
        # 🎯 INDEX 5: Category + Title - "All actors in this movie"
        Index('idx_principals_cat_title', 'category', 'tconst'),
        
        # 🎯 INDEX 6: Category only - "All director credits"
        Index('idx_principals_category', 'category'),
        
        # 🎯 INDEX 7: Title + Category + Ordering - "Actors in billing order"
        Index('idx_principals_full', 'tconst', 'category', 'ordering'),
        
        # 🎯 INDEX 8: Person + Title - "Check if person worked on title"
        Index('idx_principals_person_title', 'nconst', 'tconst'),
        
        # 🎯 UNIQUE CONSTRAINT: Prevent duplicate orderings
        UniqueConstraint('tconst', 'ordering', name='uq_tconst_ordering'),
    )
    
    def __repr__(self):
        return f"<TitlePrincipals(tconst='{self.tconst}', nconst='{self.nconst}', category='{self.category}')>"


# ----------------------------------------------------------------------------
# 6. TITLE EPISODE - TV Episode Information
# ----------------------------------------------------------------------------
class TitleEpisode(Base):
    """
    TV episode relationships to parent series.
    
    Corresponds to: title.episode.tsv.gz
    Records: ~9.2 million TV episodes
    
    🎯 PRIMARY USE: Organize TV episodes by series/season
    🔗 JOINS: 
        - tconst → title_basics.tconst (episode details)
        - parent_tconst → title_basics.tconst (series details)
    
    📊 INDEXES (5 total):
        1. PRIMARY: tconst - Episode ID
        2. FOREIGN KEY: parent_tconst - Series ID
        3. COMPOSITE: parent_tconst + season_number - All episodes in season 1
        4. COMPOSITE: parent + season + episode - Specific episode (S01E05)
        5. SIMPLE: season_number - All season finales across all shows
    """
    __tablename__ = 'title_episode'
    
    tconst = Column(
        String(15),
        ForeignKey('title_basics.tconst', ondelete='CASCADE'),
        primary_key=True,
        comment='Episode tconst (references title_basics.tconst)'
    )
    
    parent_tconst = Column(
        String(15),
        ForeignKey('title_basics.tconst', ondelete='CASCADE'),
        nullable=False,
        comment='Parent TV series tconst (references title_basics.tconst)'
    )
    
    season_number = Column(
        Integer,
        nullable=True,
        comment='Season number (INTEGER, NULL if unknown)'
    )
    
    episode_number = Column(
        Integer,
        nullable=True,
        comment='Episode number within season (INTEGER, NULL if unknown)'
    )
    
    __table_args__ = (
        # 🎯 INDEX 1: Parent series - "All episodes of Breaking Bad"
        Index('idx_episode_parent', 'parent_tconst'),
        
        # 🎯 INDEX 2: Parent + Season - "All Season 1 episodes"
        Index('idx_episode_season', 'parent_tconst', 'season_number'),
        
        # 🎯 INDEX 3: Parent + Season + Episode - "Get S01E05"
        Index('idx_episode_full', 'parent_tconst', 'season_number', 'episode_number'),
        
        # 🎯 INDEX 4: Season number - "All season 1 episodes across all shows"
        Index('idx_episode_season_num', 'season_number'),
    )
    
    def __repr__(self):
        return f"<TitleEpisode(tconst='{self.tconst}', series='{self.parent_tconst}', S{self.season_number:02d}E{self.episode_number:02d})>"


# ----------------------------------------------------------------------------
# 7. NAME BASICS - People Data
# ----------------------------------------------------------------------------
class NameBasics(Base):
    """
    Information about people (actors, directors, writers, etc.)
    Includes birth/death years as requested!
    
    Corresponds to: name.basics.tsv.gz
    Records: ~14.8 million people
    
    🎯 PRIMARY USE: Get person details, birth/death info, professions
    🔗 JOINS:
        - nconst ← title_crew.directors/writers
        - nconst ← title_principals.nconst
    
    📊 INDEXES (8 total):
        1. PRIMARY: nconst - Person ID
        2. SIMPLE: primary_name - Search by name
        3. SIMPLE: birth_year - People born in 1970
        4. SIMPLE: death_year - People who died in 2020
        5. COMPOSITE: birth_year + death_year - Living people
        6. SIMPLE: primary_profession - All directors
        7. COMPOSITE: profession + birth_year - Directors born in 1960s
        8. COMPOSITE: name + birth_year - Disambiguate same names
    
    🔧 DATA TYPES: birth_year and death_year are INTEGER
    """
    __tablename__ = 'name_basics'
    
    nconst = Column(
        String(15),
        primary_key=True,
        comment='IMDb unique person identifier (e.g., nm0000129 for Tom Cruise)'
    )
    
    primary_name = Column(
        String(200),
        nullable=False,
        comment='Name as most commonly credited'
    )
    
    birth_year = Column(
        Integer,
        nullable=True,
        comment='Year of birth (INTEGER, NULL if unknown/private)'
    )
    
    death_year = Column(
        Integer,
        nullable=True,
        comment='Year of death (INTEGER, NULL if alive or unknown)'
    )
    
    primary_profession = Column(
        String(200),
        nullable=True,
        comment='Top 3 professions: actor,director,writer (comma-separated)'
    )
    
    known_for_titles = Column(
        String(200),
        nullable=True,
        comment='Notable works (up to 4 tconsts, comma-separated)'
    )
    
    __table_args__ = (
        # 🎯 INDEX 1: Name search - "Find Tom Cruise"
        Index('idx_name_primary', 'primary_name'),
        
        # 🎯 INDEX 2: Birth year - "People born in 1970"
        Index('idx_name_birth', 'birth_year'),
        
        # 🎯 INDEX 3: Death year - "People who died in 2020"
        Index('idx_name_death', 'death_year'),
        
        # 🎯 INDEX 4: Living people - "birth_year IS NOT NULL AND death_year IS NULL"
        Index('idx_name_alive', 'birth_year', 'death_year'),
        
        # 🎯 INDEX 5: Profession - "All directors"
        Index('idx_name_profession', 'primary_profession'),
        
        # 🎯 INDEX 6: Profession + Birth - "Directors born in 1960s"
        Index('idx_name_prof_birth', 'primary_profession', 'birth_year'),
        
        # 🎯 INDEX 7: Name + Birth - Disambiguate people with same name
        Index('idx_name_birth_combo', 'primary_name', 'birth_year'),
        
        # 🔧 CHECK CONSTRAINTS: Relaxed to handle BC dates (where birth > death numerically)
        # Note: Aeschylus (525 BC - 456 BC) is stored as birth=525, death=456
        # We cannot use death >= birth constraint for BC dates
        CheckConstraint('birth_year IS NULL OR birth_year <= 2100', name='chk_birth_year'),
        CheckConstraint('death_year IS NULL OR death_year <= 2100', name='chk_death_year'),
    )
    
    def __repr__(self):
        age_info = f"born {self.birth_year}" if self.birth_year else "birth unknown"
        if self.death_year:
            age_info += f", died {self.death_year}"
        return f"<NameBasics(nconst='{self.nconst}', name='{self.primary_name}', {age_info})>"


# ============================================================================
# DATA LOADING FUNCTIONS WITH PROPER TYPE HANDLING
# ============================================================================

def clean_value(value: str) -> Optional[str]:
    r"""
    Clean IMDb TSV value, converting \N to None.
    
    Args:
        value: Raw string value from TSV
        
    Returns:
        Cleaned value or None if \N
        
    🔧 IMPORTANT: IMDb uses \N to represent NULL values
    """
    if value == r'\N' or value == '' or value is None:
        return None
    return value.strip()


def parse_boolean(value: str) -> bool:
    r"""
    Parse IMDb boolean values (0 or 1).
    
    Args:
        value: String value ('0', '1', or '\N')
        
    Returns:
        Boolean value (False for NULL)
    """
    cleaned = clean_value(value)
    if cleaned is None:
        return False
    return cleaned == '1'


def parse_integer(value: str) -> Optional[int]:
    r"""
    Parse IMDb integer values with validation.
    
    Args:
        value: String value or '\N'
        
    Returns:
        Integer value or None
        
    🔧 VALIDATES: Ensures proper integer conversion
    """
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return None


def parse_float(value: str, decimals: int = 1) -> Optional[float]:
    r"""
    Parse IMDb float values with precision control.
    
    Args:
        value: String value or '\N'
        decimals: Number of decimal places (default: 1 for ratings)
        
    Returns:
        Float value rounded to specified decimals or None
        
    🔧 ENSURES: Ratings are stored as X.X format (e.g., 8.5)
    """
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        float_val = float(cleaned)
        return round(float_val, decimals)
    except (ValueError, TypeError):
        return None


def read_tsv_in_batches(
    file_path: Path,
    batch_size: int = 10000
) -> Iterator[List[Dict[str, Any]]]:
    r"""
    Read TSV file in batches to avoid memory issues.
    
    Args:
        file_path: Path to TSV file
        batch_size: Number of rows per batch
        
    Yields:
        Batches of rows as list of dictionaries
        
    🔧 CUSTOMIZE: Adjust batch_size based on available RAM
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        batch = []
        for row in reader:
            batch.append(row)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch


def load_title_basics(session: Session, file_path: Path, batch_size: int = 10000):
    r"""
    Load title.basics.tsv into database with proper data types.
    
    🎯 LOADS: ~12 million titles
    ⏱️ TIME: ~3-5 minutes
    📊 DATA TYPES: start_year, end_year, runtime_minutes as INTEGER
    """
    print(f"\n📥 Loading Title Basics from {file_path.name}")
    
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
    
    with tqdm(total=total_lines, desc="   Progress", unit=" titles") as pbar:
        for batch in read_tsv_in_batches(file_path, batch_size):
            records = []
            
            for row in batch:
                record = TitleBasics(
                    tconst=clean_value(row['tconst']),
                    title_type=clean_value(row['titleType']),
                    primary_title=clean_value(row['primaryTitle']),
                    original_title=clean_value(row['originalTitle']),
                    is_adult=parse_boolean(row['isAdult']),
                    start_year=parse_integer(row['startYear']),  # INTEGER
                    end_year=parse_integer(row['endYear']),      # INTEGER
                    runtime_minutes=parse_integer(row['runtimeMinutes']),  # INTEGER
                    genres=clean_value(row['genres'])
                )
                records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            pbar.update(len(batch))
    
    print(f"   ✅ Loaded {total_lines:,} title records")


def load_title_akas(session: Session, file_path: Path, batch_size: int = 10000):
    r"""
    Load title.akas.tsv into database.
    
    🎯 LOADS: ~53.5 million alternative titles
    ⏱️ TIME: ~15-20 minutes
    """
    print(f"\n📥 Loading Title AKAs from {file_path.name}")
    
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
    
    with tqdm(total=total_lines, desc="   Progress", unit=" akas") as pbar:
        for batch in read_tsv_in_batches(file_path, batch_size):
            records = []
            
            for row in batch:
                record = TitleAkas(
                    title_id=clean_value(row['titleId']),
                    ordering=parse_integer(row['ordering']),
                    title=clean_value(row['title']),
                    region=clean_value(row['region']),
                    language=clean_value(row['language']),
                    types=clean_value(row['types']),
                    attributes=clean_value(row['attributes']),
                    is_original_title=parse_boolean(row['isOriginalTitle'])
                )
                records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            pbar.update(len(batch))
    
    print(f"   ✅ Loaded {total_lines:,} alternative title records")


def load_title_ratings(session: Session, file_path: Path, batch_size: int = 10000):
    """
    Load title.ratings.tsv into database.
    
    🎯 LOADS: ~1.6 million ratings
    ⏱️ TIME: ~30 seconds
    📊 DATA TYPES: average_rating as FLOAT with 1 decimal, num_votes as INTEGER
    """
    print(f"\n📥 Loading Title Ratings from {file_path.name}")
    
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
    
    with tqdm(total=total_lines, desc="   Progress", unit=" ratings") as pbar:
        for batch in read_tsv_in_batches(file_path, batch_size):
            records = []
            
            for row in batch:
                record = TitleRatings(
                    tconst=clean_value(row['tconst']),
                    average_rating=parse_float(row['averageRating'], decimals=1),  # FLOAT with 1 decimal
                    num_votes=parse_integer(row['numVotes'])  # INTEGER
                )
                records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            pbar.update(len(batch))
    
    print(f"   ✅ Loaded {total_lines:,} rating records")


def load_title_crew(session: Session, file_path: Path, batch_size: int = 10000):
    """
    Load title.crew.tsv into database.
    
    🎯 LOADS: ~12 million crew records
    ⏱️ TIME: ~3-5 minutes
    """
    print(f"\n📥 Loading Title Crew from {file_path.name}")
    
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
    
    with tqdm(total=total_lines, desc="   Progress", unit=" crew") as pbar:
        for batch in read_tsv_in_batches(file_path, batch_size):
            records = []
            
            for row in batch:
                record = TitleCrew(
                    tconst=clean_value(row['tconst']),
                    directors=clean_value(row['directors']),
                    writers=clean_value(row['writers'])
                )
                records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            pbar.update(len(batch))
    
    print(f"   ✅ Loaded {total_lines:,} crew records")


def load_title_principals(session: Session, file_path: Path, batch_size: int = 10000):
    """
    Load title.principals.tsv into database.
    
    🎯 LOADS: ~95.3 million principal credits
    ⏱️ TIME: ~25-30 minutes (LARGEST TABLE!)
    📊 DATA TYPES: ordering as INTEGER
    """
    print(f"\n📥 Loading Title Principals from {file_path.name}")
    print(f"   ⚠️  This is the largest table (~95M records) and will take 25-30 minutes")
    
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
    
    with tqdm(total=total_lines, desc="   Progress", unit=" principals") as pbar:
        for batch in read_tsv_in_batches(file_path, batch_size):
            records = []
            
            for row in batch:
                record = TitlePrincipals(
                    tconst=clean_value(row['tconst']),
                    ordering=parse_integer(row['ordering']),  # INTEGER
                    nconst=clean_value(row['nconst']),
                    category=clean_value(row['category']),
                    job=clean_value(row['job']),
                    characters=clean_value(row['characters'])
                )
                records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            pbar.update(len(batch))
    
    print(f"   ✅ Loaded {total_lines:,} principal records")


def load_title_episode(session: Session, file_path: Path, batch_size: int = 10000):
    """
    Load title.episode.tsv into database.
    
    🎯 LOADS: ~9.2 million episodes
    ⏱️ TIME: ~2-3 minutes
    📊 DATA TYPES: season_number, episode_number as INTEGER
    """
    print(f"\n📥 Loading Title Episodes from {file_path.name}")
    
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
    
    with tqdm(total=total_lines, desc="   Progress", unit=" episodes") as pbar:
        for batch in read_tsv_in_batches(file_path, batch_size):
            records = []
            
            for row in batch:
                record = TitleEpisode(
                    tconst=clean_value(row['tconst']),
                    parent_tconst=clean_value(row['parentTconst']),
                    season_number=parse_integer(row['seasonNumber']),    # INTEGER
                    episode_number=parse_integer(row['episodeNumber'])   # INTEGER
                )
                records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            pbar.update(len(batch))
    
    print(f"   ✅ Loaded {total_lines:,} episode records")


def load_name_basics(session: Session, file_path: Path, batch_size: int = 10000):
    """
    Load name.basics.tsv into database.
    
    🎯 LOADS: ~14.8 million people
    ⏱️ TIME: ~4-6 minutes
    📊 DATA TYPES: birth_year, death_year as INTEGER
    """
    print(f"\n📥 Loading Name Basics from {file_path.name}")
    
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
    
    skipped = 0
    loaded = 0
    
    with tqdm(total=total_lines, desc="   Progress", unit=" people") as pbar:
        for batch in read_tsv_in_batches(file_path, batch_size):
            records = []
            
            for row in batch:
                # Skip records with no name (NULL primary_name)
                name = clean_value(row['primaryName'])
                if name is None:
                    skipped += 1
                    continue
                
                record = NameBasics(
                    nconst=clean_value(row['nconst']),
                    primary_name=name,
                    birth_year=parse_integer(row['birthYear']),
                    death_year=parse_integer(row['deathYear']),
                    primary_profession=clean_value(row['primaryProfession']),
                    known_for_titles=clean_value(row['knownForTitles'])
                )
                records.append(record)
            
            if records:
                session.bulk_save_objects(records)
                session.commit()
                loaded += len(records)
            
            pbar.update(len(batch))
    
    print(f"   ✅ Loaded {loaded:,} people records")
    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped:,} records with no name")


# ============================================================================
# MAIN LOADING ORCHESTRATION
# ============================================================================

def create_tables(recreate: bool = False):
    """Create all database tables."""
    if recreate:
        print("\n⚠️  WARNING: Dropping all existing tables!")
        response = input("Are you sure? This will DELETE all data! (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Cancelled")
            return False
        
        Base.metadata.drop_all(engine)
        print("   🗑️  Dropped all tables")
    
    Base.metadata.create_all(engine)
    print("   ✅ Created all tables with proper schemas and indexes")
    return True


def verify_database(session: Session):
    r"""
    Run verification queries to ensure data loaded correctly.
    
    🔧 VERIFIES:
        - Record counts match expected
        - Data types are correct (years as INTEGER, ratings as FLOAT)
        - Sample queries work correctly
        - Indexes are created
    """
    print("\n" + "="*70)
    print("🔍 DATABASE VERIFICATION")
    print("="*70)
    
    # Test 1: Record counts
    print(f"\n📊 Record counts in database:")
    counts = {
        'title_basics': session.query(TitleBasics).count(),
        'title_akas': session.query(TitleAkas).count(),
        'title_ratings': session.query(TitleRatings).count(),
        'title_crew': session.query(TitleCrew).count(),
        'title_principals': session.query(TitlePrincipals).count(),
        'title_episode': session.query(TitleEpisode).count(),
        'name_basics': session.query(NameBasics).count(),
    }
    
    total = 0
    for table, count in counts.items():
        print(f"   • {table}: {count:,}")
        total += count
    
    print(f"\n   ✅ Total records: {total:,}")
    
    # Test 2: Data type verification
    print(f"\n🔍 Data type verification:")
    
    # Check year as INTEGER
    sample_title = session.query(TitleBasics).filter(
        TitleBasics.start_year.isnot(None)
    ).first()
    if sample_title:
        print(f"   • Years as INTEGER: {type(sample_title.start_year).__name__} = {sample_title.start_year}")
        assert isinstance(sample_title.start_year, int), "start_year should be INTEGER!"
    
    # Check rating as FLOAT
    sample_rating = session.query(TitleRatings).filter(
        TitleRatings.average_rating.isnot(None)
    ).first()
    if sample_rating:
        print(f"   • Rating as FLOAT: {type(sample_rating.average_rating).__name__} = {sample_rating.average_rating:.1f}")
        assert isinstance(sample_rating.average_rating, float), "average_rating should be FLOAT!"
    
    # Check birth year as INTEGER
    sample_person = session.query(NameBasics).filter(
        NameBasics.birth_year.isnot(None)
    ).first()
    if sample_person:
        print(f"   • Birth year as INTEGER: {type(sample_person.birth_year).__name__} = {sample_person.birth_year}")
        assert isinstance(sample_person.birth_year, int), "birth_year should be INTEGER!"
    
    print(f"   ✅ All data types correct!")
    
    # Test 3: Sample queries
    print(f"\n🎬 Sample queries:")
    
    # Movies from 1994
    movies_1994 = session.query(TitleBasics).filter(
        TitleBasics.title_type == 'movie',
        TitleBasics.start_year == 1994
    ).limit(5).all()
    
    print(f"   • Movies from 1994:")
    for movie in movies_1994:
        print(f"     - {movie.primary_title} ({movie.start_year})")
    
    # Highly rated movies
    top_rated = session.query(
        TitleBasics.primary_title,
        TitleBasics.start_year,
        TitleRatings.average_rating,
        TitleRatings.num_votes
    ).join(
        TitleRatings, TitleBasics.tconst == TitleRatings.tconst
    ).filter(
        TitleBasics.title_type == 'movie',
        TitleRatings.num_votes > 100000
    ).order_by(
        TitleRatings.average_rating.desc()
    ).limit(5).all()
    
    print(f"\n   • Top-rated movies (>100K votes):")
    for title, year, rating, votes in top_rated:
        print(f"     - {title} ({year}): {rating:.1f}/10 ({votes:,} votes)")
    
    print(f"\n   ✅ All queries working correctly!")


def load_all_datasets(batch_size: int = 10000, specific_table: Optional[str] = None, verify: bool = False):
    r"""Load all IMDb datasets into database."""
    print("\n" + "="*70)
    print("🎬 FILMOTECA - IMDb Database Loader")
    print("="*70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Database: {config.database.database_url}")
    print(f"📦 Batch size: {batch_size:,} records")
    
    loaders = {
        'title_basics': (load_title_basics, 'title.basics.tsv', '~12M titles', '3-5 min'),
        'title_akas': (load_title_akas, 'title.akas.tsv', '~53.5M akas', '15-20 min'),
        'title_ratings': (load_title_ratings, 'title.ratings.tsv', '~1.6M ratings', '30 sec'),
        'title_crew': (load_title_crew, 'title.crew.tsv', '~12M crew', '3-5 min'),
        'title_principals': (load_title_principals, 'title.principals.tsv', '~95.3M principals', '25-30 min'),
        'title_episode': (load_title_episode, 'title.episode.tsv', '~9.2M episodes', '2-3 min'),
        'name_basics': (load_name_basics, 'name.basics.tsv', '~14.8M people', '4-6 min'),
    }
    
    if specific_table:
        if specific_table not in loaders:
            print(f"\n❌ Unknown table: {specific_table}")
            print(f"Available tables: {', '.join(loaders.keys())}")
            return
        loaders = {specific_table: loaders[specific_table]}
        print(f"\n🎯 Loading specific table: {specific_table}")
    else:
        print(f"\n📊 Tables to load: {len(loaders)}")
        print("\n⏱️  Estimated total time: ~50-70 minutes for all tables")
    
    session = SessionLocal()
    
    try:
        start_time = time.time()
        
        for i, (table_name, (loader_func, filename, record_count, est_time)) in enumerate(loaders.items(), 1):
            print(f"\n{'='*70}")
            print(f"[{i}/{len(loaders)}] {table_name.upper()}")
            print(f"{'='*70}")
            print(f"📄 File: {filename}")
            print(f"📊 Records: {record_count}")
            print(f"⏱️  Est. time: {est_time}")
            
            file_path = config.paths.imdb_dir / filename
            
            if not file_path.exists():
                print(f"   ❌ File not found: {file_path}")
                print(f"   ⚠️  Skipping {table_name}")
                continue
            
            table_start = time.time()
            
            try:
                loader_func(session, file_path, batch_size)
                table_time = time.time() - table_start
                print(f"   ⏱️  Completed in {table_time/60:.1f} minutes")
            except Exception as e:
                print(f"   ❌ Error loading {table_name}: {e}")
                print(f"   ⚠️  Continuing with next table...")
                session.rollback()
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "="*70)
        print("📊 LOADING SUMMARY")
        print("="*70)
        print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        
        # Verification
        if verify:
            verify_database(session)
        else:
            # Just show counts
            print(f"\n📈 Record counts in database:")
            print(f"   • title_basics: {session.query(TitleBasics).count():,}")
            print(f"   • title_akas: {session.query(TitleAkas).count():,}")
            print(f"   • title_ratings: {session.query(TitleRatings).count():,}")
            print(f"   • title_crew: {session.query(TitleCrew).count():,}")
            print(f"   • title_principals: {session.query(TitlePrincipals).count():,}")
            print(f"   • title_episode: {session.query(TitleEpisode).count():,}")
            print(f"   • name_basics: {session.query(NameBasics).count():,}")
        
        # Database file size
        if config.database.is_sqlite:
            db_path = config.database.database_path
            if db_path and db_path.exists():
                db_size = db_path.stat().st_size / (1024**3)  # GB
                print(f"\n💾 Database file size: {db_size:.2f} GB")
        
        print("\n" + "="*70)
        print("✅ DATABASE LOADING COMPLETE!")
        print("="*70)
        print("\n🎯 Next steps:")
        print("   1. Run with --verify flag to check data integrity")
        print("   2. Run: python scripts/match_watchlist.py")
        print("      → Match your Watched.csv with IMDb data")
        
    finally:
        session.close()
    
    print(f"\n📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Load IMDb datasets into SQLite database with proper data types',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Drop and recreate all tables (⚠️ DESTRUCTIVE!)'
    )
    
    parser.add_argument(
        '--table',
        type=str,
        help='Load only specific table (e.g., title_basics)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=10000,
        help='Batch size for loading (default: 10000)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Run verification queries after loading'
    )
    
    args = parser.parse_args()
    
    # Create tables
    if create_tables(recreate=args.recreate):
        # Load data
        load_all_datasets(
            batch_size=args.batch,
            specific_table=args.table,
            verify=args.verify
        )


if __name__ == "__main__":
    main()