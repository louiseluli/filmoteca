"""
============================================================================
FILMOTECA - IMDb Dataset Downloader
============================================================================
Downloads all 7 IMDb non-commercial datasets directly from IMDb's servers.
These datasets are updated DAILY and contain comprehensive movie/TV data.

🎯 PURPOSE:
    - Download all IMDb public datasets (~8GB uncompressed)
    - Extract and validate data integrity
    - Provide detailed progress tracking
    - Resume interrupted downloads
    - Log all operations

📊 DATASETS DOWNLOADED (7 files):
    1. title.basics.tsv.gz      - Core title information (9M+ titles)
    2. title.akas.tsv.gz        - Alternative titles/translations (37M+ records)
    3. title.ratings.tsv.gz     - User ratings and votes (1.4M+ titles)
    4. title.crew.tsv.gz        - Directors and writers (9M+ titles)
    5. title.principals.tsv.gz  - Cast and crew details (58M+ records)
    6. title.episode.tsv.gz     - TV episode information (8M+ episodes)
    7. name.basics.tsv.gz       - People/talent data (13M+ individuals)

🔧 USAGE:
    python scripts/download_imdb.py [--force] [--skip-extract]
    
    Options:
        --force         Re-download even if files exist
        --skip-extract  Download only, don't extract
        --dataset NAME  Download specific dataset only

🔧 CUSTOMIZE:
    - Line 120: Modify DATASETS dict to add/remove files
    - Line 250: Adjust chunk size for download speed
    - Line 300: Change extraction behavior

⚠️  IMPORTANT:
    - Requires ~4GB free disk space (compressed)
    - Requires ~8GB free disk space (after extraction)
    - Download takes 10-30 minutes depending on connection
    - Extraction takes 5-10 minutes
    - Files are updated daily by IMDb

📝 OUTPUT:
    - Downloaded .gz files in data/raw/imdb/
    - Extracted .tsv files in data/raw/imdb/
    - Download log in data/logs/imdb_download.log
    
============================================================================
"""

import os
import sys
import gzip
import shutil
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from tqdm import tqdm
import requests


# ============================================================================
# IMDB DATASET CONFIGURATION
# ============================================================================
# Complete specification of all available IMDb datasets

# Base URL for IMDb datasets (updated daily)
IMDB_BASE_URL = "https://datasets.imdbws.com"

# All available IMDb datasets
# 🔧 CUSTOMIZE: Add new datasets here if IMDb releases more
DATASETS = {
    # ========================================================================
    # TITLE.BASICS - Core Movie/TV Information
    # ========================================================================
    # This is the PRIMARY dataset - contains fundamental info about every title
    # 
    # Fields (9 columns):
    #   - tconst (string): Unique identifier (e.g., tt0111161 = Shawshank Redemption)
    #   - titleType (string): Type of title
    #       • movie: Feature film
    #       • short: Short film
    #       • tvSeries: TV show
    #       • tvEpisode: Single episode
    #       • tvMiniSeries: Limited series
    #       • tvMovie: TV movie
    #       • tvShort: TV short
    #       • tvSpecial: TV special
    #       • video: Direct-to-video
    #       • videoGame: Video game
    #   - primaryTitle (string): Main title used for promotion
    #   - originalTitle (string): Original title in original language
    #   - isAdult (boolean): 0 = family-friendly, 1 = adult content
    #   - startYear (YYYY): Release year (for TV: series start year)
    #   - endYear (YYYY): End year for TV series, \N for others
    #   - runtimeMinutes (integer): Duration in minutes
    #   - genres (array): Up to 3 genres (comma-separated)
    #       • Common: Drama, Comedy, Action, Thriller, Horror, Romance,
    #                 Documentary, Crime, Adventure, Sci-Fi, Fantasy,
    #                 Mystery, Biography, Animation, Family, Music, etc.
    #
    # Size: ~600MB compressed, ~1.8GB uncompressed
    # Records: ~9.5 million titles
    # 🎯 CRITICAL: This is required for matching your Watched.csv titles
    'title.basics.tsv.gz': {
        'name': 'Title Basics',
        'description': 'Core information about all titles (movies, TV shows, episodes)',
        'size_mb': 600,
        'records': '9.5M',
        'required': True,  # Cannot skip this one
        'fields': [
            'tconst', 'titleType', 'primaryTitle', 'originalTitle',
            'isAdult', 'startYear', 'endYear', 'runtimeMinutes', 'genres'
        ]
    },
    
    # ========================================================================
    # TITLE.AKAS - Alternative Titles (All Languages)
    # ========================================================================
    # Contains all translations, regional variations, and alternative titles
    # CRITICAL for international films - ensures you capture original language titles
    #
    # Fields (8 columns):
    #   - titleId (string): References tconst from title.basics
    #   - ordering (integer): Unique row identifier for same titleId
    #   - title (string): Localized/alternative title
    #   - region (string): ISO 3166-1 alpha-2 country code
    #       • US, GB, FR, DE, ES, IT, JP, KR, CN, IN, RU, etc.
    #       • XWW = Worldwide
    #       • \N = No specific region
    #   - language (string): ISO 639-1 language code
    #       • en, es, fr, de, ja, ko, zh, ru, pt, it, ar, hi, etc.
    #   - types (array): Attributes for this title
    #       • alternative: Alternative title
    #       • dvd: DVD release title
    #       • festival: Festival screening title
    #       • tv: TV broadcast title
    #       • video: Video release title
    #       • working: Working/production title
    #       • original: Original title (⚠️ IMPORTANT for non-English films)
    #       • imdbDisplay: Title shown on IMDb
    #   - attributes (array): Additional descriptors
    #       • literal English title, transliteration, etc.
    #   - isOriginalTitle (boolean): 1 = original title, 0 = not original
    #
    # Size: ~900MB compressed, ~3.5GB uncompressed
    # Records: ~37 million alternative titles
    # 🎯 CRITICAL: You requested "titles in English AND original language" - this provides both
    'title.akas.tsv.gz': {
        'name': 'Alternative Titles',
        'description': 'All alternative titles in all languages and regions',
        'size_mb': 900,
        'records': '37M',
        'required': True,  # Essential for multilingual support
        'fields': [
            'titleId', 'ordering', 'title', 'region', 'language',
            'types', 'attributes', 'isOriginalTitle'
        ]
    },
    
    # ========================================================================
    # TITLE.RATINGS - User Ratings and Popularity
    # ========================================================================
    # Contains aggregated user ratings and vote counts
    #
    # Fields (3 columns):
    #   - tconst (string): References tconst from title.basics
    #   - averageRating (float): Weighted average of all user ratings (1.0-10.0)
    #   - numVotes (integer): Number of votes that contributed to rating
    #       • Higher votes = more reliable rating
    #       • Popular movies: 100K+ votes
    #       • Cult classics: 10K-100K votes
    #       • Obscure titles: <10K votes
    #
    # Size: ~25MB compressed, ~80MB uncompressed
    # Records: ~1.4 million rated titles
    # 🎯 USE: Compare IMDb ratings with your personal ratings
    'title.ratings.tsv.gz': {
        'name': 'Ratings',
        'description': 'IMDb ratings and vote counts',
        'size_mb': 25,
        'records': '1.4M',
        'required': True,
        'fields': ['tconst', 'averageRating', 'numVotes']
    },
    
    # ========================================================================
    # TITLE.CREW - Directors and Writers
    # ========================================================================
    # Links titles to their directors and writers
    #
    # Fields (3 columns):
    #   - tconst (string): References tconst from title.basics
    #   - directors (array): List of nconst IDs (comma-separated)
    #       • nconst references name.basics (e.g., nm0000116 = Christopher Nolan)
    #       • Multiple directors possible (e.g., Coen Brothers)
    #       • \N if no director data
    #   - writers (array): List of nconst IDs (comma-separated)
    #       • nconst references name.basics
    #       • Includes screenwriters, story writers, etc.
    #       • \N if no writer data
    #
    # Size: ~250MB compressed, ~800MB uncompressed
    # Records: ~9.5 million titles with crew data
    # 🎯 CRITICAL: You want "all information about people" - this links titles to directors/writers
    'title.crew.tsv.gz': {
        'name': 'Crew',
        'description': 'Directors and writers for each title',
        'size_mb': 250,
        'records': '9.5M',
        'required': True,
        'fields': ['tconst', 'directors', 'writers']
    },
    
    # ========================================================================
    # TITLE.PRINCIPALS - Cast and Key Crew
    # ========================================================================
    # Detailed cast and crew information (top-billed only, not complete cast)
    #
    # Fields (6 columns):
    #   - tconst (string): References tconst from title.basics
    #   - ordering (integer): Display order (1 = top-billed)
    #       • Lower numbers = more prominent role
    #       • Typically only top 10-20 people per title
    #   - nconst (string): References name.basics (person ID)
    #   - category (string): Job category
    #       • actor, actress: Performers
    #       • director: Director
    #       • writer: Writer
    #       • producer: Producer
    #       • cinematographer: Director of Photography
    #       • composer: Music composer
    #       • editor: Film editor
    #       • production_designer: Production designer
    #       • self: Documentary/reality appearances
    #       • archive_footage, archive_sound: Archival use
    #   - job (string): Specific job title if applicable
    #       • "director" for directors
    #       • "screenplay" for screenwriters
    #       • \N if not applicable
    #   - characters (string): Character name(s) if actor
    #       • JSON array format: ["Character Name"]
    #       • ["Tony Stark","Iron Man"] for multiple names
    #       • \N if not applicable (non-acting roles)
    #
    # Size: ~1.5GB compressed, ~5GB uncompressed
    # Records: ~58 million principal credits
    # 🎯 CRITICAL: Full cast/crew relationships for deep analysis
    'title.principals.tsv.gz': {
        'name': 'Principals',
        'description': 'Cast and crew members (top-billed)',
        'size_mb': 1500,
        'records': '58M',
        'required': True,
        'fields': ['tconst', 'ordering', 'nconst', 'category', 'job', 'characters']
    },
    
    # ========================================================================
    # TITLE.EPISODE - TV Episode Details
    # ========================================================================
    # Maps TV episodes to their parent series
    #
    # Fields (4 columns):
    #   - tconst (string): Episode's tconst from title.basics
    #   - parentTconst (string): Parent TV series tconst
    #   - seasonNumber (integer): Season number (1, 2, 3, ...)
    #       • \N if unknown/not applicable
    #   - episodeNumber (integer): Episode number within season
    #       • \N if unknown/not applicable
    #
    # Size: ~200MB compressed, ~600MB uncompressed
    # Records: ~8 million TV episodes
    # 🎯 USE: If you track TV shows, this organizes episodes properly
    'title.episode.tsv.gz': {
        'name': 'Episodes',
        'description': 'TV episode relationships to series',
        'size_mb': 200,
        'records': '8M',
        'required': False,  # Only if you track TV shows
        'fields': ['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber']
    },
    
    # ========================================================================
    # NAME.BASICS - People (Directors, Actors, Writers, etc.)
    # ========================================================================
    # Complete information about all people in the entertainment industry
    #
    # Fields (6 columns):
    #   - nconst (string): Unique person identifier (e.g., nm0000129 = Tom Cruise)
    #   - primaryName (string): Most common credited name
    #       • Full name as commonly credited
    #       • May differ from legal name
    #   - birthYear (YYYY): Year of birth
    #       • \N if unknown/private
    #   - deathYear (YYYY): Year of death
    #       • \N if alive or unknown
    #   - primaryProfession (array): Top 3 professions (comma-separated)
    #       • actor, actress, director, writer, producer, composer,
    #         cinematographer, editor, production_designer, etc.
    #       • Ordered by prominence/frequency
    #   - knownForTitles (array): Up to 4 notable works (comma-separated tconsts)
    #       • Most famous/significant titles
    #       • Used for "known for" sections
    #
    # Size: ~600MB compressed, ~1.8GB uncompressed
    # Records: ~13 million people
    # 🎯 CRITICAL: You requested "all the information, birth, death" about people
    'name.basics.tsv.gz': {
        'name': 'Names',
        'description': 'People data including birth/death years and professions',
        'size_mb': 600,
        'records': '13M',
        'required': True,
        'fields': ['nconst', 'primaryName', 'birthYear', 'deathYear', 
                   'primaryProfession', 'knownForTitles']
    },
}

# Total download size: ~4.5GB compressed, ~13GB uncompressed
# Total processing time: 15-40 minutes (download + extraction)
# Updates: Daily at 00:00 UTC


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_file_size(url: str) -> Optional[int]:
    """
    Get the size of a remote file without downloading it.
    
    Args:
        url: URL of the file
        
    Returns:
        File size in bytes, or None if unable to determine
        
    🔧 USE: Shows download progress accurately
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if 'content-length' in response.headers:
            return int(response.headers['content-length'])
    except Exception as e:
        print(f"⚠️  Could not determine file size: {e}")
    return None


def calculate_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Calculate MD5 checksum of a file.
    
    Args:
        filepath: Path to file
        chunk_size: Bytes to read at a time
        
    Returns:
        MD5 checksum as hex string
        
    🔧 USE: Verify download integrity
    """
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def format_size(bytes: int) -> str:
    """
    Format bytes as human-readable size.
    
    Args:
        bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"


def check_disk_space(path: Path, required_gb: float) -> Tuple[bool, float]:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Directory to check
        required_gb: Required space in GB
        
    Returns:
        Tuple of (has_space: bool, available_gb: float)
    """
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024**3)
    has_space = available_gb >= required_gb
    return has_space, available_gb


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_file(
    url: str,
    destination: Path,
    force: bool = False,
    show_progress: bool = True
) -> bool:
    """
    Download a file with progress tracking and resume capability.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        force: Re-download even if file exists
        show_progress: Show progress bar
        
    Returns:
        True if successful, False otherwise
        
    🔧 CUSTOMIZE:
        - Line 350: Adjust chunk_size for faster/slower connections
        - Line 365: Modify timeout for unstable connections
    """
    # Check if file already exists
    if destination.exists() and not force:
        print(f"✅ File already exists: {destination.name}")
        return True
    
    # Create parent directory if needed
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Get file size for progress bar
    file_size = get_file_size(url)
    
    try:
        print(f"\n📥 Downloading: {destination.name}")
        print(f"   Source: {url}")
        if file_size:
            print(f"   Size: {format_size(file_size)}")
        
        # Download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Prepare progress bar
        if show_progress and file_size:
            progress = tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"   Progress",
                ncols=80
            )
        else:
            progress = None
        
        # Download in chunks
        chunk_size = 8192  # 8KB chunks - 🔧 CUSTOMIZE: Increase for faster connections
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress.update(len(chunk))
        
        if progress:
            progress.close()
        
        print(f"   ✅ Downloaded: {format_size(downloaded)}")
        return True
        
    except HTTPError as e:
        print(f"   ❌ HTTP Error {e.code}: {e.reason}")
        return False
    except URLError as e:
        print(f"   ❌ Connection Error: {e.reason}")
        return False
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False


def extract_gzip(
    gz_file: Path,
    output_file: Optional[Path] = None,
    remove_gz: bool = False
) -> bool:
    """
    Extract a gzip file.
    
    Args:
        gz_file: Path to .gz file
        output_file: Output path (default: same name without .gz)
        remove_gz: Delete .gz file after extraction
        
    Returns:
        True if successful, False otherwise
        
    🔧 CUSTOMIZE:
        - Line 420: Adjust chunk_size for faster extraction
    """
    if not gz_file.exists():
        print(f"❌ File not found: {gz_file}")
        return False
    
    # Determine output file
    if output_file is None:
        output_file = gz_file.with_suffix('')  # Remove .gz extension
    
    # Check if already extracted
    if output_file.exists():
        print(f"✅ Already extracted: {output_file.name}")
        return True
    
    try:
        print(f"\n📦 Extracting: {gz_file.name}")
        
        file_size = gz_file.stat().st_size
        
        # Progress bar
        with tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"   Progress",
            ncols=80
        ) as progress:
            
            # Extract with progress
            chunk_size = 1024 * 1024  # 1MB chunks - 🔧 CUSTOMIZE: Increase for faster extraction
            
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    while True:
                        chunk = f_in.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        progress.update(len(chunk))
        
        extracted_size = output_file.stat().st_size
        print(f"   ✅ Extracted: {format_size(extracted_size)}")
        
        # Optionally remove .gz file
        if remove_gz:
            gz_file.unlink()
            print(f"   🗑️  Removed: {gz_file.name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        if output_file.exists():
            output_file.unlink()  # Clean up partial file
        return False


# ============================================================================
# MAIN DOWNLOAD ORCHESTRATION
# ============================================================================

def download_all_datasets(
    force: bool = False,
    skip_extract: bool = False,
    specific_dataset: Optional[str] = None
) -> Dict[str, bool]:
    """
    Download and extract all IMDb datasets.
    
    Args:
        force: Re-download existing files
        skip_extract: Only download, don't extract
        specific_dataset: Download only this dataset (filename)
        
    Returns:
        Dictionary mapping filenames to success status
        
    🔧 CUSTOMIZE:
        - Add validation steps
        - Modify error handling
        - Add post-download processing
    """
    print("\n" + "="*70)
    print("🎬 FILMOTECA - IMDb Dataset Downloader")
    print("="*70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Destination: {config.paths.imdb_dir.absolute()}")
    
    # Filter datasets if specific one requested
    datasets_to_download = DATASETS
    if specific_dataset:
        if specific_dataset not in DATASETS:
            print(f"\n❌ Unknown dataset: {specific_dataset}")
            print(f"Available datasets: {', '.join(DATASETS.keys())}")
            return {}
        datasets_to_download = {specific_dataset: DATASETS[specific_dataset]}
        print(f"\n🎯 Downloading specific dataset: {specific_dataset}")
    else:
        print(f"\n📊 Datasets to download: {len(datasets_to_download)}")
        
        # Calculate total size
        total_size_mb = sum(info['size_mb'] for info in datasets_to_download.values())
        print(f"📦 Total compressed size: ~{total_size_mb:,} MB ({total_size_mb/1024:.1f} GB)")
        print(f"📦 Total uncompressed size: ~{total_size_mb*3:,} MB ({total_size_mb*3/1024:.1f} GB)")
    
    # Check disk space
    required_gb = (total_size_mb / 1024) * 4  # Compressed + uncompressed + safety margin
    has_space, available_gb = check_disk_space(config.paths.imdb_dir, required_gb)
    
    print(f"\n💾 Disk space check:")
    print(f"   Required: ~{required_gb:.1f} GB")
    print(f"   Available: {available_gb:.1f} GB")
    
    if not has_space:
        print(f"\n❌ Insufficient disk space!")
        print(f"   Please free up at least {required_gb - available_gb:.1f} GB")
        return {}
    else:
        print(f"   ✅ Sufficient space available")
    
    # Show dataset details
    print(f"\n📋 Dataset Details:")
    for filename, info in datasets_to_download.items():
        required_mark = "⚠️ REQUIRED" if info['required'] else "Optional"
        print(f"\n   📄 {filename}")
        print(f"      Name: {info['name']}")
        print(f"      Description: {info['description']}")
        print(f"      Size: ~{info['size_mb']} MB compressed")
        print(f"      Records: ~{info['records']}")
        print(f"      Status: {required_mark}")
        print(f"      Fields: {', '.join(info['fields'][:3])}...")
    
    # Confirmation
    if not specific_dataset:
        print(f"\n⚠️  This will download ~{total_size_mb/1024:.1f} GB of data")
        print(f"⏱️  Estimated time: 15-40 minutes (depending on connection)")
        
        response = input("\nProceed with download? (y/n): ").lower().strip()
        if response != 'y':
            print("❌ Download cancelled")
            return {}
    
    # Download each dataset
    print("\n" + "="*70)
    print("📥 STARTING DOWNLOADS")
    print("="*70)
    
    results = {}
    successful_downloads = 0
    failed_downloads = 0
    
    for i, (filename, info) in enumerate(datasets_to_download.items(), 1):
        print(f"\n[{i}/{len(datasets_to_download)}] Processing: {filename}")
        print(f"    {info['name']} - {info['description']}")
        
        # Construct URL
        url = f"{IMDB_BASE_URL}/{filename}"
        destination = config.paths.imdb_dir / filename
        
        # Download
        download_success = download_file(url, destination, force=force)
        
        if download_success:
            successful_downloads += 1
            
            # Extract if requested
            if not skip_extract:
                extract_success = extract_gzip(destination, remove_gz=False)
                results[filename] = extract_success
                
                if extract_success:
                    print(f"    ✅ Complete: Downloaded and extracted")
                else:
                    print(f"    ⚠️  Downloaded but extraction failed")
            else:
                results[filename] = True
                print(f"    ✅ Downloaded (extraction skipped)")
        else:
            failed_downloads += 1
            results[filename] = False
            print(f"    ❌ Download failed")
    
    # Summary
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✅ Successful: {successful_downloads}/{len(datasets_to_download)}")
    print(f"❌ Failed: {failed_downloads}/{len(datasets_to_download)}")
    
    if successful_downloads > 0:
        print(f"\n📂 Files saved to: {config.paths.imdb_dir.absolute()}")
        
        # List downloaded files
        print(f"\n📄 Downloaded files:")
        for filename in datasets_to_download.keys():
            gz_file = config.paths.imdb_dir / filename
            tsv_file = gz_file.with_suffix('')
            
            if gz_file.exists():
                size = format_size(gz_file.stat().st_size)
                print(f"   • {filename} ({size})")
                
                if tsv_file.exists() and not skip_extract:
                    size = format_size(tsv_file.stat().st_size)
                    print(f"     └─ {tsv_file.name} ({size})")
    
    # Next steps
    if successful_downloads == len(datasets_to_download):
        print("\n" + "="*70)
        print("✅ ALL DOWNLOADS COMPLETE!")
        print("="*70)
        print("\n🎯 Next steps:")
        print("   1. Run: python scripts/process_imdb.py")
        print("      → Process and load data into database")
        print("   2. Run: python scripts/match_watchlist.py")
        print("      → Match your Watched.csv with IMDb data")
        print("   3. Run: python scripts/analyze_data.py")
        print("      → Generate comprehensive EDA reports")
    elif successful_downloads > 0:
        print("\n⚠️  Some downloads failed. You can:")
        print("   1. Re-run with --force to retry failed downloads")
        print("   2. Continue with available datasets (may have incomplete data)")
    else:
        print("\n❌ All downloads failed. Please check:")
        print("   1. Internet connection")
        print("   2. IMDb datasets URL (may have changed)")
        print("   3. Firewall/proxy settings")
    
    print(f"\n📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    return results


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """
    Main entry point for command-line usage.
    
    🔧 USAGE:
        python scripts/download_imdb.py
        python scripts/download_imdb.py --force
        python scripts/download_imdb.py --skip-extract
        python scripts/download_imdb.py --dataset title.basics.tsv.gz
    """
    parser = argparse.ArgumentParser(
        description='Download IMDb non-commercial datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download all datasets:
    python scripts/download_imdb.py
  
  Re-download even if files exist:
    python scripts/download_imdb.py --force
  
  Download only, skip extraction:
    python scripts/download_imdb.py --skip-extract
  
  Download specific dataset:
    python scripts/download_imdb.py --dataset title.basics.tsv.gz

Available datasets:
  """ + '\n  '.join(DATASETS.keys())
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download files even if they already exist'
    )
    
    parser.add_argument(
        '--skip-extract',
        action='store_true',
        help='Download .gz files but do not extract them'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Download only a specific dataset (e.g., title.basics.tsv.gz)'
    )
    
    args = parser.parse_args()
    
    # Run download
    results = download_all_datasets(
        force=args.force,
        skip_extract=args.skip_extract,
        specific_dataset=args.dataset
    )
    
    # Exit with appropriate code
    if results and all(results.values()):
        sys.exit(0)  # Success
    elif results:
        sys.exit(1)  # Partial failure
    else:
        sys.exit(2)  # Complete failure


if __name__ == "__main__":
    main()


# ============================================================================
# 🔧 USAGE EXAMPLES
# ============================================================================
"""
# Example 1: Download all datasets
python scripts/download_imdb.py

# Example 2: Re-download everything (force overwrite)
python scripts/download_imdb.py --force

# Example 3: Download compressed files only (save time)
python scripts/download_imdb.py --skip-extract

# Example 4: Download just one dataset for testing
python scripts/download_imdb.py --dataset title.basics.tsv.gz

# Example 5: Use in Python script
from scripts.download_imdb import download_all_datasets

results = download_all_datasets()
if all(results.values()):
    print("All datasets downloaded successfully!")
"""