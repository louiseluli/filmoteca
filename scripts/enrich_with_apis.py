"""
============================================================================
FILMOTECA - Multi-Source API Enrichment
============================================================================
Enriches movie data with TMDB, Does the Dog Die, and Wikidata APIs.
Adds posters, keywords, content warnings, and cultural context.

🎯 PURPOSE:
    - Enhance 2,280 movies with rich API data
    - TMDB: Posters, trailers, keywords, similar movies
    - DDD: Content warnings and trigger information
    - Wikidata: Awards, relationships, cultural context
    - Prepare for advanced recommendation engine

📊 ENRICHMENT SOURCES:
    1. TMDB (The Movie Database)
       - Movie posters and backdrops
       - Plot keywords for similarity
       - Cast photos
       - Trailers and videos
       - TMDB's own recommendations
       - Production companies
    
    2. Does the Dog Die
       - Content warnings (violence, death, etc.)
       - Trigger warnings
       - Sensitive topics
       - Community ratings
    
    3. Wikidata (Optional)
       - Awards received
       - Cultural significance
       - Box office data
       - Critical reception

🔧 USAGE:
    python scripts/enrich_with_apis.py [--source all|tmdb|ddd|wikidata]
    
    Options:
        --source SOURCE   Which APIs to use (default: all)
        --sample N        Test with N movies first
        --force           Re-enrich even if cached
        --skip-images     Don't download poster images

📊 OUTPUT:
    - data/processed/watched_fully_enriched.json
    - data/processed/posters/ (poster images)
    - data/processed/enrichment_report.txt
    
============================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse
import json
import time
import requests
from urllib.parse import quote

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from tqdm import tqdm
import pandas as pd


# ============================================================================
# API CLIENTS
# ============================================================================

class TMDBClient:
    """Client for The Movie Database API."""
    
    def __init__(self, api_key: str, read_token: str):
        self.api_key = api_key
        self.read_token = read_token
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {read_token}',
            'Content-Type': 'application/json;charset=utf-8'
        })
        self.rate_limit = 40  # 40 requests per 10 seconds
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.25:  # 4 requests per second
            time.sleep(0.25 - elapsed)
        self.last_request_time = time.time()
    
    def get_movie_by_imdb(self, imdb_id: str) -> Optional[Dict]:
        """Get TMDB movie data using IMDb ID."""
        self._rate_limit_wait()
        
        try:
            # Find movie by IMDb ID
            response = self.session.get(
                f"{self.base_url}/find/{imdb_id}",
                params={'external_source': 'imdb_id'}
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('movie_results'):
                return data['movie_results'][0]
            return None
            
        except Exception as e:
            print(f"Error fetching TMDB for {imdb_id}: {e}")
            return None
    
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Get detailed movie information."""
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                f"{self.base_url}/movie/{tmdb_id}",
                params={
                    'append_to_response': 'keywords,videos,recommendations,credits,images,external_ids'
                }
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Error fetching details for TMDB ID {tmdb_id}: {e}")
            return None
    
    def get_person_details(self, tmdb_person_id: int) -> Optional[Dict]:
        """Get person (actor/director) details."""
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                f"{self.base_url}/person/{tmdb_person_id}",
                params={
                    'append_to_response': 'images,movie_credits,external_ids'
                }
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Error fetching person {tmdb_person_id}: {e}")
            return None


class DDDClient:
    """Client for Does the Dog Die API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.doesthedogdie.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'X-API-KEY': api_key
        })
    
    def search_by_imdb(self, imdb_id: str) -> Optional[Dict]:
        """Search for content warnings by IMDb ID."""
        try:
            response = self.session.get(
                f"{self.base_url}/dddsearch",
                params={'imdb': imdb_id},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('items'):
                return data['items'][0]
            return None
            
        except Exception as e:
            print(f"Error fetching DDD for {imdb_id}: {e}")
            return None
    
    def get_media_details(self, ddd_id: int) -> Optional[Dict]:
        """Get detailed content warnings for a movie."""
        try:
            response = self.session.get(
                f"{self.base_url}/media/{ddd_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Error fetching DDD details for {ddd_id}: {e}")
            return None


class WikidataClient:
    """Client for Wikidata API (optional)."""
    
    def __init__(self):
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.session = requests.Session()
    
    def get_entity_by_imdb(self, imdb_id: str) -> Optional[Dict]:
        """Get Wikidata entity by IMDb ID."""
        try:
            # Search for entity with this IMDb ID
            response = self.session.get(
                self.base_url,
                params={
                    'action': 'wbgetentities',
                    'props': 'claims|labels',
                    'ids': f'P345:{imdb_id}',
                    'format': 'json'
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if 'entities' in data:
                return data['entities']
            return None
            
        except Exception as e:
            print(f"Error fetching Wikidata for {imdb_id}: {e}")
            return None


# ============================================================================
# ENRICHMENT LOGIC
# ============================================================================

def enrich_with_tmdb(movie: Dict, tmdb_client: TMDBClient) -> Dict:
    """Enrich a single movie with TMDB data."""
    imdb_id = movie.get('tconst')
    if not imdb_id:
        return movie
    
    # Get TMDB movie
    tmdb_movie = tmdb_client.get_movie_by_imdb(imdb_id)
    if not tmdb_movie:
        return movie
    
    tmdb_id = tmdb_movie.get('id')
    
    # Get detailed data
    details = tmdb_client.get_movie_details(tmdb_id)
    if not details:
        movie['tmdb_basic'] = tmdb_movie
        return movie
    
    # Extract useful data
    enrichment = {
        'tmdb_id': tmdb_id,
        'tmdb_poster_path': details.get('poster_path'),
        'tmdb_backdrop_path': details.get('backdrop_path'),
        'tmdb_overview': details.get('overview'),
        'tmdb_tagline': details.get('tagline'),
        'tmdb_homepage': details.get('homepage'),
        'tmdb_budget': details.get('budget'),
        'tmdb_revenue': details.get('revenue'),
        'tmdb_popularity': details.get('popularity'),
        'tmdb_vote_average': details.get('vote_average'),
        'tmdb_vote_count': details.get('vote_count'),
        
        # Keywords for similarity
        'tmdb_keywords': [
            kw.get('name') 
            for kw in details.get('keywords', {}).get('keywords', [])
        ],
        
        # Videos (trailers)
        'tmdb_videos': [
            {
                'key': v.get('key'),
                'site': v.get('site'),
                'type': v.get('type'),
                'name': v.get('name')
            }
            for v in details.get('videos', {}).get('results', [])
            if v.get('site') == 'YouTube' and v.get('type') in ['Trailer', 'Teaser']
        ],
        
        # TMDB recommendations
        'tmdb_recommendations': [
            {
                'id': rec.get('id'),
                'title': rec.get('title'),
                'vote_average': rec.get('vote_average')
            }
            for rec in details.get('recommendations', {}).get('results', [])[:10]
        ],
        
        # Production info
        'tmdb_production_companies': [
            pc.get('name')
            for pc in details.get('production_companies', [])
        ],
        'tmdb_production_countries': [
            pc.get('name')
            for pc in details.get('production_countries', [])
        ],
        
        # Collection info
        'tmdb_collection': details.get('belongs_to_collection', {}).get('name') if details.get('belongs_to_collection') else None,
    }
    
    movie.update(enrichment)
    return movie


def enrich_with_ddd(movie: Dict, ddd_client: DDDClient) -> Dict:
    """Enrich a single movie with Does the Dog Die data."""
    imdb_id = movie.get('tconst')
    if not imdb_id:
        return movie
    
    # Search DDD
    ddd_item = ddd_client.search_by_imdb(imdb_id)
    if not ddd_item:
        return movie
    
    ddd_id = ddd_item.get('id')
    
    # Get detailed warnings
    details = ddd_client.get_media_details(ddd_id)
    if not details:
        return movie
    
    # Extract content warnings
    topic_stats = details.get('topicItemStats', [])
    
    warnings = {}
    for topic in topic_stats:
        topic_info = topic.get('topic', {})
        topic_name = topic_info.get('name')
        
        if topic_name:
            warnings[topic_name] = {
                'yes_votes': topic.get('yesSum', 0),
                'no_votes': topic.get('noSum', 0),
                'is_yes': topic.get('isYes', 0) == 1,
                'comment': topic.get('comment'),
                'is_spoiler': topic_info.get('isSpoiler', False)
            }
    
    enrichment = {
        'ddd_id': ddd_id,
        'ddd_warnings': warnings,
        'ddd_warning_count': len([w for w in warnings.values() if w['is_yes']]),
        'ddd_major_warnings': [
            name for name, data in warnings.items()
            if data['is_yes'] and data['yes_votes'] > 10
        ]
    }
    
    movie.update(enrichment)
    return movie


def enrich_with_wikidata(movie: Dict, wikidata_client: WikidataClient) -> Dict:
    """Enrich a single movie with Wikidata."""
    imdb_id = movie.get('tconst')
    if not imdb_id:
        return movie
    
    entity = wikidata_client.get_entity_by_imdb(imdb_id)
    if not entity:
        return movie
    
    # Extract relevant Wikidata info
    # This is simplified - Wikidata has extensive data
    enrichment = {
        'wikidata_id': list(entity.keys())[0] if entity else None,
    }
    
    movie.update(enrichment)
    return movie


def download_poster(poster_path: str, output_dir: Path, filename: str) -> Optional[str]:
    """Download a poster image from TMDB."""
    if not poster_path:
        return None
    
    try:
        url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        filepath = output_dir / f"{filename}.jpg"
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return str(filepath)
        
    except Exception as e:
        print(f"Error downloading poster: {e}")
        return None


# ============================================================================
# MAIN ENRICHMENT PROCESS
# ============================================================================

def load_enriched_data() -> pd.DataFrame:
    """Load the basic enriched data."""
    data_file = config.paths.processed_data_dir / 'watched_enriched.json'
    
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        print(f"Run 'python scripts/match_watchlist.py' first!")
        return None
    
    df = pd.read_json(data_file)
    print(f"✅ Loaded {len(df):,} movies")
    
    return df


def enrich_movies(
    df: pd.DataFrame,
    sources: List[str] = ['tmdb', 'ddd'],
    sample: Optional[int] = None,
    skip_images: bool = False
) -> pd.DataFrame:
    """Enrich all movies with API data."""
    
    # Initialize clients
    clients = {}
    
    if 'tmdb' in sources and config.api.has_tmdb():
        print("✅ TMDB API configured")
        clients['tmdb'] = TMDBClient(
            config.api.tmdb_api_key,
            config.api.tmdb_read_token
        )
    else:
        print("⚠️  TMDB API not configured")
    
    if 'ddd' in sources and config.api.has_ddd():
        print("✅ Does the Dog Die API configured")
        clients['ddd'] = DDDClient(config.api.ddd_api_key)
    else:
        print("⚠️  DDD API not configured")
    
    if 'wikidata' in sources:
        print("✅ Wikidata enabled")
        clients['wikidata'] = WikidataClient()
    
    if not clients:
        print("❌ No API clients configured!")
        return df
    
    # Create poster directory
    poster_dir = config.paths.processed_data_dir / 'posters'
    poster_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample if requested
    if sample:
        df = df.head(sample)
        print(f"\n🔬 Testing with {sample} movies")
    
    # Enrich each movie
    enriched_movies = []
    
    print(f"\n{'='*70}")
    print("🔄 ENRICHING MOVIES")
    print(f"{'='*70}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enriching"):
        movie = row.to_dict()
        
        # TMDB enrichment
        if 'tmdb' in clients:
            movie = enrich_with_tmdb(movie, clients['tmdb'])
            
            # Download poster
            if not skip_images and movie.get('tmdb_poster_path'):
                poster_path = download_poster(
                    movie['tmdb_poster_path'],
                    poster_dir,
                    movie['tconst']
                )
                if poster_path:
                    movie['local_poster_path'] = poster_path
        
        # DDD enrichment
        if 'ddd' in clients:
            movie = enrich_with_ddd(movie, clients['ddd'])
        
        # Wikidata enrichment
        if 'wikidata' in clients:
            movie = enrich_with_wikidata(movie, clients['wikidata'])
        
        enriched_movies.append(movie)
    
    return pd.DataFrame(enriched_movies)


def generate_enrichment_report(df: pd.DataFrame) -> str:
    """Generate enrichment statistics report."""
    
    report = []
    report.append("="*70)
    report.append("ENRICHMENT REPORT")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total movies: {len(df):,}")
    
    # TMDB stats
    if 'tmdb_id' in df.columns:
        tmdb_count = df['tmdb_id'].notna().sum()
        poster_count = df['tmdb_poster_path'].notna().sum()
        keyword_count = df['tmdb_keywords'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        
        report.append(f"\n{'='*70}")
        report.append("TMDB ENRICHMENT")
        report.append(f"{'='*70}")
        report.append(f"Movies matched: {tmdb_count:,} ({tmdb_count/len(df)*100:.1f}%)")
        report.append(f"Posters available: {poster_count:,} ({poster_count/len(df)*100:.1f}%)")
        report.append(f"Total keywords: {keyword_count:,}")
        report.append(f"Avg keywords per movie: {keyword_count/tmdb_count:.1f}")
    
    # DDD stats
    if 'ddd_id' in df.columns:
        ddd_count = df['ddd_id'].notna().sum()
        warning_movies = df['ddd_warning_count'].apply(lambda x: x > 0 if pd.notna(x) else False).sum()
        
        report.append(f"\n{'='*70}")
        report.append("DOES THE DOG DIE ENRICHMENT")
        report.append(f"{'='*70}")
        report.append(f"Movies matched: {ddd_count:,} ({ddd_count/len(df)*100:.1f}%)")
        report.append(f"Movies with warnings: {warning_movies:,}")
        
        if 'ddd_major_warnings' in df.columns:
            all_warnings = []
            for warnings in df['ddd_major_warnings'].dropna():
                if isinstance(warnings, list):
                    all_warnings.extend(warnings)
            
            if all_warnings:
                from collections import Counter
                top_warnings = Counter(all_warnings).most_common(10)
                report.append(f"\nTop 10 warnings:")
                for warning, count in top_warnings:
                    report.append(f"  • {warning}: {count:,} movies")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Enrich movies with TMDB, DDD, and Wikidata APIs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='all',
        help='Which APIs to use: all, tmdb, ddd, wikidata (comma-separated)'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        help='Test with N movies first'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-enrich even if already enriched'
    )
    
    parser.add_argument(
        '--skip-images',
        action='store_true',
        help="Don't download poster images"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎬 FILMOTECA - API Enrichment")
    print("="*70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse sources
    if args.source == 'all':
        sources = ['tmdb', 'ddd']
    else:
        sources = [s.strip() for s in args.source.split(',')]
    
    print(f"\n🔧 Enrichment sources: {', '.join(sources)}")
    
    # Load data
    df = load_enriched_data()
    if df is None:
        return
    
    # Enrich
    enriched_df = enrich_movies(
        df,
        sources=sources,
        sample=args.sample,
        skip_images=args.skip_images
    )
    
    # Save
    output_file = config.paths.processed_data_dir / 'watched_fully_enriched.json'
    enriched_df.to_json(output_file, orient='records', indent=2, force_ascii=False)
    
    print(f"\n💾 Saved enriched data to: {output_file}")
    print(f"   📊 Format: JSON")
    print(f"   📏 Size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"   📝 Records: {len(enriched_df):,}")
    
    # Generate report
    report = generate_enrichment_report(enriched_df)
    print(f"\n{report}")
    
    # Save report
    report_file = config.paths.processed_data_dir / 'enrichment_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Saved report to: {report_file}")
    
    print("\n" + "="*70)
    print("✅ ENRICHMENT COMPLETE!")
    print("="*70)
    print("\n🎯 Next steps:")
    print("   1. Review enrichment_report.txt")
    print("   2. Run: python scripts/build_recommendations.py")
    print("      → Generate personalized recommendations")
    
    print(f"\n📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()