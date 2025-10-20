"""
============================================================================
FILMOTECA - Watchlist Matcher
============================================================================
Matches your Watched.csv to the IMDb database using tconst (IMDb ID).
Enriches your viewing history with complete IMDb data.

🎯 PURPOSE:
    - Load your Watched.csv (2,280 movies)
    - Match each movie to IMDb database using tconst
    - Enrich with additional data (alternative titles, cast, crew)
    - Generate enriched dataset for analysis
    - Identify any missing matches

🔧 USAGE:
    python scripts/match_watchlist.py [--output FORMAT]
    
    Options:
        --output FORMAT    Output format: json, csv, parquet (default: json)
        --verify          Run verification checks
        --sample N        Process only first N records for testing

📊 OUTPUT:
    - data/processed/watched_enriched.json (or csv/parquet)
    - data/processed/match_report.txt
    - Match statistics and quality report

============================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from sqlalchemy import text
from scripts.load_imdb_to_db import (
    engine, SessionLocal,
    TitleBasics, TitleAkas, TitleRatings, TitleCrew, 
    TitlePrincipals, NameBasics
)
import pandas as pd
from tqdm import tqdm


def load_watched_csv(file_path: Path) -> pd.DataFrame:
    """
    Load and clean the Watched.csv file.
    
    Returns:
        DataFrame with watched movies
    """
    print(f"\n📥 Loading watched history from {file_path.name}")
    
    df = pd.read_csv(file_path)
    
    print(f"   ✅ Loaded {len(df):,} watched movies")
    print(f"\n📋 Columns in Watched.csv:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   • {col}: {non_null:,}/{len(df):,} non-null")
    
    # Clean the tconst column (remove 'tt' prefix if present, we'll add it back)
    if 'Const' in df.columns:
        df['tconst'] = df['Const'].apply(lambda x: x if pd.isna(x) else str(x).strip())
        print(f"\n   ✅ Found tconst in 'Const' column")
    else:
        print(f"\n   ❌ No 'Const' column found!")
        print(f"   Available columns: {list(df.columns)}")
        return None
    
    return df


def match_to_imdb(watched_df: pd.DataFrame, session) -> pd.DataFrame:
    """
    Match watched movies to IMDb database using tconst.
    Enrich with IMDb data.
    
    Args:
        watched_df: DataFrame with watched movies
        session: SQLAlchemy session
        
    Returns:
        Enriched DataFrame with IMDb data
    """
    print(f"\n🔍 Matching {len(watched_df):,} movies to IMDb database...")
    
    results = []
    matched = 0
    not_matched = 0
    
    for idx, row in tqdm(watched_df.iterrows(), total=len(watched_df), desc="   Matching"):
        tconst = row['tconst']
        
        if pd.isna(tconst):
            not_matched += 1
            results.append({
                **row.to_dict(),
                'match_status': 'no_tconst',
                'imdb_data': None
            })
            continue
        
        # Query IMDb database for this tconst
        title = session.query(TitleBasics).filter(
            TitleBasics.tconst == tconst
        ).first()
        
        if not title:
            not_matched += 1
            results.append({
                **row.to_dict(),
                'match_status': 'not_found',
                'imdb_data': None
            })
            continue
        
        # Get rating
        rating = session.query(TitleRatings).filter(
            TitleRatings.tconst == tconst
        ).first()
        
        # Get crew (directors/writers)
        crew = session.query(TitleCrew).filter(
            TitleCrew.tconst == tconst
        ).first()
        
        # Get alternative titles (including original language)
        akas = session.query(TitleAkas).filter(
            TitleAkas.title_id == tconst,
            TitleAkas.is_original_title == True
        ).first()
        
        # Get top cast (limit to top 10)
        principals = session.query(
            TitlePrincipals.nconst,
            TitlePrincipals.category,
            TitlePrincipals.characters,
            NameBasics.primary_name
        ).join(
            NameBasics, TitlePrincipals.nconst == NameBasics.nconst
        ).filter(
            TitlePrincipals.tconst == tconst,
            TitlePrincipals.category.in_(['actor', 'actress'])
        ).order_by(
            TitlePrincipals.ordering
        ).limit(10).all()
        
        # Get directors
        director_names = []
        if crew and crew.directors:
            director_ids = [d.strip() for d in crew.directors.split(',') if d.strip()]
            directors = session.query(NameBasics).filter(
                NameBasics.nconst.in_(director_ids)
            ).all()
            director_names = [d.primary_name for d in directors]
        
        # Build enriched record
        enriched = {
            **row.to_dict(),
            'match_status': 'matched',
            
            # IMDb core data
            'imdb_title_type': title.title_type,
            'imdb_primary_title': title.primary_title,
            'imdb_original_title': title.original_title,
            'imdb_is_adult': title.is_adult,
            'imdb_start_year': title.start_year,
            'imdb_end_year': title.end_year,
            'imdb_runtime_minutes': title.runtime_minutes,
            'imdb_genres': title.genres,
            
            # Rating data
            'imdb_average_rating': rating.average_rating if rating else None,
            'imdb_num_votes': rating.num_votes if rating else None,
            
            # Crew data
            'imdb_directors': ','.join(director_names) if director_names else None,
            'imdb_writers': crew.writers if crew else None,
            
            # Original language title
            'imdb_original_language_title': akas.title if akas else None,
            'imdb_original_language': akas.language if akas else None,
            
            # Cast (top 10)
            'imdb_cast': [
                {
                    'name': p.primary_name,
                    'category': p.category,
                    'characters': p.characters
                }
                for p in principals
            ] if principals else []
        }
        
        results.append(enriched)
        matched += 1
    
    print(f"\n📊 Matching Results:")
    print(f"   ✅ Matched: {matched:,}/{len(watched_df):,} ({matched/len(watched_df)*100:.1f}%)")
    print(f"   ❌ Not matched: {not_matched:,}")
    
    return pd.DataFrame(results)


def analyze_matches(enriched_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the matching results and generate statistics.
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\n📈 Analyzing matches...")
    
    matched = enriched_df[enriched_df['match_status'] == 'matched']
    not_matched = enriched_df[enriched_df['match_status'] != 'matched']
    
    stats = {
        'total_movies': len(enriched_df),
        'matched': len(matched),
        'not_matched': len(not_matched),
        'match_rate': len(matched) / len(enriched_df) * 100,
        
        # Type distribution
        'type_distribution': matched['imdb_title_type'].value_counts().to_dict() if len(matched) > 0 else {},
        
        # Year range
        'year_range': {
            'min': int(matched['imdb_start_year'].min()) if len(matched) > 0 and matched['imdb_start_year'].notna().any() else None,
            'max': int(matched['imdb_start_year'].max()) if len(matched) > 0 and matched['imdb_start_year'].notna().any() else None,
            'avg': float(matched['imdb_start_year'].mean()) if len(matched) > 0 and matched['imdb_start_year'].notna().any() else None
        },
        
        # Rating comparison
        'rating_comparison': {
            'your_avg': float(matched['Your Rating'].mean()) if 'Your Rating' in matched.columns and matched['Your Rating'].notna().any() else None,
            'imdb_avg': float(matched['imdb_average_rating'].mean()) if len(matched) > 0 and matched['imdb_average_rating'].notna().any() else None
        },
        
        # Genre distribution
        'top_genres': {},
        
        # Data completeness
        'data_completeness': {
            'has_rating': matched['imdb_average_rating'].notna().sum(),
            'has_directors': matched['imdb_directors'].notna().sum(),
            'has_cast': matched['imdb_cast'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum(),
            'has_original_title': matched['imdb_original_language_title'].notna().sum()
        }
    }
    
    # Count genres (they're comma-separated)
    if len(matched) > 0 and matched['imdb_genres'].notna().any():
        genre_counts = {}
        for genres in matched['imdb_genres'].dropna():
            for genre in str(genres).split(','):
                genre = genre.strip()
                if genre:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        stats['top_genres'] = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Print matching statistics in a nice format."""
    print("\n" + "="*70)
    print("📊 MATCHING STATISTICS")
    print("="*70)
    
    print(f"\n✅ Overall:")
    print(f"   • Total movies: {stats['total_movies']:,}")
    print(f"   • Matched: {stats['matched']:,} ({stats['match_rate']:.1f}%)")
    print(f"   • Not matched: {stats['not_matched']:,}")
    
    if stats['type_distribution']:
        print(f"\n🎬 Content Types:")
        for title_type, count in stats['type_distribution'].items():
            pct = count / stats['matched'] * 100
            print(f"   • {title_type}: {count:,} ({pct:.1f}%)")
    
    if stats['year_range']['min']:
        print(f"\n📅 Year Range:")
        print(f"   • Oldest: {stats['year_range']['min']}")
        print(f"   • Newest: {stats['year_range']['max']}")
        print(f"   • Average: {stats['year_range']['avg']:.0f}")
    
    if stats['rating_comparison']['your_avg'] and stats['rating_comparison']['imdb_avg']:
        print(f"\n⭐ Rating Comparison:")
        print(f"   • Your average: {stats['rating_comparison']['your_avg']:.2f}/10")
        print(f"   • IMDb average: {stats['rating_comparison']['imdb_avg']:.2f}/10")
        diff = stats['rating_comparison']['your_avg'] - stats['rating_comparison']['imdb_avg']
        if diff > 0:
            print(f"   • You rate {abs(diff):.2f} points HIGHER than IMDb")
        else:
            print(f"   • You rate {abs(diff):.2f} points LOWER than IMDb")
    
    if stats['top_genres']:
        print(f"\n🎭 Top Genres:")
        for genre, count in list(stats['top_genres'].items())[:5]:
            pct = count / stats['matched'] * 100
            print(f"   • {genre}: {count:,} ({pct:.1f}%)")
    
    print(f"\n📋 Data Completeness:")
    for key, value in stats['data_completeness'].items():
        pct = value / stats['matched'] * 100 if stats['matched'] > 0 else 0
        print(f"   • {key.replace('_', ' ').title()}: {value:,}/{stats['matched']:,} ({pct:.1f}%)")


def save_enriched_data(df: pd.DataFrame, output_format: str = 'json'):
    """
    Save enriched data to file.
    
    Args:
        df: Enriched DataFrame
        output_format: Output format (json, csv, parquet)
    """
    output_dir = config.paths.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_format == 'json':
        output_file = output_dir / 'watched_enriched.json'
        df.to_json(output_file, orient='records', indent=2, force_ascii=False)
        print(f"\n💾 Saved enriched data to: {output_file}")
        
    elif output_format == 'csv':
        output_file = output_dir / 'watched_enriched.csv'
        # Convert list columns to JSON strings for CSV
        df_csv = df.copy()
        if 'imdb_cast' in df_csv.columns:
            df_csv['imdb_cast'] = df_csv['imdb_cast'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
        df_csv.to_csv(output_file, index=False)
        print(f"\n💾 Saved enriched data to: {output_file}")
        
    elif output_format == 'parquet':
        output_file = output_dir / 'watched_enriched.parquet'
        # Convert list columns to JSON strings for Parquet
        df_parquet = df.copy()
        if 'imdb_cast' in df_parquet.columns:
            df_parquet['imdb_cast'] = df_parquet['imdb_cast'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
        df_parquet.to_parquet(output_file, index=False)
        print(f"\n💾 Saved enriched data to: {output_file}")
    
    print(f"   📊 Format: {output_format.upper()}")
    print(f"   📏 Size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"   📝 Records: {len(df):,}")


def save_report(stats: Dict[str, Any]):
    """Save matching report to text file."""
    output_dir = config.paths.processed_data_dir
    report_file = output_dir / 'match_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FILMOTECA - WATCHLIST MATCHING REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        f.write(f"\n{'='*70}\n")
        f.write("OVERALL STATISTICS\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total movies: {stats['total_movies']:,}\n")
        f.write(f"Matched: {stats['matched']:,} ({stats['match_rate']:.1f}%)\n")
        f.write(f"Not matched: {stats['not_matched']:,}\n")
        
        if stats['type_distribution']:
            f.write(f"\n{'='*70}\n")
            f.write("CONTENT TYPES\n")
            f.write(f"{'='*70}\n")
            for title_type, count in stats['type_distribution'].items():
                pct = count / stats['matched'] * 100
                f.write(f"{title_type}: {count:,} ({pct:.1f}%)\n")
        
        if stats['year_range']['min']:
            f.write(f"\n{'='*70}\n")
            f.write("YEAR RANGE\n")
            f.write(f"{'='*70}\n")
            f.write(f"Oldest: {stats['year_range']['min']}\n")
            f.write(f"Newest: {stats['year_range']['max']}\n")
            f.write(f"Average: {stats['year_range']['avg']:.0f}\n")
        
        if stats['top_genres']:
            f.write(f"\n{'='*70}\n")
            f.write("TOP GENRES\n")
            f.write(f"{'='*70}\n")
            for genre, count in stats['top_genres'].items():
                pct = count / stats['matched'] * 100
                f.write(f"{genre}: {count:,} ({pct:.1f}%)\n")
        
        f.write(f"\n{'='*70}\n")
        f.write("DATA COMPLETENESS\n")
        f.write(f"{'='*70}\n")
        for key, value in stats['data_completeness'].items():
            pct = value / stats['matched'] * 100 if stats['matched'] > 0 else 0
            f.write(f"{key.replace('_', ' ').title()}: {value:,}/{stats['matched']:,} ({pct:.1f}%)\n")
    
    print(f"\n📄 Saved report to: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Match Watched.csv to IMDb database using tconst',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='json',
        choices=['json', 'csv', 'parquet'],
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Run verification checks'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        help='Process only first N records (for testing)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎬 FILMOTECA - Watchlist Matcher")
    print("="*70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load watched CSV
    watched_df = load_watched_csv(config.paths.watchlist_file)
    
    if watched_df is None:
        print("\n❌ Failed to load watched CSV")
        return
    
    # Sample if requested
    if args.sample:
        print(f"\n🔬 Testing mode: Processing first {args.sample} records")
        watched_df = watched_df.head(args.sample)
    
    # Match to IMDb
    session = SessionLocal()
    
    try:
        enriched_df = match_to_imdb(watched_df, session)
        
        # Analyze
        stats = analyze_matches(enriched_df)
        
        # Print statistics
        print_statistics(stats)
        
        # Save results
        save_enriched_data(enriched_df, args.output)
        save_report(stats)
        
        print("\n" + "="*70)
        print("✅ MATCHING COMPLETE!")
        print("="*70)
        print("\n🎯 Next steps:")
        print("   1. Review: data/processed/match_report.txt")
        print("   2. Explore: data/processed/watched_enriched.json")
        print("   3. Run: python scripts/analyze_data.py")
        print("      → Generate EDA visualizations")
        print("   4. Run: python scripts/recommend.py")
        print("      → Get movie recommendations")
        
    finally:
        session.close()
    
    print(f"\n📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()