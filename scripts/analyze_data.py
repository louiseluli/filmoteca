"""
============================================================================
FILMOTECA - Exploratory Data Analysis
============================================================================
Comprehensive analysis and visualization of your movie watching history.
Generates beautiful reports and insights from your enriched data.

🎯 PURPOSE:
    - Analyze 2,280 movies with IMDb data
    - Generate stunning visualizations
    - Discover your movie preferences
    - Compare your ratings to IMDb
    - Identify favorite directors, actors, genres
    - Track viewing patterns over time

📊 VISUALIZATIONS (20+ charts):
    1. Rating Analysis
       - Your ratings vs IMDb ratings
       - Rating distribution
       - Rating bias by genre
       - Rating evolution over time
    
    2. Genre Analysis
       - Genre distribution
       - Genre combinations
       - Genre preferences by decade
       - Word cloud of genres
    
    3. Temporal Analysis
       - Movies watched by year
       - Decade preferences
       - Release year distribution
       - Viewing timeline
    
    4. People Analysis
       - Top directors
       - Director ratings
       - Most frequent actors
       - Collaboration networks
    
    5. Content Analysis
       - Runtime distribution
       - Type distribution (movie, TV, etc.)
       - Language analysis
       - Popularity vs rating

🔧 USAGE:
    python scripts/analyze_data.py [--format html|pdf] [--open]
    
    Options:
        --format FORMAT   Output format: html or pdf (default: html)
        --open           Open report in browser after generation
        --export-figs    Export individual figures as PNG

📊 OUTPUT:
    - data/reports/eda/eda_report.html (interactive)
    - data/reports/eda/eda_report.pdf (printable)
    - data/reports/eda/figures/ (individual charts)
    
============================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import argparse
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_enriched_data() -> pd.DataFrame:
    """Load the enriched watched data."""
    print(f"\n📥 Loading enriched data...")
    
    data_file = config.paths.processed_data_dir / 'watched_enriched.json'
    
    if not data_file.exists():
        print(f"   ❌ File not found: {data_file}")
        print(f"   Run 'python scripts/match_watchlist.py' first!")
        return None
    
    df = pd.read_json(data_file)
    
    print(f"   ✅ Loaded {len(df):,} movies")
    print(f"   📊 Columns: {len(df.columns)}")
    print(f"   ✅ Matched: {(df['match_status'] == 'matched').sum():,}")
    
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for analysis."""
    print(f"\n🔧 Preparing data for analysis...")
    
    # Keep only matched movies
    df = df[df['match_status'] == 'matched'].copy()
    
    # Convert dates
    if 'Date Rated' in df.columns:
        df['Date Rated'] = pd.to_datetime(df['Date Rated'], errors='coerce')
    
    # Parse genres into lists
    df['genre_list'] = df['imdb_genres'].apply(
        lambda x: [g.strip() for g in str(x).split(',') if g.strip()] if pd.notna(x) else []
    )
    
    # Add decade
    df['decade'] = (df['imdb_start_year'] // 10 * 10).astype('Int64')
    
    # Parse directors
    df['director_list'] = df['imdb_directors'].apply(
        lambda x: [d.strip() for d in str(x).split(',') if d.strip()] if pd.notna(x) else []
    )
    
    # Calculate rating difference
    df['rating_diff'] = df['Your Rating'] - df['IMDb Rating']
    
    print(f"   ✅ Prepared {len(df):,} movies for analysis")
    
    return df


# ============================================================================
# RATING ANALYSIS
# ============================================================================

def plot_rating_comparison(df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Compare your ratings vs IMDb ratings."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎬 Rating Analysis: You vs IMDb', fontsize=20, fontweight='bold')
    
    # 1. Scatter plot: Your rating vs IMDb
    ax = axes[0, 0]
    ax.scatter(df['IMDb Rating'], df['Your Rating'], alpha=0.5, s=30, c='#FF6B6B')
    ax.plot([1, 10], [1, 10], 'k--', alpha=0.3, label='Perfect agreement')
    ax.set_xlabel('IMDb Rating', fontsize=12)
    ax.set_ylabel('Your Rating', fontsize=12)
    ax.set_title('Your Ratings vs IMDb Ratings', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution comparison
    ax = axes[0, 1]
    ax.hist(df['IMDb Rating'], bins=30, alpha=0.7, label='IMDb', color='#4ECDC4', edgecolor='black')
    ax.hist(df['Your Rating'].dropna(), bins=30, alpha=0.7, label='Your Ratings', color='#FF6B6B', edgecolor='black')
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Rating Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Rating difference distribution
    ax = axes[1, 0]
    rating_diff = df['rating_diff'].dropna()
    ax.hist(rating_diff, bins=30, color='#95E1D3', edgecolor='black', alpha=0.8)
    ax.axvline(rating_diff.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {rating_diff.mean():.2f}')
    ax.set_xlabel('Rating Difference (You - IMDb)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Rating Bias Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    📊 RATING STATISTICS
    
    Your Ratings:
    • Mean: {df['Your Rating'].mean():.2f}
    • Median: {df['Your Rating'].median():.2f}
    • Std Dev: {df['Your Rating'].std():.2f}
    • Rated movies: {df['Your Rating'].notna().sum():,}
    
    IMDb Ratings:
    • Mean: {df['IMDb Rating'].mean():.2f}
    • Median: {df['IMDb Rating'].median():.2f}
    • Std Dev: {df['IMDb Rating'].std():.2f}
    
    Rating Difference:
    • Mean: {rating_diff.mean():.2f}
    • You rate {'HIGHER' if rating_diff.mean() > 0 else 'LOWER'} than IMDb
    • Agreement: {(abs(rating_diff) < 1).sum() / len(rating_diff) * 100:.1f}% within 1 point
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig


def plot_genre_analysis(df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Analyze genre preferences."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎭 Genre Analysis', fontsize=20, fontweight='bold')
    
    # 1. Top genres
    ax = axes[0, 0]
    genre_counts = pd.Series([g for genres in df['genre_list'] for g in genres]).value_counts()
    top_genres = genre_counts.head(15)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_genres)))
    bars = ax.barh(range(len(top_genres)), top_genres.values, color=colors)
    ax.set_yticks(range(len(top_genres)))
    ax.set_yticklabels(top_genres.index)
    ax.set_xlabel('Number of Movies', fontsize=12)
    ax.set_title('Top 15 Genres', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, top_genres.values)):
        ax.text(count + 10, i, f'{count}', va='center', fontsize=10)
    
    # 2. Genre combinations
    ax = axes[0, 1]
    genre_combos = df['imdb_genres'].value_counts().head(15)
    ax.barh(range(len(genre_combos)), genre_combos.values, color='#FF6B6B')
    ax.set_yticks(range(len(genre_combos)))
    ax.set_yticklabels([combo[:30] + '...' if len(combo) > 30 else combo 
                        for combo in genre_combos.index], fontsize=9)
    ax.set_xlabel('Number of Movies', fontsize=12)
    ax.set_title('Top 15 Genre Combinations', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # 3. Genre word cloud
    ax = axes[1, 0]
    ax.axis('off')
    genre_text = ' '.join([g for genres in df['genre_list'] for g in genres])
    if genre_text:
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             colormap='viridis').generate(genre_text)
        ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title('Genre Word Cloud', fontsize=14, fontweight='bold')
    
    # 4. IMDb ratings by genre (not personal ratings)
    ax = axes[1, 1]
    genre_ratings = []
    for genre in top_genres.head(10).index:
        ratings = df[df['genre_list'].apply(lambda x: genre in x)]['IMDb Rating'].dropna()
        if len(ratings) > 0:
            genre_ratings.append((genre, ratings.mean(), len(ratings)))
    
    if genre_ratings:
        genre_ratings.sort(key=lambda x: x[1], reverse=True)
        genres, avg_ratings, counts = zip(*genre_ratings)
        
        colors_by_rating = plt.cm.RdYlGn(np.array(avg_ratings) / 10)
        bars = ax.barh(range(len(genres)), avg_ratings, color=colors_by_rating)
        ax.set_yticks(range(len(genres)))
        ax.set_yticklabels(genres)
        ax.set_xlabel('Average IMDb Rating', fontsize=12)
        ax.set_title('IMDb Average Rating by Genre', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.invert_yaxis()
        
        # Add labels
        for i, (bar, rating, count) in enumerate(zip(bars, avg_ratings, counts)):
            ax.text(rating + 0.1, i, f'{rating:.2f} ({count})', 
                   va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_temporal_analysis(df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Analyze temporal patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📅 Temporal Analysis', fontsize=20, fontweight='bold')
    
    # 1. Movies by release year
    ax = axes[0, 0]
    year_counts = df['imdb_start_year'].value_counts().sort_index()
    ax.fill_between(year_counts.index, year_counts.values, alpha=0.7, color='#4ECDC4')
    ax.plot(year_counts.index, year_counts.values, color='#FF6B6B', linewidth=2)
    ax.set_xlabel('Release Year', fontsize=12)
    ax.set_ylabel('Number of Movies', fontsize=12)
    ax.set_title('Movies by Release Year', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Movies by decade
    ax = axes[0, 1]
    decade_counts = df['decade'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(decade_counts)))
    bars = ax.bar(decade_counts.index.astype(str), decade_counts.values, 
                  color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Number of Movies', fontsize=12)
    ax.set_title('Movies by Decade', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Add count labels
    for bar, count in zip(bars, decade_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}', ha='center', va='bottom', fontsize=10)
    
    # 3. Average IMDb rating by decade (removed personal rating)
    ax = axes[1, 0]
    decade_ratings = df.groupby('decade')['IMDb Rating'].mean().sort_index()
    
    x = decade_ratings.index.astype(str)
    x_pos = np.arange(len(x))
    
    colors = plt.cm.RdYlGn(decade_ratings.values / 10)
    bars = ax.bar(x_pos, decade_ratings.values, color=colors, 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Average IMDb Rating', fontsize=12)
    ax.set_title('Average IMDb Rating by Decade', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rating in zip(bars, decade_ratings.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rating:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Runtime distribution
    ax = axes[1, 1]
    runtime = df['imdb_runtime_minutes'].dropna()
    ax.hist(runtime, bins=50, color='#95E1D3', edgecolor='black', alpha=0.8)
    ax.axvline(runtime.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {runtime.mean():.0f} min')
    ax.axvline(runtime.median(), color='blue', linestyle='--', linewidth=2,
              label=f'Median: {runtime.median():.0f} min')
    ax.set_xlabel('Runtime (minutes)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Runtime Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_people_analysis(df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Analyze directors and actors."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('👥 People Analysis', fontsize=20, fontweight='bold')
    
    # 1. Top directors by count
    ax = axes[0, 0]
    director_counts = pd.Series([d for directors in df['director_list'] for d in directors]).value_counts()
    top_directors = director_counts.head(15)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(top_directors)))
    bars = ax.barh(range(len(top_directors)), top_directors.values, color=colors)
    ax.set_yticks(range(len(top_directors)))
    ax.set_yticklabels(top_directors.index)
    ax.set_xlabel('Number of Movies', fontsize=12)
    ax.set_title('Top 15 Directors (by count)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, count) in enumerate(zip(bars, top_directors.values)):
        ax.text(count + 0.3, i, f'{count}', va='center', fontsize=10)
    
    # 2. Top directors by IMDb rating (replacing empty personal rating chart)
    ax = axes[0, 1]
    director_ratings = []
    for director in director_counts.head(30).index:
        movies = df[df['director_list'].apply(lambda x: director in x)]
        ratings = movies['IMDb Rating'].dropna()
        if len(ratings) >= 5:  # At least 5 movies for reliable average
            director_ratings.append((director, ratings.mean(), len(ratings)))
    
    if director_ratings:
        director_ratings.sort(key=lambda x: x[1], reverse=True)
        directors, avg_ratings, counts = zip(*director_ratings[:15])
        
        colors_by_rating = plt.cm.RdYlGn(np.array(avg_ratings) / 10)
        bars = ax.barh(range(len(directors)), avg_ratings, color=colors_by_rating)
        ax.set_yticks(range(len(directors)))
        ax.set_yticklabels(directors)
        ax.set_xlabel('Average IMDb Rating', fontsize=12)
        ax.set_title('Top Directors by IMDb Rating (≥5 movies)', fontsize=14, fontweight='bold')
        ax.set_xlim(5, 9)
        ax.invert_yaxis()
        
        for i, (bar, rating, count) in enumerate(zip(bars, avg_ratings, counts)):
            ax.text(rating + 0.05, i, f'{rating:.2f} ({count})', 
                   va='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Not enough data',
               ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    # 3. Content type distribution - BAR CHART (cleaner than pie for skewed data)
    ax = axes[1, 0]
    type_counts = df['imdb_title_type'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
    
    bars = ax.barh(range(len(type_counts)), type_counts.values, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_yticks(range(len(type_counts)))
    ax.set_yticklabels(type_counts.index, fontsize=11)
    ax.set_xlabel('Number of Movies', fontsize=12)
    ax.set_title('Content Type Distribution', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add count and percentage labels
    total = type_counts.sum()
    for i, (bar, count) in enumerate(zip(bars, type_counts.values)):
        percentage = count / total * 100
        ax.text(count + 20, i, f'{count} ({percentage:.1f}%)', 
               va='center', fontsize=10, fontweight='bold')
    
    # 4. Top actors (from cast) - PROPER BAR CHART
    ax = axes[1, 1]
    
    # Extract all actors
    all_actors = []
    for cast_list in df['imdb_cast']:
        if isinstance(cast_list, list):
            for person in cast_list:
                if isinstance(person, dict) and 'name' in person:
                    all_actors.append(person['name'])
    
    if all_actors:
        actor_counts = pd.Series(all_actors).value_counts().head(15)
        colors = plt.cm.Set2(np.linspace(0, 1, len(actor_counts)))
        bars = ax.barh(range(len(actor_counts)), actor_counts.values, color=colors)
        ax.set_yticks(range(len(actor_counts)))
        ax.set_yticklabels(actor_counts.index, fontsize=10)
        ax.set_xlabel('Number of Movies', fontsize=12)
        ax.set_title('Top 15 Actors (by appearances)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, actor_counts.values)):
            ax.text(count + 0.3, i, f'{count}', va='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No actor data available',
               ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_advanced_analysis(df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Advanced analysis visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📊 Content Analysis', fontsize=20, fontweight='bold')
    
    # 1. Popularity analysis
    ax = axes[0, 0]
    mask = df['Num Votes'].notna()
    scatter = ax.scatter(df[mask]['Num Votes'], df[mask]['IMDb Rating'],
                        c=df[mask]['imdb_start_year'], cmap='viridis',
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Number of Votes (log scale)', fontsize=12)
    ax.set_ylabel('IMDb Rating', fontsize=12)
    ax.set_title('Popularity vs IMDb Rating', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Release Year')
    ax.grid(True, alpha=0.3)
    
    # 2. Runtime by decade
    ax = axes[0, 1]
    decade_runtime = df.groupby('decade')['imdb_runtime_minutes'].agg(['mean', 'median']).sort_index()
    
    x = decade_runtime.index.astype(str)
    width = 0.35
    x_pos = np.arange(len(x))
    
    ax.bar(x_pos - width/2, decade_runtime['mean'], width,
           label='Mean', color='#FF6B6B', alpha=0.8)
    ax.bar(x_pos + width/2, decade_runtime['median'], width,
           label='Median', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Runtime (minutes)', fontsize=12)
    ax.set_title('Average Runtime by Decade', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Language distribution (fixed)
    ax = axes[1, 0]
    lang_counts = df['imdb_original_language'].value_counts().head(15)
    
    if len(lang_counts) > 0 and lang_counts.sum() > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, len(lang_counts)))
        bars = ax.barh(range(len(lang_counts)), lang_counts.values, color=colors)
        ax.set_yticks(range(len(lang_counts)))
        # Replace language codes with readable names
        lang_names = []
        for code in lang_counts.index:
            if pd.isna(code):
                lang_names.append('Unknown')
            else:
                lang_names.append(str(code).upper())
        ax.set_yticklabels(lang_names)
        ax.set_xlabel('Number of Movies', fontsize=12)
        ax.set_title('Top 15 Original Languages', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        for i, (bar, count) in enumerate(zip(bars, lang_counts.values)):
            ax.text(count + 5, i, f'{count}', va='center', fontsize=10)
    else:
        # If no language data, show content type vs decade
        type_decade = df.groupby(['decade', 'imdb_title_type']).size().unstack(fill_value=0)
        type_decade[type_decade.columns[:3]].plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
        ax.set_xlabel('Decade', fontsize=12)
        ax.set_ylabel('Number of Movies', fontsize=12)
        ax.set_title('Content Types by Decade', fontsize=14, fontweight='bold')
        ax.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate interesting stats
    oldest_movie = df.loc[df['imdb_start_year'].idxmin()]
    newest_movie = df.loc[df['imdb_start_year'].idxmax()]
    longest_movie = df.loc[df['imdb_runtime_minutes'].idxmax()]
    most_popular = df.loc[df['Num Votes'].idxmax()]
    
    stats_text = f"""
    🏆 INTERESTING FACTS
    
    Oldest Movie:
    • {oldest_movie['imdb_primary_title']} ({oldest_movie['imdb_start_year']})
    
    Newest Movie:
    • {newest_movie['imdb_primary_title']} ({newest_movie['imdb_start_year']})
    
    Longest Movie:
    • {longest_movie['imdb_primary_title']}
    • {longest_movie['imdb_runtime_minutes']:.0f} minutes
      ({longest_movie['imdb_runtime_minutes']/60:.1f} hours)
    
    Most Popular:
    • {most_popular['imdb_primary_title']}
    • {most_popular['Num Votes']:,.0f} votes
    
    Total Watch Time:
    • {df['imdb_runtime_minutes'].sum() / 60:.0f} hours
    • {df['imdb_runtime_minutes'].sum() / (60*24):.0f} days
    • {df['imdb_runtime_minutes'].sum() / (60*24*365):.1f} years
    """
    
    ax.text(0.1, 0.95, stats_text, fontsize=10, family='monospace',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
    
    plt.tight_layout()
    return fig


def generate_html_report(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate comprehensive HTML report."""
    print(f"\n📄 Generating HTML report...")
    
    html_file = output_dir / 'eda_report.html'
    
    # Calculate statistics
    stats = {
        'total_movies': len(df),
        'total_runtime_hours': df['imdb_runtime_minutes'].sum() / 60,
        'total_runtime_days': df['imdb_runtime_minutes'].sum() / (60*24),
        'avg_imdb_rating': df['IMDb Rating'].mean(),
        'oldest_year': int(df['imdb_start_year'].min()),
        'newest_year': int(df['imdb_start_year'].max()),
        'unique_directors': len(set([d for directors in df['director_list'] for d in directors])),
        'unique_genres': len(set([g for genres in df['genre_list'] for g in genres])),
        'unique_actors': len(set([a for cast in df['imdb_cast'] 
                                  for person in (cast if isinstance(cast, list) else [])
                                  for a in ([person.get('name')] if isinstance(person, dict) else [])])),
    }
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Filmoteca - EDA Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 60px 40px;
                text-align: center;
            }}
            
            h1 {{
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            
            .subtitle {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                padding: 40px;
                background: #f8f9fa;
            }}
            
            .stat-card {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }}
            
            .stat-number {{
                font-size: 2.5em;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 10px;
            }}
            
            .stat-label {{
                font-size: 1em;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            h2 {{
                color: #667eea;
                margin: 40px 0 20px 0;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
                font-size: 2em;
            }}
            
            .visualization {{
                margin: 30px 0;
                text-align: center;
            }}
            
            .visualization img {{
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            footer {{
                background: #2c3e50;
                color: white;
                padding: 30px;
                text-align: center;
            }}
            
            .emoji {{
                font-size: 1.5em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎬 Filmoteca</h1>
                <p class="subtitle">Exploratory Data Analysis Report</p>
                <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </header>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{stats['total_movies']:,}</div>
                    <div class="stat-label">Movies Watched</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-number">{stats['total_runtime_days']:.0f}</div>
                    <div class="stat-label">Days of Runtime</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-number">{stats['avg_imdb_rating']:.2f}</div>
                    <div class="stat-label">Avg IMDb Rating</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-number">{stats['unique_directors']:,}</div>
                    <div class="stat-label">Unique Directors</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-number">{stats['oldest_year']}</div>
                    <div class="stat-label">Oldest Movie</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-number">{stats['unique_actors']:,}</div>
                    <div class="stat-label">Unique Actors</div>
                </div>
            </div>
            
            <div class="content">
                <h2><span class="emoji">🎭</span> Genre Analysis</h2>
                <div class="visualization">
                    <img src="figures/genre_analysis.png" alt="Genre Analysis">
                </div>
                
                <h2><span class="emoji">📅</span> Temporal Analysis</h2>
                <div class="visualization">
                    <img src="figures/temporal_analysis.png" alt="Temporal Analysis">
                </div>
                
                <h2><span class="emoji">👥</span> People Analysis</h2>
                <div class="visualization">
                    <img src="figures/people_analysis.png" alt="People Analysis">
                </div>
                
                <h2><span class="emoji">📊</span> Content Analysis</h2>
                <div class="visualization">
                    <img src="figures/content_analysis.png" alt="Content Analysis">
                </div>
            </div>
            
            <footer>
                <p><strong>Filmoteca</strong> - Your Personal Movie Analytics Platform</p>
                <p>Data sourced from IMDb • Analysis powered by Python</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   ✅ HTML report saved: {html_file}")
    
    return html_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive EDA report',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='html',
        choices=['html', 'pdf'],
        help='Output format (default: html)'
    )
    
    parser.add_argument(
        '--open',
        action='store_true',
        help='Open report in browser after generation'
    )
    
    parser.add_argument(
        '--export-figs',
        action='store_true',
        help='Export individual figures as PNG'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎬 FILMOTECA - Exploratory Data Analysis")
    print("="*70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = config.paths.reports_dir / 'eda'
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_enriched_data()
    if df is None:
        return
    
    # Prepare data
    df = prepare_data(df)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n1️⃣  Genre Analysis...")
    fig1 = plot_genre_analysis(df, output_dir)
    fig1.savefig(figures_dir / 'genre_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("2️⃣  Temporal Analysis...")
    fig2 = plot_temporal_analysis(df, output_dir)
    fig2.savefig(figures_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("3️⃣  People Analysis...")
    fig3 = plot_people_analysis(df, output_dir)
    fig3.savefig(figures_dir / 'people_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("4️⃣  Content Analysis...")
    fig4 = plot_advanced_analysis(df, output_dir)
    fig4.savefig(figures_dir / 'content_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print("\n   ✅ All visualizations generated!")
    
    # Generate HTML report
    html_file = generate_html_report(df, output_dir)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📂 Output directory: {output_dir}")
    print(f"📄 HTML report: {html_file}")
    print(f"🖼️  Figures: {figures_dir}")
    print(f"📊 Total figures: 4")
    
    # Open in browser if requested
    if args.open:
        import webbrowser
        webbrowser.open(f'file://{html_file.absolute()}')
        print(f"\n🌐 Opened in browser")
    
    print(f"\n📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()