"""
============================================================================
FILMOTECA - Deep Taste Profile Analysis
============================================================================
Analyzes your 2,280 watched movies to extract your true cinematic DNA.
Identifies patterns beyond basic genres - your specific niches and preferences.

🎯 ANALYZES:
    - Temporal preferences (which decades you love)
    - Genre combinations (Screwball, Pre-Code, Shaw Brothers)
    - Keyword clusters (themes you gravitate toward)
    - Director/Actor networks (creative teams you follow)
    - Hidden patterns (the "trash" factor, cult classics)
    
============================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config


class TasteProfileAnalyzer:
    """Extract the essence of your cinematic taste."""
    
    def __init__(self, enriched_data_path: Path = None):
        """Initialize with enriched movie data."""
        if enriched_data_path is None:
            enriched_data_path = config.paths.processed_data_dir / 'watched_fully_enriched.json'
        
        print(f"🎬 Loading your movie history...")
        self.df = pd.read_json(enriched_data_path)
        print(f"   ✅ Loaded {len(self.df):,} movies")
        
        # Your stated preferences
        self.stated_preferences = {
            'screwball': ['romantic comedy', 'mistaken identity', 'fast-talking', 'madcap'],
            'pre_code': ['pre-code', 'risque', '1930s', 'provocative'],
            'early_scifi': ['alien', 'robot', 'dystopia', 'time travel', 'space'],
            'classic_horror': ['monster', 'vampire', 'gothic', 'haunted', 'supernatural'],
            'kung_fu': ['martial arts', 'kung fu', 'shaw brothers', 'revenge', 'honor'],
            'trash': ['b-movie', 'exploitation', 'cult', 'grindhouse', 'camp']
        }
        
    def analyze_temporal_preferences(self) -> Dict:
        """Analyze your preference for different eras."""
        print("\n📅 Analyzing Temporal Preferences...")
        
        # Group by decade
        self.df['decade'] = (self.df['imdb_start_year'] // 10) * 10
        decade_counts = self.df['decade'].value_counts().sort_index()
        
        # Identify peak decades
        peak_decades = decade_counts.nlargest(5)
        
        # Special era detection
        eras = {
            'pre_code': len(self.df[(self.df['imdb_start_year'] >= 1929) & 
                                   (self.df['imdb_start_year'] <= 1934)]),
            'classic_hollywood': len(self.df[(self.df['imdb_start_year'] >= 1935) & 
                                            (self.df['imdb_start_year'] <= 1960)]),
            'new_hollywood': len(self.df[(self.df['imdb_start_year'] >= 1967) & 
                                        (self.df['imdb_start_year'] <= 1982)]),
            'video_nasty_era': len(self.df[(self.df['imdb_start_year'] >= 1978) & 
                                          (self.df['imdb_start_year'] <= 1985) &
                                          (self.df['imdb_genres'].str.contains('Horror', na=False))]),
            'shaw_brothers_golden': len(self.df[(self.df['imdb_start_year'] >= 1970) & 
                                               (self.df['imdb_start_year'] <= 1985) &
                                               (self.df['tmdb_keywords'].apply(
                                                   lambda x: any('martial' in str(k).lower() or 
                                                               'kung' in str(k).lower() 
                                                               for k in (x if isinstance(x, list) else []))
                                               ))])
        }
        
        # Calculate era intensity (movies per year in that era)
        era_intensity = {}
        for era, count in eras.items():
            if era == 'pre_code':
                years = 6  # 1929-1934
            elif era == 'classic_hollywood':
                years = 26  # 1935-1960
            elif era == 'new_hollywood':
                years = 16  # 1967-1982
            elif era == 'video_nasty_era':
                years = 8  # 1978-1985
            elif era == 'shaw_brothers_golden':
                years = 16  # 1970-1985
            else:
                years = 1
            
            era_intensity[era] = count / years
        
        analysis = {
            'decade_distribution': decade_counts.to_dict(),
            'peak_decades': peak_decades.to_dict(),
            'special_eras': eras,
            'era_intensity': era_intensity,
            'oldest_movie': int(self.df['imdb_start_year'].min()),
            'newest_movie': int(self.df['imdb_start_year'].max()),
            'median_year': int(self.df['imdb_start_year'].median()),
            'classic_ratio': len(self.df[self.df['imdb_start_year'] < 1970]) / len(self.df)
        }
        
        print(f"   📊 Peak decades: {', '.join(f'{d}s' for d in peak_decades.index[:3])}")
        print(f"   🎭 Pre-Code films (1929-1934): {eras['pre_code']}")
        print(f"   🥋 Shaw Brothers era films: {eras['shaw_brothers_golden']}")
        print(f"   📼 Video Nasty era horror: {eras['video_nasty_era']}")
        
        return analysis
    
    def analyze_genre_combinations(self) -> Dict:
        """Find your unique genre combinations."""
        print("\n🎭 Analyzing Genre Combinations...")
        
        # Single genres
        all_genres = []
        for genres in self.df['imdb_genres'].dropna():
            if isinstance(genres, str):
                all_genres.extend([g.strip() for g in genres.split(',')])
        
        genre_counts = Counter(all_genres)
        
        # Genre combinations
        combo_counts = self.df['imdb_genres'].value_counts().head(20)
        
        # Unique combinations (your special tastes)
        unique_combos = []
        for combo in combo_counts.index[:50]:
            if 'Comedy' in combo and 'Romance' in combo and 'Drama' not in combo:
                # Potential screwball
                unique_combos.append(('screwball_potential', combo))
            if 'Horror' in combo and ('Sci-Fi' in combo or 'Thriller' in combo):
                # Classic horror-scifi
                unique_combos.append(('horror_hybrid', combo))
            if 'Action' in combo and 'Drama' in combo and 'History' not in combo:
                # Potential kung fu
                unique_combos.append(('action_drama', combo))
        
        # Genre diversity score
        total_movies = len(self.df)
        genre_diversity = len(genre_counts) / total_movies
        
        analysis = {
            'top_genres': dict(genre_counts.most_common(15)),
            'top_combinations': combo_counts.head(15).to_dict(),
            'unique_combinations': unique_combos[:10],
            'genre_diversity_score': genre_diversity,
            'horror_percentage': genre_counts.get('Horror', 0) / total_movies,
            'comedy_percentage': genre_counts.get('Comedy', 0) / total_movies,
            'action_percentage': genre_counts.get('Action', 0) / total_movies
        }
        
        print(f"   🎬 Top 3 genres: {', '.join(list(genre_counts.keys())[:3])}")
        print(f"   💀 Horror films: {genre_counts.get('Horror', 0)} ({analysis['horror_percentage']*100:.1f}%)")
        print(f"   🎭 Genre diversity: {genre_diversity:.3f}")
        
        return analysis
    
    def analyze_keywords_themes(self) -> Dict:
        """Extract thematic preferences from TMDB keywords."""
        print("\n🏷️ Analyzing Thematic Keywords...")
        
        # Extract all keywords
        all_keywords = []
        keyword_by_decade = defaultdict(list)
        
        for _, movie in self.df.iterrows():
            if isinstance(movie['tmdb_keywords'], list):
                keywords = movie['tmdb_keywords']
                all_keywords.extend(keywords)
                
                decade = (movie['imdb_start_year'] // 10) * 10
                keyword_by_decade[decade].extend(keywords)
        
        keyword_counts = Counter(all_keywords)
        
        # Identify thematic clusters
        theme_clusters = {
            'screwball_indicators': [k for k, v in keyword_counts.items() 
                                    if any(term in k.lower() for term in 
                                         ['mistaken', 'romantic', 'comedy', 'marriage', 'divorce'])],
            'horror_themes': [k for k, v in keyword_counts.items() 
                            if any(term in k.lower() for term in 
                                 ['monster', 'vampire', 'zombie', 'haunted', 'supernatural', 'slasher'])],
            'scifi_themes': [k for k, v in keyword_counts.items() 
                           if any(term in k.lower() for term in 
                                ['alien', 'robot', 'time', 'space', 'future', 'dystopia'])],
            'martial_arts': [k for k, v in keyword_counts.items() 
                           if any(term in k.lower() for term in 
                                ['martial', 'kung', 'fight', 'revenge', 'honor', 'sword'])],
            'cult_trash': [k for k, v in keyword_counts.items() 
                         if any(term in k.lower() for term in 
                              ['cult', 'exploitation', 'camp', 'b-movie', 'grindhouse'])]
        }
        
        # Calculate theme strengths
        theme_strengths = {}
        for theme, keywords in theme_clusters.items():
            if keywords:
                theme_strengths[theme] = sum(keyword_counts[k] for k in keywords) / len(self.df)
        
        analysis = {
            'top_keywords': dict(keyword_counts.most_common(30)),
            'total_unique_keywords': len(keyword_counts),
            'avg_keywords_per_movie': len(all_keywords) / len(self.df),
            'theme_clusters': {k: v[:10] for k, v in theme_clusters.items()},
            'theme_strengths': theme_strengths,
            'decade_themes': {
                decade: Counter(keywords).most_common(5) 
                for decade, keywords in keyword_by_decade.items() 
                if decade in [1930, 1940, 1970, 1980, 2000]
            }
        }
        
        print(f"   🏷️ Total unique keywords: {len(keyword_counts):,}")
        print(f"   📊 Avg keywords per movie: {analysis['avg_keywords_per_movie']:.1f}")
        print(f"   👻 Horror theme strength: {theme_strengths.get('horror_themes', 0):.3f}")
        print(f"   🥋 Martial arts strength: {theme_strengths.get('martial_arts', 0):.3f}")
        
        return analysis
    
    def analyze_creator_networks(self) -> Dict:
        """Build collaboration networks of directors and actors."""
        print("\n🎬 Analyzing Creator Networks...")
        
        # Extract directors
        director_counts = Counter()
        director_movies = defaultdict(list)
        
        for _, movie in self.df.iterrows():
            if pd.notna(movie.get('imdb_directors')):
                directors = str(movie['imdb_directors']).split(',')
                for director in directors:
                    director = director.strip()
                    if director:
                        director_counts[director] += 1
                        director_movies[director].append(movie['imdb_primary_title'])
        
        # Extract top actors from cast
        actor_counts = Counter()
        actor_collaborations = defaultdict(set)
        
        for _, movie in self.df.iterrows():
            if isinstance(movie.get('imdb_cast'), list):
                # Get top 3 actors from each movie
                top_actors = [p['name'] for p in movie['imdb_cast'][:3] 
                             if isinstance(p, dict) and 'name' in p]
                
                for actor in top_actors:
                    actor_counts[actor] += 1
                    
                    # Track collaborations
                    if pd.notna(movie.get('imdb_directors')):
                        directors = str(movie['imdb_directors']).split(',')
                        for director in directors:
                            actor_collaborations[actor].add(director.strip())
        
        # Find frequent collaborations
        frequent_pairs = []
        for actor, directors in actor_collaborations.items():
            if actor_counts[actor] >= 3:  # Actor appears in 3+ movies
                for director in directors:
                    if director_counts[director] >= 3:  # Director has 3+ movies
                        collab_count = sum(1 for _, m in self.df.iterrows() 
                                         if pd.notna(m.get('imdb_directors')) and 
                                         director in str(m['imdb_directors']) and
                                         isinstance(m.get('imdb_cast'), list) and
                                         any(p.get('name') == actor for p in m['imdb_cast'][:5] 
                                            if isinstance(p, dict)))
                        if collab_count >= 2:
                            frequent_pairs.append((actor, director, collab_count))
        
        frequent_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Identify auteur preferences
        auteur_directors = [d for d, count in director_counts.most_common(20) if count >= 5]
        
        analysis = {
            'top_directors': dict(director_counts.most_common(15)),
            'top_actors': dict(actor_counts.most_common(20)),
            'auteur_directors': auteur_directors,
            'frequent_collaborations': frequent_pairs[:15],
            'total_unique_directors': len(director_counts),
            'total_unique_actors': len(actor_counts),
            'directors_with_multiple_films': len([d for d, c in director_counts.items() if c >= 2]),
            'actors_with_multiple_films': len([a for a, c in actor_counts.items() if c >= 3])
        }
        
        print(f"   🎬 Top directors: {', '.join(list(director_counts.keys())[:5])}")
        print(f"   ⭐ Top actors: {', '.join(list(actor_counts.keys())[:5])}")
        print(f"   🎭 Frequent collaborations found: {len(frequent_pairs)}")
        print(f"   🏆 Auteur directors (5+ films): {len(auteur_directors)}")
        
        return analysis
    
    def identify_hidden_patterns(self) -> Dict:
        """Find the hidden patterns - cult classics, trash cinema, etc."""
        print("\n🔍 Identifying Hidden Patterns...")
        
        # Low budget successes (potential "trash" cinema)
        trash_indicators = []
        cult_potential = []
        
        for _, movie in self.df.iterrows():
            # Low budget but you watched it
            if pd.notna(movie.get('tmdb_budget')) and movie['tmdb_budget'] < 1000000:
                trash_indicators.append({
                    'title': movie['imdb_primary_title'],
                    'year': movie['imdb_start_year'],
                    'budget': movie['tmdb_budget'],
                    'genres': movie['imdb_genres']
                })
            
            # Low votes but decent rating (cult classics)
            if (pd.notna(movie.get('Num Votes')) and 
                movie['Num Votes'] < 10000 and 
                pd.notna(movie.get('IMDb Rating')) and 
                movie['IMDb Rating'] >= 6.0):
                cult_potential.append({
                    'title': movie['imdb_primary_title'],
                    'year': movie['imdb_start_year'],
                    'rating': movie['IMDb Rating'],
                    'votes': movie['Num Votes']
                })
        
        # Runtime preferences (short vs epic)
        runtime_stats = self.df['imdb_runtime_minutes'].describe()
        short_films = len(self.df[self.df['imdb_runtime_minutes'] < 80])
        epic_films = len(self.df[self.df['imdb_runtime_minutes'] > 150])
        
        # Language diversity
        language_counts = self.df['imdb_original_language'].value_counts()
        non_english = len(self.df[self.df['imdb_original_language'] != 'en'])
        
        # Production company patterns (for Shaw Brothers detection)
        production_companies = []
        for companies in self.df['tmdb_production_companies'].dropna():
            if isinstance(companies, list):
                production_companies.extend(companies)
        
        company_counts = Counter(production_companies)
        potential_shaw = [c for c in company_counts.keys() 
                         if 'shaw' in c.lower() or 'brothers' in c.lower()]
        
        analysis = {
            'low_budget_films': len(trash_indicators),
            'cult_classics': len(cult_potential),
            'runtime_preferences': {
                'mean': runtime_stats['mean'],
                'median': runtime_stats['50%'],
                'short_films': short_films,
                'epic_films': epic_films
            },
            'language_diversity': {
                'total_languages': len(language_counts),
                'non_english_films': non_english,
                'non_english_percentage': non_english / len(self.df),
                'top_languages': language_counts.head(10).to_dict()
            },
            'production_patterns': {
                'top_companies': dict(company_counts.most_common(10)),
                'potential_shaw_brothers': potential_shaw
            },
            'obscurity_preference': len(cult_potential) / len(self.df)
        }
        
        print(f"   🎬 Low budget films: {len(trash_indicators)}")
        print(f"   💎 Potential cult classics: {len(cult_potential)}")
        print(f"   🌍 Non-English films: {non_english} ({non_english/len(self.df)*100:.1f}%)")
        print(f"   📏 Runtime preference: {runtime_stats['50%']:.0f} min median")
        
        return analysis
    
    def generate_taste_profile(self) -> Dict:
        """Generate comprehensive taste profile."""
        print("\n" + "="*70)
        print("🧬 GENERATING YOUR CINEMA DNA")
        print("="*70)
        
        profile = {
            'metadata': {
                'total_movies': len(self.df),
                'analysis_date': datetime.now().isoformat(),
                'data_quality': {
                    'tmdb_matched': len(self.df[self.df['tmdb_id'].notna()]) / len(self.df),
                    'keywords_available': len(self.df[self.df['tmdb_keywords'].notna()]) / len(self.df)
                }
            },
            'temporal': self.analyze_temporal_preferences(),
            'genres': self.analyze_genre_combinations(),
            'themes': self.analyze_keywords_themes(),
            'creators': self.analyze_creator_networks(),
            'patterns': self.identify_hidden_patterns()
        }
        
        # Generate summary
        print("\n" + "="*70)
        print("📊 YOUR CINEMATIC DNA SUMMARY")
        print("="*70)
        
        print("\n🎭 Core Identity:")
        print(f"   • Classic Film Enthusiast: {profile['temporal']['classic_ratio']*100:.1f}% pre-1970")
        print(f"   • Genre Explorer: {profile['genres']['genre_diversity_score']:.3f} diversity score")
        print(f"   • Cult Film Collector: {profile['patterns']['obscurity_preference']*100:.1f}% obscure films")
        
        print("\n📅 Temporal Sweet Spots:")
        for decade, count in list(profile['temporal']['peak_decades'].items())[:3]:
            print(f"   • {decade}s: {count} films")
        
        print("\n🎬 Auteur Following:")
        for director, count in list(profile['creators']['top_directors'].items())[:5]:
            print(f"   • {director}: {count} films")
        
        print("\n🏷️ Thematic Interests:")
        for theme, strength in profile['themes']['theme_strengths'].items():
            if strength > 0.01:
                print(f"   • {theme.replace('_', ' ').title()}: {strength:.3f} strength")
        
        return profile
    
    def save_profile(self, profile: Dict, output_path: Path = None):
        """Save taste profile to JSON."""
        if output_path is None:
            output_path = config.paths.processed_data_dir / 'taste_profile.json'
        
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        
        print(f"\n💾 Saved taste profile to: {output_path}")


def main():
    """Run taste profile analysis."""
    print("\n" + "="*70)
    print("🎬 FILMOTECA - Taste Profile Analysis")
    print("="*70)
    
    # Initialize analyzer
    analyzer = TasteProfileAnalyzer()
    
    # Generate comprehensive profile
    profile = analyzer.generate_taste_profile()
    
    # Save profile
    analyzer.save_profile(profile)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\n🎯 Your taste profile has been generated!")
    print("   This will power your personalized recommendations.")
    print("\n📂 Profile saved to: data/processed/taste_profile.json")
    print("\n🚀 Next step: Run build_recommendations.py")


if __name__ == "__main__":
    main()