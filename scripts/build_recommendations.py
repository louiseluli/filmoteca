"""
============================================================================
FILMOTECA - Content-DNA Hybrid Recommendation Engine
============================================================================
Generates personalized movie recommendations using:
- Content-DNA vectors (genres, keywords, era, creators)
- Creator collaboration networks (directors, actors)
- Your unique taste profile (no ratings needed!)

NO MINIMUM RATINGS OR VOTES - finds obscure gems!

🎯 RECOMMENDATION MODES:
    - DEEP DIVE: More of exactly what you love
    - DIVERSE: Explore across genres intelligently
    - CREATOR: Follow director/actor collaborations
    - HIDDEN GEMS: Obscure films matching your taste
    - TIME MACHINE: Era-specific recommendations
    - THEMATIC: Keyword-based discoveries
    
============================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import networkx as nx
from collections import defaultdict, Counter
import json
from typing import List, Dict, Tuple, Optional, Set
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from sqlalchemy import create_engine, text
from scripts.load_imdb_to_db import SessionLocal, TitleBasics, TitleRatings


class ContentDNAEngine:
    """Build Content-DNA vectors for every movie."""
    
    def __init__(self, watched_movies: pd.DataFrame, taste_profile: Dict):
        """
        Initialize with watched movies and taste profile.
        
        Args:
            watched_movies: DataFrame with enriched watched movies
            taste_profile: Generated taste profile from analyzer
        """
        self.watched = watched_movies
        self.taste_profile = taste_profile
        
        # Feature weights based on your priorities
        self.feature_weights = {
            'genres': 0.25,        # Priority 1
            'keywords': 0.22,      # Priority 2  
            'era': 0.18,          # Priority 3
            'directors': 0.15,    # Priority 4
            'actors': 0.12,       # Priority 5
            'runtime': 0.04,      # Additional
            'language': 0.04      # Additional
        }
        
        print(f"🧬 Initializing Content-DNA Engine...")
        print(f"   📊 Watched movies: {len(self.watched):,}")
        print(f"   🎯 Feature weights: {self.feature_weights}")
    
    def prepare_features(self):
        """Prepare all features for vectorization."""
        print("\n📐 Preparing features for DNA extraction...")
        
        # 1. Genres - TF-IDF vectorization
        print("   🎭 Processing genres...")
        self.watched['genre_string'] = self.watched['imdb_genres'].fillna('')
        self.genre_vectorizer = TfidfVectorizer(max_features=50)
        self.genre_matrix = self.genre_vectorizer.fit_transform(self.watched['genre_string'])
        
        # 2. Keywords - TF-IDF with higher weight for rare keywords
        print("   🏷️ Processing keywords...")
        self.watched['keyword_string'] = self.watched['tmdb_keywords'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
        self.keyword_vectorizer = TfidfVectorizer(max_features=200, min_df=2, max_df=0.8)
        self.keyword_matrix = self.keyword_vectorizer.fit_transform(self.watched['keyword_string'])
        
        # 3. Era/Decade - One-hot encoding
        print("   📅 Processing decades...")
        self.watched['decade'] = (self.watched['imdb_start_year'] // 10) * 10
        decades = pd.get_dummies(self.watched['decade'], prefix='decade')
        self.decade_matrix = decades.values
        self.decade_columns = decades.columns
        
        # 4. Directors - Binary encoding
        print("   🎬 Processing directors...")
        director_lists = []
        for directors in self.watched['imdb_directors'].fillna(''):
            if directors:
                director_lists.append([d.strip() for d in str(directors).split(',')])
            else:
                director_lists.append([])
        
        self.director_binarizer = MultiLabelBinarizer(sparse_output=False)
        self.director_matrix = self.director_binarizer.fit_transform(director_lists)
        
        # 5. Actors - Binary encoding (top 3 actors only)
        print("   ⭐ Processing actors...")
        actor_lists = []
        for cast_list in self.watched['imdb_cast']:
            if isinstance(cast_list, list) and len(cast_list) > 0:
                # Get top 3 actors
                actors = [p['name'] for p in cast_list[:3] 
                         if isinstance(p, dict) and 'name' in p]
                actor_lists.append(actors)
            else:
                actor_lists.append([])
        
        self.actor_binarizer = MultiLabelBinarizer(sparse_output=False)
        self.actor_matrix = self.actor_binarizer.fit_transform(actor_lists)
        
        # 6. Runtime - Normalized continuous feature
        print("   ⏱️ Processing runtime...")
        runtime = self.watched['imdb_runtime_minutes'].fillna(
            self.watched['imdb_runtime_minutes'].median()
        )
        self.runtime_scaler = StandardScaler()
        self.runtime_matrix = self.runtime_scaler.fit_transform(runtime.values.reshape(-1, 1))
        
        # 7. Language - Binary encoding
        print("   🌍 Processing languages...")
        languages = pd.get_dummies(self.watched['imdb_original_language'].fillna('en'))
        self.language_matrix = languages.values
        self.language_columns = languages.columns
        
        print("   ✅ All features prepared!")
    
    def create_content_dna(self) -> np.ndarray:
        """Create weighted Content-DNA matrix for all movies."""
        print("\n🧬 Creating Content-DNA vectors...")
        
        # Prepare all features
        self.prepare_features()
        
        # Normalize each feature matrix
        from sklearn.preprocessing import normalize
        
        # Convert sparse matrices to dense
        genre_dense = self.genre_matrix.toarray()
        keyword_dense = self.keyword_matrix.toarray()
        
        # Normalize
        genre_norm = normalize(genre_dense, axis=1)
        keyword_norm = normalize(keyword_dense, axis=1)
        decade_norm = normalize(self.decade_matrix, axis=1)
        director_norm = normalize(self.director_matrix, axis=1)
        actor_norm = normalize(self.actor_matrix, axis=1)
        runtime_norm = self.runtime_matrix
        language_norm = normalize(self.language_matrix, axis=1)
        
        # Apply weights and concatenate
        weighted_features = np.hstack([
            genre_norm * self.feature_weights['genres'],
            keyword_norm * self.feature_weights['keywords'],
            decade_norm * self.feature_weights['era'],
            director_norm * self.feature_weights['directors'],
            actor_norm * self.feature_weights['actors'],
            runtime_norm * self.feature_weights['runtime'],
            language_norm * self.feature_weights['language']
        ])
        
        print(f"   ✅ Created DNA matrix: {weighted_features.shape}")
        return weighted_features
    
    def get_taste_vector(self, dna_matrix: np.ndarray) -> np.ndarray:
        """Calculate your personal taste vector (average of all watched)."""
        taste_vector = np.mean(dna_matrix, axis=0)
        print(f"   🎯 Your taste vector shape: {taste_vector.shape}")
        return taste_vector


class CreatorCollaborationNetwork:
    """Build and analyze creator collaboration networks."""
    
    def __init__(self, watched_movies: pd.DataFrame):
        """Initialize with watched movies."""
        self.watched = watched_movies
        self.graph = nx.Graph()
        print("🕸️ Building Creator Collaboration Network...")
        
    def build_network(self):
        """Build the collaboration graph."""
        print("   📊 Analyzing collaborations...")
        
        collaboration_count = 0
        
        for _, movie in self.watched.iterrows():
            movie_id = movie['tconst']
            
            # Get directors
            directors = []
            if pd.notna(movie['imdb_directors']):
                directors = [d.strip() for d in str(movie['imdb_directors']).split(',')]
            
            # Get top actors
            actors = []
            if isinstance(movie['imdb_cast'], list):
                actors = [p['name'] for p in movie['imdb_cast'][:5] 
                         if isinstance(p, dict) and 'name' in p]
            
            # Add nodes
            for director in directors:
                self.graph.add_node(f"DIR:{director}", type='director', name=director)
            
            for actor in actors:
                self.graph.add_node(f"ACT:{actor}", type='actor', name=actor)
            
            # Add edges (collaborations)
            # Director-Actor collaborations
            for director in directors:
                for actor in actors:
                    dir_node = f"DIR:{director}"
                    act_node = f"ACT:{actor}"
                    
                    if self.graph.has_edge(dir_node, act_node):
                        self.graph[dir_node][act_node]['weight'] += 1
                        self.graph[dir_node][act_node]['movies'].append(movie_id)
                    else:
                        self.graph.add_edge(dir_node, act_node, weight=1, movies=[movie_id])
                        collaboration_count += 1
            
            # Actor-Actor collaborations (ensemble)
            for i, actor1 in enumerate(actors):
                for actor2 in actors[i+1:]:
                    act1_node = f"ACT:{actor1}"
                    act2_node = f"ACT:{actor2}"
                    
                    if self.graph.has_edge(act1_node, act2_node):
                        self.graph[act1_node][act2_node]['weight'] += 1
                        self.graph[act1_node][act2_node]['movies'].append(movie_id)
                    else:
                        self.graph.add_edge(act1_node, act2_node, weight=1, movies=[movie_id])
                        collaboration_count += 1
        
        print(f"   ✅ Network built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        print(f"   🤝 Total collaborations: {collaboration_count}")
    
    def find_frequent_collaborators(self, min_collaborations: int = 2) -> List[Tuple]:
        """Find creators who frequently work together."""
        frequent = []
        
        for edge in self.graph.edges(data=True):
            if edge[2]['weight'] >= min_collaborations:
                node1_name = self.graph.nodes[edge[0]]['name']
                node2_name = self.graph.nodes[edge[1]]['name']
                frequent.append((node1_name, node2_name, edge[2]['weight']))
        
        frequent.sort(key=lambda x: x[2], reverse=True)
        return frequent
    
    def get_creator_network(self, creator_name: str, creator_type: str = None) -> Set[str]:
        """Get all creators connected to a specific creator."""
        # Find the node
        node_id = None
        if creator_type:
            prefix = "DIR:" if creator_type == 'director' else "ACT:"
            node_id = f"{prefix}{creator_name}"
        else:
            # Try both
            for prefix in ["DIR:", "ACT:"]:
                potential_id = f"{prefix}{creator_name}"
                if potential_id in self.graph:
                    node_id = potential_id
                    break
        
        if node_id and node_id in self.graph:
            neighbors = set()
            for neighbor in self.graph.neighbors(node_id):
                neighbor_name = self.graph.nodes[neighbor]['name']
                neighbors.add(neighbor_name)
            return neighbors
        
        return set()


class HybridRecommendationEngine:
    """Main recommendation engine combining Content-DNA and Collaboration."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        print("\n" + "="*70)
        print("🚀 INITIALIZING HYBRID RECOMMENDATION ENGINE")
        print("="*70)
        
        # Load enriched watched movies
        enriched_path = config.paths.processed_data_dir / 'watched_fully_enriched.json'
        self.watched = pd.read_json(enriched_path)
        
        # Load taste profile
        profile_path = config.paths.processed_data_dir / 'taste_profile.json'
        with open(profile_path, 'r') as f:
            self.taste_profile = json.load(f)
        
        # Initialize components
        self.dna_engine = ContentDNAEngine(self.watched, self.taste_profile)
        self.collab_network = CreatorCollaborationNetwork(self.watched)
        
        # Build DNA and network
        self.content_dna = self.dna_engine.create_content_dna()
        self.taste_vector = self.dna_engine.get_taste_vector(self.content_dna)
        
        self.collab_network.build_network()
        
        # Connect to IMDb database for recommendations
        self.session = SessionLocal()
        
        print("\n✅ Engine initialized successfully!")
    
    def get_candidate_movies(self, limit: int = 10000) -> pd.DataFrame:
        """Get candidate movies from IMDb database."""
        print(f"\n📚 Loading candidate movies from IMDb...")
        
        # Get movies not in watched list
        watched_ids = set(self.watched['tconst'].tolist())
        
        # Query for candidate movies - NO MINIMUM RATING OR VOTES!
        query = """
        SELECT 
            tb.tconst,
            tb.primary_title,
            tb.start_year,
            tb.genres,
            tb.runtime_minutes,
            tr.average_rating,
            tr.num_votes
        FROM title_basics tb
        LEFT JOIN title_ratings tr ON tb.tconst = tr.tconst
        WHERE tb.title_type = 'movie'
            AND tb.is_adult = 0
            AND tb.start_year IS NOT NULL
        ORDER BY RANDOM()
        LIMIT :limit
        """
        
        with self.session.connection() as conn:
            result = pd.read_sql(text(query), conn, params={'limit': limit})
        
        # Filter out watched movies
        candidates = result[~result['tconst'].isin(watched_ids)]
        
        print(f"   ✅ Loaded {len(candidates):,} candidate movies")
        return candidates
    
    def calculate_content_similarity(self, candidates: pd.DataFrame) -> np.ndarray:
        """Calculate content similarity for candidates."""
        print("\n🧬 Calculating Content-DNA similarity...")
        
        # Create simplified DNA for candidates (using available features)
        candidate_features = []
        
        for _, movie in candidates.iterrows():
            # Genre vector
            genre_str = movie['genres'] if pd.notna(movie['genres']) else ''
            genre_vec = self.dna_engine.genre_vectorizer.transform([genre_str]).toarray()[0]
            
            # Decade vector
            decade = (movie['start_year'] // 10) * 10 if pd.notna(movie['start_year']) else 1990
            decade_vec = np.zeros(len(self.dna_engine.decade_columns))
            decade_col = f'decade_{decade}'
            if decade_col in self.dna_engine.decade_columns:
                decade_idx = list(self.dna_engine.decade_columns).index(decade_col)
                decade_vec[decade_idx] = 1.0
            
            # Runtime vector
            runtime = movie['runtime_minutes'] if pd.notna(movie['runtime_minutes']) else 100
            runtime_vec = self.dna_engine.runtime_scaler.transform([[runtime]])[0]
            
            # Combine with weights (simplified - no keywords/creators for candidates)
            feature = np.concatenate([
                genre_vec * self.dna_engine.feature_weights['genres'],
                np.zeros(200) * self.dna_engine.feature_weights['keywords'],  # No keywords
                decade_vec * self.dna_engine.feature_weights['era'],
                np.zeros(self.dna_engine.director_matrix.shape[1]) * self.dna_engine.feature_weights['directors'],
                np.zeros(self.dna_engine.actor_matrix.shape[1]) * self.dna_engine.feature_weights['actors'],
                runtime_vec * self.dna_engine.feature_weights['runtime'],
                np.zeros(self.dna_engine.language_matrix.shape[1]) * self.dna_engine.feature_weights['language']
            ])
            
            candidate_features.append(feature)
        
        candidate_matrix = np.array(candidate_features)
        
        # Calculate similarity to taste vector
        similarities = cosine_similarity([self.taste_vector], candidate_matrix)[0]
        
        print(f"   ✅ Calculated similarities for {len(similarities):,} movies")
        return similarities
    
    def recommend_deep_dive(self, n_recommendations: int = 50) -> pd.DataFrame:
        """DEEP DIVE MODE: More of exactly what you love."""
        print("\n🎯 DEEP DIVE MODE: Finding movies exactly like your favorites...")
        
        candidates = self.get_candidate_movies(limit=5000)
        similarities = self.calculate_content_similarity(candidates)
        
        # Sort by similarity
        candidates['similarity_score'] = similarities
        candidates['confidence'] = candidates['similarity_score'] * 100
        
        # Get top recommendations
        recommendations = candidates.nlargest(n_recommendations, 'similarity_score')
        
        # Add explanation
        recommendations['reason'] = recommendations.apply(
            lambda x: f"Matches your taste DNA by {x['similarity_score']*100:.1f}%", axis=1
        )
        
        print(f"   ✅ Found {len(recommendations)} deep dive recommendations")
        return recommendations
    
    def recommend_diverse(self, n_recommendations: int = 50) -> pd.DataFrame:
        """DIVERSE MODE: Explore intelligently across genres."""
        print("\n🌈 DIVERSE MODE: Expanding your horizons...")
        
        # Identify underexplored genres
        watched_genres = Counter()
        for genres in self.watched['imdb_genres'].dropna():
            for genre in str(genres).split(','):
                watched_genres[genre.strip()] += 1
        
        # Get average for normalization
        avg_genre_count = np.mean(list(watched_genres.values()))
        
        # Find underrepresented genres
        all_genres = ['Western', 'Musical', 'Documentary', 'War', 'Sport', 
                     'Biography', 'History', 'Music', 'Family', 'Animation']
        
        underexplored = []
        for genre in all_genres:
            count = watched_genres.get(genre, 0)
            if count < avg_genre_count * 0.5:  # Less than 50% of average
                underexplored.append(genre)
        
        print(f"   📊 Underexplored genres: {', '.join(underexplored)}")
        
        # Get movies from underexplored genres that still match taste
        candidates = self.get_candidate_movies(limit=5000)
        
        # Filter for underexplored genres
        diverse_candidates = []
        for _, movie in candidates.iterrows():
            if pd.notna(movie['genres']):
                for genre in underexplored:
                    if genre in movie['genres']:
                        diverse_candidates.append(movie)
                        break
        
        if not diverse_candidates:
            print("   ⚠️ No diverse candidates found, falling back to general exploration")
            diverse_candidates = candidates.sample(min(200, len(candidates))).to_dict('records')
        
        diverse_df = pd.DataFrame(diverse_candidates)
        
        # Calculate similarity to ensure quality
        similarities = self.calculate_content_similarity(diverse_df)
        diverse_df['similarity_score'] = similarities
        diverse_df['diversity_bonus'] = 0.2  # Bonus for being diverse
        diverse_df['final_score'] = diverse_df['similarity_score'] + diverse_df['diversity_bonus']
        
        # Get top diverse recommendations
        recommendations = diverse_df.nlargest(n_recommendations, 'final_score')
        
        # Add explanation
        recommendations['reason'] = recommendations.apply(
            lambda x: f"Expands your taste into {x['genres']} ({x['similarity_score']*100:.1f}% match)",
            axis=1
        )
        
        print(f"   ✅ Found {len(recommendations)} diverse recommendations")
        return recommendations
    
    def recommend_hidden_gems(self, n_recommendations: int = 50) -> pd.DataFrame:
        """HIDDEN GEMS MODE: Obscure films matching your taste."""
        print("\n💎 HIDDEN GEMS MODE: Finding obscure treasures...")
        
        # Query specifically for low-vote movies
        query = """
        SELECT 
            tb.tconst,
            tb.primary_title,
            tb.start_year,
            tb.genres,
            tb.runtime_minutes,
            tr.average_rating,
            tr.num_votes
        FROM title_basics tb
        LEFT JOIN title_ratings tr ON tb.tconst = tr.tconst
        WHERE tb.title_type = 'movie'
            AND tb.is_adult = 0
            AND tb.start_year IS NOT NULL
            AND (tr.num_votes < 5000 OR tr.num_votes IS NULL)
        ORDER BY RANDOM()
        LIMIT 5000
        """
        
        with self.session.connection() as conn:
            candidates = pd.read_sql(text(query), conn)
        
        # Filter out watched
        watched_ids = set(self.watched['tconst'].tolist())
        candidates = candidates[~candidates['tconst'].isin(watched_ids)]
        
        # Calculate similarity
        similarities = self.calculate_content_similarity(candidates)
        candidates['similarity_score'] = similarities
        
        # Bonus for being truly obscure
        candidates['obscurity_bonus'] = candidates['num_votes'].apply(
            lambda x: 0.3 if pd.isna(x) or x < 1000 else 0.1
        )
        candidates['final_score'] = candidates['similarity_score'] + candidates['obscurity_bonus']
        
        # Get top hidden gems
        recommendations = candidates.nlargest(n_recommendations, 'final_score')
        
        # Add explanation
        recommendations['reason'] = recommendations.apply(
            lambda x: f"Hidden gem with only {x['num_votes']:.0f} votes" 
                     if pd.notna(x['num_votes']) 
                     else "Completely undiscovered gem",
            axis=1
        )
        
        print(f"   ✅ Found {len(recommendations)} hidden gems")
        return recommendations
    
    def save_recommendations(self, recommendations: Dict[str, pd.DataFrame]):
        """Save recommendations to file."""
        output_path = config.paths.processed_data_dir / 'recommendations.json'
        
        # Convert to JSON-serializable format
        output = {}
        for mode, recs in recommendations.items():
            output[mode] = recs.to_dict('records')
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Saved recommendations to: {output_path}")
        
        # Also save summary
        summary_path = config.paths.processed_data_dir / 'recommendation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FILMOTECA - PERSONALIZED RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            
            for mode, recs in recommendations.items():
                f.write(f"\n{mode.upper()} RECOMMENDATIONS\n")
                f.write("-"*40 + "\n")
                for _, movie in recs.head(10).iterrows():
                    f.write(f"• {movie['primary_title']} ({movie['start_year']:.0f})\n")
                    f.write(f"  {movie['genres']}\n")
                    if pd.notna(movie.get('average_rating')):
                        f.write(f"  IMDb: {movie['average_rating']:.1f}/10")
                        if pd.notna(movie.get('num_votes')):
                            f.write(f" ({movie['num_votes']:.0f} votes)")
                    f.write(f"\n  Why: {movie['reason']}\n\n")
        
        print(f"📄 Saved summary to: {summary_path}")


def main():
    """Generate personalized recommendations."""
    print("\n" + "="*70)
    print("🎬 FILMOTECA - Hybrid Recommendation System")
    print("="*70)
    
    # First, ensure taste profile exists
    profile_path = config.paths.processed_data_dir / 'taste_profile.json'
    if not profile_path.exists():
        print("\n⚠️ Taste profile not found. Generating it first...")
        from scripts.analyze_taste_profile import TasteProfileAnalyzer
        analyzer = TasteProfileAnalyzer()
        profile = analyzer.generate_taste_profile()
        analyzer.save_profile(profile)
    
    # Initialize recommendation engine
    engine = HybridRecommendationEngine()
    
    # Generate recommendations in different modes
    recommendations = {}
    
    print("\n" + "="*70)
    print("🎯 GENERATING PERSONALIZED RECOMMENDATIONS")
    print("="*70)
    
    # 1. Deep Dive
    recommendations['deep_dive'] = engine.recommend_deep_dive(n_recommendations=50)
    
    # 2. Diverse
    recommendations['diverse'] = engine.recommend_diverse(n_recommendations=50)
    
    # 3. Hidden Gems
    recommendations['hidden_gems'] = engine.recommend_hidden_gems(n_recommendations=50)
    
    # Save all recommendations
    engine.save_recommendations(recommendations)
    
    print("\n" + "="*70)
    print("✅ RECOMMENDATIONS COMPLETE!")
    print("="*70)
    
    print("\n📊 Generated Recommendations:")
    for mode, recs in recommendations.items():
        print(f"   • {mode.replace('_', ' ').title()}: {len(recs)} movies")
    
    print("\n🎬 Sample Recommendations:")
    for mode, recs in recommendations.items():
        print(f"\n{mode.replace('_', ' ').upper()}:")
        for _, movie in recs.head(3).iterrows():
            print(f"   • {movie['primary_title']} ({movie['start_year']:.0f})")
    
    print("\n📂 Files saved:")
    print(f"   • data/processed/recommendations.json")
    print(f"   • data/processed/recommendation_summary.txt")
    print(f"   • data/processed/taste_profile.json")
    
    print("\n🚀 Next steps:")
    print("   1. Review recommendation_summary.txt")
    print("   2. Run: python scripts/generate_recommendation_dashboard.py")
    print("      → Create interactive HTML dashboard")


if __name__ == "__main__":
    main()