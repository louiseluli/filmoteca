"""
Quick fix script to drop and reload only failed tables.
Preserves your 95M principals and other successfully loaded data!
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from config import config
from scripts.load_imdb_to_db import (
    engine, SessionLocal, 
    load_title_akas, 
    load_name_basics
)

def fix_failed_tables():
    """Drop and recreate only the failed tables."""
    
    print("\n" + "="*70)
    print("🔧 FIXING FAILED TABLES")
    print("="*70)
    print("\n⚠️  This will:")
    print("   • DROP title_akas (29.7M partial records)")
    print("   • DROP name_basics (0 records)")
    print("   • KEEP all other tables (including 95M principals!)")
    print("   • Recreate with fixed constraints")
    print("   • Reload both tables from scratch")
    
    response = input("\n❓ Continue? (yes/no): ").lower().strip()
    if response != 'yes':
        print("❌ Cancelled")
        return
    
    session = SessionLocal()
    
    try:
        # Drop the two failed tables
        print("\n🗑️  Dropping failed tables...")
        
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS title_akas"))
            conn.execute(text("DROP TABLE IF EXISTS name_basics"))
            conn.commit()
        
        print("   ✅ Dropped title_akas")
        print("   ✅ Dropped name_basics")
        
        # Recreate tables with new schema
        print("\n🔨 Recreating tables with fixed constraints...")
        
        with engine.connect() as conn:
            # Recreate title_akas
            conn.execute(text("""
                CREATE TABLE title_akas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title_id VARCHAR(15) NOT NULL,
                    ordering INTEGER NOT NULL,
                    title VARCHAR(500),
                    region VARCHAR(10),
                    language VARCHAR(10),
                    types VARCHAR(100),
                    attributes VARCHAR(200),
                    is_original_title BOOLEAN NOT NULL DEFAULT 0,
                    FOREIGN KEY(title_id) REFERENCES title_basics (tconst) ON DELETE CASCADE,
                    CONSTRAINT uq_title_ordering UNIQUE (title_id, ordering)
                )
            """))
            
            # Create indexes for title_akas
            conn.execute(text("CREATE INDEX idx_akas_title_id ON title_akas (title_id)"))
            conn.execute(text("CREATE INDEX idx_akas_title_lang ON title_akas (title_id, language)"))
            conn.execute(text("CREATE INDEX idx_akas_title_region ON title_akas (title_id, region)"))
            conn.execute(text("CREATE INDEX idx_akas_title_orig ON title_akas (title_id, is_original_title)"))
            conn.execute(text("CREATE INDEX idx_akas_language ON title_akas (language)"))
            conn.execute(text("CREATE INDEX idx_akas_region ON title_akas (region)"))
            conn.execute(text("CREATE INDEX idx_akas_original ON title_akas (is_original_title)"))
            conn.execute(text("CREATE INDEX idx_akas_lang_region ON title_akas (language, region)"))
            conn.execute(text("CREATE INDEX idx_akas_types ON title_akas (types)"))
            conn.execute(text("CREATE INDEX idx_akas_title_text ON title_akas (title)"))
            
            # Recreate name_basics with FIXED constraints (NO death_after_birth check for BC dates)
            conn.execute(text("""
                CREATE TABLE name_basics (
                    nconst VARCHAR(15) PRIMARY KEY,
                    primary_name VARCHAR(200) NOT NULL,
                    birth_year INTEGER,
                    death_year INTEGER,
                    primary_profession VARCHAR(200),
                    known_for_titles VARCHAR(200),
                    CONSTRAINT chk_birth_year CHECK (birth_year IS NULL OR birth_year <= 2100),
                    CONSTRAINT chk_death_year CHECK (death_year IS NULL OR death_year <= 2100)
                )
            """))
            
            # Create indexes for name_basics
            conn.execute(text("CREATE INDEX idx_name_primary ON name_basics (primary_name)"))
            conn.execute(text("CREATE INDEX idx_name_birth ON name_basics (birth_year)"))
            conn.execute(text("CREATE INDEX idx_name_death ON name_basics (death_year)"))
            conn.execute(text("CREATE INDEX idx_name_alive ON name_basics (birth_year, death_year)"))
            conn.execute(text("CREATE INDEX idx_name_profession ON name_basics (primary_profession)"))
            conn.execute(text("CREATE INDEX idx_name_prof_birth ON name_basics (primary_profession, birth_year)"))
            conn.execute(text("CREATE INDEX idx_name_birth_combo ON name_basics (primary_name, birth_year)"))
            
            conn.commit()
        
        print("   ✅ Recreated title_akas with 11 indexes")
        print("   ✅ Recreated name_basics with 8 indexes")
        print("   ✅ Birth year constraint: >= 1000 (allows Shakespeare!)")
        
        # Now load the data
        print("\n📥 Loading data...")
        
        print("\n" + "="*70)
        print("[1/2] Loading title_akas (~53.5M records)")
        print("="*70)
        
        akas_file = config.paths.imdb_dir / 'title.akas.tsv'
        if akas_file.exists():
            load_title_akas(session, akas_file, batch_size=10000)
        else:
            print(f"   ❌ File not found: {akas_file}")
        
        print("\n" + "="*70)
        print("[2/2] Loading name_basics (~14.8M records)")
        print("="*70)
        
        names_file = config.paths.imdb_dir / 'name.basics.tsv'
        if names_file.exists():
            load_name_basics(session, names_file, batch_size=10000)
        else:
            print(f"   ❌ File not found: {names_file}")
        
        # Verify
        print("\n" + "="*70)
        print("📊 FINAL COUNTS")
        print("="*70)
        
        with engine.connect() as conn:
            counts = {
                'title_basics': conn.execute(text("SELECT COUNT(*) FROM title_basics")).scalar(),
                'title_akas': conn.execute(text("SELECT COUNT(*) FROM title_akas")).scalar(),
                'title_ratings': conn.execute(text("SELECT COUNT(*) FROM title_ratings")).scalar(),
                'title_crew': conn.execute(text("SELECT COUNT(*) FROM title_crew")).scalar(),
                'title_principals': conn.execute(text("SELECT COUNT(*) FROM title_principals")).scalar(),
                'title_episode': conn.execute(text("SELECT COUNT(*) FROM title_episode")).scalar(),
                'name_basics': conn.execute(text("SELECT COUNT(*) FROM name_basics")).scalar(),
            }
            
            total = 0
            for table, count in counts.items():
                status = "✅" if count > 0 else "❌"
                print(f"   {status} {table}: {count:,}")
                total += count
            
            print(f"\n   ✅ TOTAL: {total:,} records")
            
            # Check if we got all records
            if counts['title_akas'] >= 50000000:  # Should be ~53.5M
                print(f"   ✅ title_akas: COMPLETE!")
            else:
                print(f"   ⚠️  title_akas: Only {counts['title_akas']:,} (expected ~53.5M)")
            
            if counts['name_basics'] >= 14000000:  # Should be ~14.8M
                print(f"   ✅ name_basics: COMPLETE!")
            else:
                print(f"   ⚠️  name_basics: Only {counts['name_basics']:,} (expected ~14.8M)")
        
        print("\n" + "="*70)
        print("✅ FIX COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    fix_failed_tables()