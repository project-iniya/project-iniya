"""
NOT FINALIZED - Work in Progress
Winget-based application downloader and searcher.
"""


import subprocess
import re
from typing import List, Dict, Tuple, Optional
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed

# Packages to ignore - converted to set for O(1) lookup
SUSPICIOUS_TAGS = {
    "beta", "dev", "nightly", "canary", "preview",
    "portable", "trial", "crack", "cracked", "mod",
    "launcher", "wrapper", "unofficial", "community",
    "fork", "alpha"
}

# Scoring weights - centralized configuration
class ScoringWeights:
    NAME_EXACT_MATCH = 3.0      # High weight for exact name match
    NAME_PARTIAL = 1.0          # Lower for partial matches
    ID_EXACT_BONUS = 50         # Big bonus for query in ID
    ID_PARTIAL = 0.8
    PUBLISHER_MATCH = 1.2       # Publisher trust
    ORG_MATCH = 2.0             # Organization in ID (e.g., "Google" in Google.Chrome)
    ORG_PUBLISHER_BONUS = 30    # Official source bonus
    SUSPICIOUS_PENALTY = 50     # Heavy penalty for unofficial builds
    ID_LENGTH_PENALTY = 0.2


def run_winget_command(args: List[str], timeout: int = 30) -> Optional[str]:
    """Run winget command with proper error handling and encoding detection."""
    try:
        result = subprocess.run(
            ["winget"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        
        # Try UTF-8 first, fallback to UTF-16
        for encoding in ["utf-8", "utf-16"]:
            try:
                return result.stdout.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # Last resort
        return result.stdout.decode("utf-8", errors="replace")
    
    except subprocess.TimeoutExpired:
        print(f"Warning: winget command timed out")
        return None
    except FileNotFoundError:
        print("Error: winget is not installed or not in PATH")
        return None
    except Exception as e:
        print(f"Error running winget: {e}")
        return None


def parse_winget_table(text: str) -> List[Dict[str, str]]:
    """Parse winget's tabular output into structured data."""
    if not text:
        return []
    
    lines = text.splitlines()
    
    # Find header line - look for line containing "Name" and "Id"
    header_idx = None
    for i, line in enumerate(lines):
        # Must contain both Name and Id
        if re.search(r'\bName\b', line) and re.search(r'\bId\b', line):
            header_idx = i
            break
    
    if header_idx is None:
        return []
    
    # Check if we have enough lines (header + separator + at least 1 data row)
    if header_idx + 2 >= len(lines):
        return []
    
    header = lines[header_idx]
    separator = lines[header_idx + 1]
    
    # Verify separator line (should contain mostly dashes)
    if separator.count('-') < 10:
        return []
    
    # Parse column positions from HEADER (not separator)
    # Find where each column name starts
    column_names = ['Name', 'Id', 'Version', 'Match', 'Source']
    columns = []
    
    for col_name in column_names:
        # Find column name in header (case insensitive)
        match = re.search(r'\b' + re.escape(col_name) + r'\b', header, re.IGNORECASE)
        if match:
            columns.append({
                'name': col_name,
                'start': match.start()
            })
    
    if len(columns) < 2:  # Need at least Name and Id
        return []
    
    # Sort columns by position
    columns.sort(key=lambda x: x['start'])
    
    # Find Name, Id, and Version column indices
    name_idx = next((i for i, c in enumerate(columns) if c['name'] == 'Name'), None)
    id_idx = next((i for i, c in enumerate(columns) if c['name'] == 'Id'), None)
    version_idx = next((i for i, c in enumerate(columns) if c['name'] == 'Version'), None)
    
    if name_idx is None or id_idx is None:
        return []
    
    # Parse data rows
    apps = []
    for line in lines[header_idx + 2:]:
        # Skip empty lines or lines that are separators
        stripped = line.strip()
        if not stripped or stripped.startswith('-'):
            continue
        
        try:
            # Extract Name (from name column start to id column start)
            name = line[columns[name_idx]['start']:columns[id_idx]['start']].strip()
            
            # Extract Id
            if version_idx is not None and version_idx > id_idx:
                app_id = line[columns[id_idx]['start']:columns[version_idx]['start']].strip()
            elif id_idx + 1 < len(columns):
                app_id = line[columns[id_idx]['start']:columns[id_idx + 1]['start']].strip()
            else:
                app_id = line[columns[id_idx]['start']:].strip()
            
            # Extract Version if available
            version = ""
            if version_idx is not None:
                if version_idx + 1 < len(columns):
                    version = line[columns[version_idx]['start']:columns[version_idx + 1]['start']].strip()
                else:
                    version = line[columns[version_idx]['start']:].strip()
            
            # Only add if we have both name and id
            if name and app_id:
                apps.append({
                    "Name": name,
                    "Id": app_id,
                    "Version": version,
                })
        
        except (IndexError, ValueError):
            # Skip malformed lines
            continue
    
    return apps


def extract_publisher(text: str) -> str:
    """Extract publisher from winget show output."""
    if not text:
        return ""
    
    for line in text.splitlines():
        line_lower = line.lower()
        if line_lower.startswith("publisher") or "publisher:" in line_lower:
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
    return ""


def fetch_publisher(app_id: str) -> str:
    """Fetch publisher for a single app."""
    text = run_winget_command(["show", app_id, "--source", "winget"])
    return extract_publisher(text) if text else ""


def winget_search(query: str, max_results: int = 15) -> List[Dict[str, str]]:
    """Search winget and enrich results with publisher info (parallel)."""
    raw = run_winget_command(
        ["search", query, "-s", "winget", "--disable-interactivity"]
    )
    
    if not raw:
        return []
    
    apps = parse_winget_table(raw)
    
    if not apps:
        return []
    
    # Only fetch publisher for top results to avoid slowdown
    apps_to_enrich = apps[:max_results]
    
    # Fetch publishers in parallel for speed
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_app = {
            executor.submit(fetch_publisher, app["Id"]): app 
            for app in apps_to_enrich
        }
        
        for future in as_completed(future_to_app):
            app = future_to_app[future]
            try:
                app["Publisher"] = future.result()
            except Exception:
                app["Publisher"] = ""
    
    # Add empty publisher for remaining apps
    for app in apps[max_results:]:
        app["Publisher"] = ""
    
    return apps


def get_org(app_id: str) -> str:
    """Extract organization from app ID (e.g., 'Google' from 'Google.Chrome')."""
    return app_id.split(".")[0].lower() if "." in app_id else ""


def has_suspicious_tags(text: str) -> bool:
    """Check if text contains suspicious tags."""
    text_lower = text.lower()
    return any(tag in text_lower for tag in SUSPICIOUS_TAGS)


def score_app(app: Dict[str, str], query: str) -> float:
    """Score an app based on relevance to query."""
    name = app["Name"].lower()
    app_id = app["Id"].lower()
    publisher = app.get("Publisher", "").lower()
    q = query.lower()
    
    score = 0.0
    
    # Get organization from ID (e.g., "Google" from "Google.Chrome")
    org = get_org(app_id)
    
    # 1) HIGHEST PRIORITY: Org.Query pattern (e.g., Mozilla.Firefox, Valve.Steam)
    # This is the canonical winget naming convention
    if org and (app_id == f"{org}.{q}" or app_id.startswith(f"{org}.{q}.")):
        score += 300  # MASSIVE boost for canonical naming
        
        # Prefer base package over variants (e.g., Firefox over Firefox.af)
        if app_id == f"{org}.{q}":
            score += 100  # Extra boost for exact Org.Query (no suffix)
    
    # 2) Organization matches query exactly (e.g., searching "discord" finds "Discord.Discord")
    if org == q:
        score += 150
        
        # But penalize if it's not the "official" one
        # VSCodium.VSCodium should lose to Microsoft.VisualStudioCode for "vscode"
        # Check if there's a more established org for this query
        established_orgs = {
            'vscode': 'microsoft',
            'code': 'microsoft',
            'obs': 'obsproject',
        }
        expected_org = established_orgs.get(q)
        if expected_org and org != expected_org:
            score -= 200  # Heavy penalty for wrong org
    
    # 3) Exact name match
    if q == name:
        score += 120
    
    # 4) Name similarity
    name_ratio = fuzz.ratio(q, name)
    score += name_ratio * ScoringWeights.NAME_EXACT_MATCH
    
    # Bonus if query is a complete word in the name
    if re.search(r'\b' + re.escape(q) + r'\b', name, re.IGNORECASE):
        score += 40
    
    # 5) App ID similarity
    id_ratio = fuzz.partial_ratio(q, app_id)
    score += id_ratio * ScoringWeights.ID_PARTIAL
    
    # Exact substring match in ID (but not as good as org match)
    if q in app_id:
        score += ScoringWeights.ID_EXACT_BONUS
    
    # 6) Organization similarity (for partial matches)
    if org:
        org_ratio = fuzz.ratio(q, org)
        score += org_ratio * ScoringWeights.ORG_MATCH
        
        # Org starts with query
        if org.startswith(q):
            score += 50
        
        # Official apps have matching org and publisher
        if org in publisher:
            score += ScoringWeights.ORG_PUBLISHER_BONUS
    
    # 7) Publisher match (strong signal of official app)
    if publisher:
        pub_ratio = fuzz.partial_ratio(q, publisher)
        score += pub_ratio * ScoringWeights.PUBLISHER_MATCH
        
        # Boost well-known publishers
        if any(trusted in publisher for trusted in ['google', 'microsoft', 'mozilla', 'discord', 'jetbrains', 'github', 'nvidia',]):
            score += 20
    
    # 8) Penalize language-specific variants (e.g., Firefox.af, Firefox.ar)
    # These have 2-letter language codes after the main package
    if re.search(r'\.[a-z]{2}(-[a-z]{2})?$', app_id):
        score -= 150  # Heavy penalty for language variants
    
    # 9) Penalize variant builds (PreRelease, LTS, Nightly, etc.) - prefer stable base
    variant_keywords = ['prerelease', 'preview', 'lts', 'nightly', 'insider', 'canary', 'dev', 'beta', 'alpha', 'rc']
    id_parts = app_id.lower().split('.')
    if any(keyword in part for part in id_parts for keyword in variant_keywords):
        score -= 100  # Penalize variants to prefer base stable package
  
    
    # 10) Penalize suspicious builds heavily
    if has_suspicious_tags(name) or has_suspicious_tags(app_id):
        score -= ScoringWeights.SUSPICIOUS_PENALTY
    
    # 11) Prefer shorter, cleaner IDs (official apps tend to be cleaner)
    score -= len(app_id) * ScoringWeights.ID_LENGTH_PENALTY
    
    # 12) Penalize if name contains query but ID doesn't match pattern
    # This catches "MTPuTTY" vs "PuTTY.PuTTY"
    if q in name and org != q and not app_id.startswith(f"{org}.{q}"):
        score -= 30
    
    # 13) Penalize if name is way longer than query (less relevant)
    if len(name) > len(q) * 3:
        score -= 15
    
    return score


def find_best_app(query: str) -> Tuple[Optional[Tuple[float, Dict]], List[Tuple[float, Dict]]]:
    """Find the best matching app and return top 5 ranked results."""
    apps = winget_search(query)
    
    if not apps:
        return None, []
    
    # Score and rank all apps
    ranked = [(score_app(app, query), app) for app in apps]
    ranked.sort(reverse=True, key=lambda x: x[0])
    
    return ranked[0], ranked[:5]




def main():
    """Interactive search interface."""
    print("Winget App Searcher")
    print("-" * 40)
    
    query = input("Enter app name to search: ").strip()
    
    if not query:
        print("Error: Please enter a search query")
        return
    
    print(f"\nSearching for '{query}'...\n")
    
    best, top5 = find_best_app(query)
    
    if not best:
        print("No results found.")
        return
    
    print("🏆 Best match:")
    print(f"  Score: {best[0]:.2f}")
    print(f"  Name: {best[1]['Name']}")
    print(f"  ID: {best[1]['Id']}")
    print(f"  Publisher: {best[1].get('Publisher', 'Unknown')}")
    print(f"  Version: {best[1].get('Version', 'Unknown')}")
    
    print("\n📋 Top 5 results:")
    for i, (score, app) in enumerate(top5, 1):
        pub = f" - {app.get('Publisher', 'Unknown')}" if app.get('Publisher') else ""
        print(f"{i}. [{score:.1f}] {app['Name']} ({app['Id']}){pub}")

if __name__ == "__main__":
    main()