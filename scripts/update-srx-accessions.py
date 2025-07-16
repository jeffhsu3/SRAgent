#!/usr/bin/env python3
"""
Script to update srx_accession values in the SQL database based on GCP bucket directory names.
"""

import os
import sys
import re
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.exceptions import NotFound
import pandas as pd

# Import database connection modules (assuming they're in the same project structure)
try:
    from SRAgent.db.connect import db_connect
except ImportError:
    print("Error: Unable to import SRAgent.db.connect. Make sure SRAgent is installed.")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
        description="Update srx_accession values in SQL database based on GCP bucket directory names",
        epilog="""EXAMPLES:
        
Update ARC prefixes to NRX in prod database:
./srx_accession_updater.py \\
  --regex-pattern "^ARC" \\
  --replacement "NRX" \\
  --gcp-directory "gs://arc-ctc-screcounter/prodC/" \\
  --tenant prod

Dry run to see what would change:
./srx_accession_updater.py \\
  --regex-pattern "^ARC" \\
  --replacement "NRX" \\
  --gcp-directory "gs://arc-ctc-screcounter/prodC/" \\
  --tenant test \\
  --dry-run

Skip database update and only rename GCP directories:
./srx_accession_updater.py \\
  --regex-pattern "^ARC" \\
  --replacement "NRX" \\
  --gcp-directory "gs://arc-ctc-screcounter/prodC/" \\
  --tenant prod \\
  --skip-sql-database

Directory structure expected:
gs://arc-ctc-screcounter/prodC/SCRECOUNTER_2025-07-04_00-00-00/STAR/ARC0000001/
                                                                    ^^^^^^^^^
                                                                    srx_accession
        """,
        formatter_class=CustomFormatter
    )
    
    parser.add_argument(
        "--regex-pattern",
        required=True,
        help="Regex pattern to match in srx_accession values (e.g., '^ARC')"
    )
    parser.add_argument(
        "--replacement",
        required=True,
        help="Replacement string for matched pattern (e.g., 'NRX')"
    )
    parser.add_argument(
        "--gcp-directory",
        required=True,
        help="GCP directory path containing the directory structure (e.g., 'gs://arc-ctc-screcounter/prodC/')"
    )
    parser.add_argument(
        "--tenant",
        required=True,
        choices=["test", "prod"],
        help="SQL database tenant to use"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions that would be taken without executing them"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--skip-sql-database",
        action="store_true",
        help="Skip database update and only rename GCP directories"
    )
    return parser.parse_args()

def parse_gcp_path(gcp_path: str) -> Tuple[str, str]:
    """Parse a GCS path into bucket name and prefix."""
    if not gcp_path.startswith("gs://"):
        raise ValueError(f"Invalid GCP path: {gcp_path}. Must start with 'gs://'")
    
    parts = gcp_path.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    return bucket_name, prefix

def get_srx_directories(gcp_directory: str) -> List[str]:
    """
    Get all SRX accession directories from the GCP bucket.
    
    Args:
        gcp_directory: GCP directory path
    Returns:
        List of SRX accession strings found in directory names
    """
    try:
        bucket_name, prefix = parse_gcp_path(gcp_directory)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List all blobs with the prefix
        blobs = bucket.list_blobs(prefix=prefix)
        
        # Extract SRX accessions from directory paths
        srx_accessions = set()
        for blob in blobs:
            # Expected structure: .../SCRECOUNTER_*/STAR/SRX_ACCESSION/...
            path_parts = blob.name.split("/")
            
            # Look for STAR directory and get the next component
            for i, part in enumerate(path_parts):
                if part == "STAR" and i + 1 < len(path_parts):
                    potential_srx = path_parts[i + 1]
                    if potential_srx and potential_srx != "":
                        srx_accessions.add(potential_srx)
                    break
        
        return sorted(list(srx_accessions))
    
    except Exception as e:
        logger.error(f"Error listing GCP directory {gcp_directory}: {e}")
        return []

def find_matching_srx_accessions(srx_accessions: List[str], regex_pattern: str) -> List[str]:
    """
    Find SRX accessions that match the regex pattern.
    
    Args:
        srx_accessions: List of SRX accession strings
        regex_pattern: Regex pattern to match
        
    Returns:
        List of matching SRX accessions
    """
    try:
        pattern = re.compile(regex_pattern)
        matching = []
        
        for srx in srx_accessions:
            if pattern.search(srx):
                matching.append(srx)
        
        return matching
    
    except re.error as e:
        logger.error(f"Invalid regex pattern '{regex_pattern}': {e}")
        return []

def generate_new_srx_accessions(old_srx_accessions: List[str], regex_pattern: str, replacement: str) -> Dict[str, str]:
    """
    Generate new SRX accessions based on the regex pattern and replacement.
    
    Args:
        old_srx_accessions: List of old SRX accessions
        regex_pattern: Regex pattern to match
        replacement: Replacement string
        
    Returns:
        Dictionary mapping old SRX accessions to new ones
    """
    try:
        pattern = re.compile(regex_pattern)
        mapping = {}
        
        for old_srx in old_srx_accessions:
            new_srx = pattern.sub(replacement, old_srx)
            if new_srx != old_srx:
                mapping[old_srx] = new_srx
        
        return mapping
    
    except re.error as e:
        logger.error(f"Invalid regex pattern '{regex_pattern}': {e}")
        return {}

def get_database_srx_accessions(conn, old_srx_accessions: List[str]) -> Dict[str, List[str]]:
    """
    Get SRX accessions from database tables that match the old accessions.
    
    Args:
        conn: Database connection
        old_srx_accessions: List of old SRX accessions to check
        
    Returns:
        Dictionary mapping table names to lists of found SRX accessions
    """
    tables_and_columns = {
        "srx_metadata": "srx_accession",
        "srx_srr": "srx_accession",
        "screcounter_log": "sample",
        "screcounter_star_params": "sample",
        "screcounter_star_results": "sample"
    }
    
    found_accessions = {}
    
    for table_name, column_name in tables_and_columns.items():
        try:
            # Create a parameterized query to avoid SQL injection
            placeholders = ",".join(["%s"] * len(old_srx_accessions))
            query = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IN ({placeholders})"
            
            df = pd.read_sql(query, conn, params=old_srx_accessions)
            found_accessions[table_name] = df[column_name].tolist()
            
        except Exception as e:
            logger.warning(f"Error querying table {table_name}: {e}")
            found_accessions[table_name] = []
    
    return found_accessions

def update_database_srx_accessions(conn, srx_mapping: Dict[str, str], dry_run: bool = False) -> None:
    """
    Update SRX accessions in the database.
    
    Args:
        conn: Database connection
        srx_mapping: Dictionary mapping old SRX accessions to new ones
        dry_run: If True, only log what would be done without executing
    """
    tables_and_columns = {
        "srx_metadata": "srx_accession",
        "srx_srr": "srx_accession",
        "screcounter_log": "sample",
        "screcounter_star_params": "sample",
        "screcounter_star_results": "sample"
    }
    
    for table_name, column_name in tables_and_columns.items():
        for old_srx, new_srx in srx_mapping.items():
            try:
                update_query = f"UPDATE {table_name} SET {column_name} = %s WHERE {column_name} = %s"
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would execute: {update_query} with params ({new_srx}, {old_srx})")
                else:
                    with conn.cursor() as cur:
                        cur.execute(update_query, (new_srx, old_srx))
                        rows_affected = cur.rowcount
                        if rows_affected > 0:
                            logger.info(f"Updated {rows_affected} rows in {table_name}: {old_srx} -> {new_srx}")
                        else:
                            logger.debug(f"No rows updated in {table_name} for {old_srx}")
                
            except Exception as e:
                logger.error(f"Error updating table {table_name}: {e}")
                if not dry_run:
                    conn.rollback()
                    raise

def rename_gcp_directories(gcp_directory: str, srx_mapping: Dict[str, str], dry_run: bool = False) -> None:
    """
    Rename GCP directories based on SRX accession mapping.
    
    Args:
        gcp_directory: Base GCP directory path
        srx_mapping: Dictionary mapping old SRX accessions to new ones
        dry_run: If True, only log what would be done without executing
    """
    try:
        bucket_name, prefix = parse_gcp_path(gcp_directory)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List all blobs to find directories to rename
        blobs = bucket.list_blobs(prefix=prefix)
        
        # Group blobs by their SRX directory
        srx_blob_groups = {}
        for blob in blobs:
            path_parts = blob.name.split("/")
            
            # Look for STAR directory and get the next component
            for i, part in enumerate(path_parts):
                if part == "STAR" and i + 1 < len(path_parts):
                    potential_srx = path_parts[i + 1]
                    if potential_srx in srx_mapping:
                        if potential_srx not in srx_blob_groups:
                            srx_blob_groups[potential_srx] = []
                        srx_blob_groups[potential_srx].append(blob)
                    break
        
        # Rename directories by copying and deleting blobs
        for old_srx, blobs_to_move in srx_blob_groups.items():
            new_srx = srx_mapping[old_srx]
            
            for blob in blobs_to_move:
                old_name = blob.name
                new_name = old_name.replace(f"/STAR/{old_srx}/", f"/STAR/{new_srx}/")
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would rename: {old_name} -> {new_name}")
                else:
                    # Copy blob to new location
                    bucket.copy_blob(blob, bucket, new_name)
                    # Delete old blob
                    blob.delete()
                    logger.info(f"Renamed: {old_name} -> {new_name}")
    
    except Exception as e:
        logger.error(f"Error renaming GCP directories: {e}")
        raise

def main():
    """Main function."""
    load_dotenv()
    args = parse_cli_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set database tenant
    os.environ["DYNACONF"] = args.tenant
    
    logger.info(f"Starting SRX accession updater (tenant: {args.tenant})")
    logger.info(f"Regex pattern: {args.regex_pattern}")
    logger.info(f"Replacement: {args.replacement}")
    logger.info(f"GCP directory: {args.gcp_directory}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Skip SQL database: {args.skip_sql_database}")
    
    try:
        # Step 1: Get SRX accessions from GCP directories
        logger.info("# Step 1: Getting accessions from GCP directories...")
        all_srx_accessions = get_srx_directories(args.gcp_directory)
        logger.info(f"Found {len(all_srx_accessions)} accessions in GCP directories")
        
        if not all_srx_accessions:
            logger.error("No accessions found in GCP directories")
            return
        
        # Step 2: Find matching SRX accessions
        logger.info("# Step 2: Finding accessions that match the regex pattern...")
        matching_srx_accessions = find_matching_srx_accessions(all_srx_accessions, args.regex_pattern)
        logger.info(f"Found {len(matching_srx_accessions)} matching accessions")
        
        if not matching_srx_accessions:
            logger.info("No accessions match the regex pattern")
            return
        # print up to 3 matching accessions
        logger.info(f"First 3 matching accessions: {', '.join(matching_srx_accessions[:3])}")

        # Step 3: Generate new SRX accessions
        logger.info("# Step 3: Generating new accessions...")
        srx_mapping = generate_new_srx_accessions(matching_srx_accessions, args.regex_pattern, args.replacement)
        
        if not srx_mapping:
            logger.info("No accessions would be changed")
            return

        # print up to 3 srx_mapping
        logger.info(f"Example accession mapping of the {len(srx_mapping)} total changes:")
        for i, (old_srx, new_srx) in enumerate(srx_mapping.items()):
            logger.info(f"  {i+1}. {old_srx} -> {new_srx}")
            if i >= 2:
                break
        
        if not args.skip_sql_database:
            # Step 4: Check database for existing SRX accessions
            logger.info("# Step 4: Checking database for existing accessions...")
            with db_connect() as conn:
                db_accessions = get_database_srx_accessions(conn, list(srx_mapping.keys()))

                # get unique accessions from db_accessions
                unique_db_accessions = set(accession for accessions in db_accessions.values() for accession in accessions)
                logging.info(f"Found {len(unique_db_accessions)} unique accessions in the database")
                
                total_db_records = sum(len(accessions) for accessions in db_accessions.values())
                logger.info(f"Found {total_db_records} total database records to update")
                
                for table_name, accessions in db_accessions.items():
                    if accessions:
                        logger.info(f"  {table_name}: {len(accessions)} records")

                # if missing any accessions, raise an error
                if len(srx_mapping) != len(unique_db_accessions):
                    msg = f"Missing {len(unique_db_accessions) - len(srx_mapping)} accessions in the database"
                    logger.error(msg)
                    raise Exception(msg)
            
            # Step 5: Update database
            logger.info("# Step 5: Updating database accessions...")
            with db_connect() as conn:
                update_database_srx_accessions(conn, srx_mapping, args.dry_run)
                
                if not args.dry_run:
                    conn.commit()
                    logger.info("Database updates committed")
        else:
            logger.info("# Steps 4-5: Skipping database operations (--skip-sql-database flag set)")
        
        # Step 6: Rename GCP directories (or Step 4 if database is skipped)
        step_num = 4 if args.skip_sql_database else 6
        logger.info(f"# Step {step_num}: Renaming GCP directories...")
        rename_gcp_directories(args.gcp_directory, srx_mapping, args.dry_run)
        
        logger.info("SRX accession update completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during SRX accession update: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()