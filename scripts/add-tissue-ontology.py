#!/usr/bin/env python
"""
Script to update missing tissue_ontology_term_id values in the SRX metadata database.
"""

import os
import sys
import asyncio
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_core.messages import HumanMessage
from SRAgent.db.connect import db_connect
from SRAgent.db.update import db_update
from SRAgent.workflows.tissue_ontology import create_tissue_ontology_workflow
from SRAgent.tools.utils import set_entrez_access


def get_records_missing_tissue_ontology(conn, limit: int = None) -> pd.DataFrame:
    """
    Get SRX metadata records that have tissue information but lack tissue_ontology_term_id.
    
    Args:
        conn: Database connection
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with records missing tissue_ontology_term_id
    """
    query = """
    SELECT DISTINCT database, entrez_id, srx_accession, organism, tissue, disease, perturbation, cell_line
    FROM srx_metadata
    INNER JOIN screcounter_star_results ON srx_metadata.srx_accession = screcounter_star_results.sample
    WHERE (tissue_ontology_term_id IS NULL OR tissue_ontology_term_id = '')
    AND tissue IS NOT NULL 
    AND tissue != 'NaN'
    AND is_illumina = 'yes'
    AND is_single_cell = 'yes'
    AND is_paired_end = 'yes'
    AND lib_prep = '10x_Genomics'
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    return pd.read_sql(query, conn)


async def process_record(record: Dict[str, Any], workflow) -> Optional[List[str]]:
    """
    Process a single record to get tissue ontology terms.
    
    Args:
        record: Dictionary containing the record data
        workflow: The tissue ontology workflow
    Returns:
        List of Uberon IDs
    """
    # Format the message with available information
    tissue = record.get("tissue", "")
    if not tissue:
        return None
    organism = record.get("organism", "No organism provided")
    disease = record.get("disease", "No disease provided") 
    perturbation = record.get("perturbation", "No perturbation provided")
    cell_line = record.get("cell_line", "No cell line provided")
    
    # Handle empty or null values
    if disease in [None, "", "None", "none", "NA", "n/a"]:
        disease = "No disease provided"
    if perturbation in [None, "", "None", "none", "NA", "n/a"]:
        perturbation = "No perturbation provided"
    if cell_line in [None, "", "None", "none", "NA", "n/a"]:
        cell_line = "No cell line provided"

    message = "\n".join([
        f"Tissue description: {tissue}\n",
        "# Secondary information about the tissue",
        f" - The organism: {organism}",
        f" - The disease: {disease}",
        f" - The perturbation: {perturbation}",
        f" - The cell line: {cell_line}",
    ])
    
    # Call the workflow
    try:
        response = await workflow.ainvoke({"messages": [HumanMessage(content=message)]}, config={})
        return response  # This returns a list of Uberon IDs
    except Exception as e:
        print(f"Error processing record {record['srx_accession']}: {str(e)}", file=sys.stderr)
        return []


async def update_tissue_ontologies(
    target_records: pd.DataFrame,
    delay: float = 0.5,
    no_delay: bool = False,
) -> None:
    """
    Main function to update tissue ontologies.
    
    Args:
        target_records: DataFrame of target records to update
        delay: Delay in seconds between API calls
        no_delay: Skip delays between API calls
    """
    # set entrez email and api key
    set_entrez_access()

    # Create the tissue ontology workflow
    workflow = create_tissue_ontology_workflow()
    
    # Connect to database
    # Process records in batches to avoid keeping too many updates in memory
    total_records = len(target_records)
    processed = 0
    updated = 0
    failed = 0
    no_ontology = 0
    for _,record in target_records.iterrows():
        # progress indicator
        processed += 1
        if processed % 10 == 0:
            print(f"  - Processed {processed}/{total_records} records", file=sys.stderr)
            
        # Get tissue ontology terms
        ontology_ids = await process_record(record.to_dict(), workflow)

        if not ontology_ids:
            ontology_str = ""
            no_ontology += 1
        else:
            ontology_str = ",".join(ontology_ids)
            
        # Prepare update data
        update_df = pd.DataFrame([{
            "database": record["database"],
            "entrez_id": int(record["entrez_id"]),
            "tissue_ontology_term_id": ontology_str
        }])
            
        try:
            # Update the database
            with db_connect() as conn:
                db_update(update_df, "srx_metadata", conn)
            updated += 1
        except Exception as e:
            print(f"Failed to update record {record['srx_accession']}: {str(e)}", file=sys.stderr)
            failed += 1
        
        # Add a small delay to avoid overwhelming the API
        if not no_delay and processed < total_records:
            await asyncio.sleep(delay)
        
    # Print summary
    print("\n" + "="*50)
    print(f"Summary:")
    print(f" - Successfully updated: {updated}")
    print(f" - Failed updates: {failed}")
    print(f" - No ontology found: {no_ontology}")
    print(f" - Total records processed: {processed}")


def parse_args():
    """Parse command line arguments."""
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description="Update missing tissue_ontology_term_id values in SRX metadata database",
        formatter_class=CustomFormatter,
        epilog="""
Examples:
  # Update missing tissue ontologies in test database (first 10 records)
  python update_tissue_ontologies.py --tenant test --limit 10
  
  # Update all missing tissue ontologies in production database
  python update_tissue_ontologies.py --tenant prod
  
  # Update with no delay between API calls (use cautiously)
  python update_tissue_ontologies.py --tenant test --no-delay
        """
    )
    parser.add_argument(
        "--tenant",
        type=str,
        choices=["test", "prod"],
        help="Database tenant to use (optional)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of records to process"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between API calls"
    )
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="Skip delays between API calls (overrides --delay)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # Parse arguments
    args = parse_args()
    
    # Set the database tenant, 
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant
    if not os.getenv("DYNACONF"):
        print("Warning: No tenant specified. Using the 'test' tenant", file=sys.stderr)
        os.environ["DYNACONF"] = "test"
    
    # Connect to database and get records missing tissue ontology
    target_records = None
    with db_connect() as conn:
        target_records = get_records_missing_tissue_ontology(conn, limit=args.limit)

    # If dry run, just show what would be updated
    if args.dry_run:
        print(f"DRY RUN: Checking {args.tenant} database for records missing tissue_ontology_term_id...")
        print(f"\nWould process {len(target_records)} records")
        if len(target_records) <= 20:
            for _,record in target_records.iterrows():
                print(f"  - {record['srx_accession']}: {record['tissue'][:80]}{'...' if len(record['tissue']) > 80 else ''}")
        return None
    
    # Run the async main function
    try:
        asyncio.run(
            update_tissue_ontologies(
                target_records=target_records,
                delay=args.delay,
                no_delay=args.no_delay,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()