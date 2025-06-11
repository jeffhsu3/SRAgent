#!/usr/bin/env python
import os
import sys
import argparse
import time
from datetime import datetime
from Bio import Entrez
import pandas as pd
import xml.etree.ElementTree as ET
from SRAgent.db.connect import db_connect
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

console = Console()


def parse_args():
    """Parse command line arguments."""
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description="Obtain the submission dates of SRX/ERX accessions",
        formatter_class=CustomFormatter,
        epilog="""
        Examples:
          # Get submission date for SRX1234567 and ERX11876752
          python get-srx-date.py --accessions SRX1234567 ERX11876752

          # Get submission date for all SRX accessions in the database
          python get-srx-date.py --use-db

          # Get submission date for all SRX accessions in the database, limited to 10
          python get-srx-date.py --use-db --limit 10
        """
    )
    parser.add_argument(
        "--accessions", type=str, nargs="+", default=None,
        help="SRX accessions to use"
    )
    parser.add_argument(
        "--use-db", action="store_true", default=False,
        help="Use the database to get the SRX accessions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="srx_dates.csv",
        help="Output file name"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of records to process"
    )
    parser.add_argument(
        "--tenant",
        type=str,
        choices=["test", "prod"],
        help="Database tenant to use (optional)"
    )
    
    return parser.parse_args()

def get_records(conn, limit: int = None) -> pd.DataFrame:
    """
    Get SRX metadata records that have tissue information but lack tissue_ontology_term_id.
    
    Args:
        conn: Database connection
        limit: Maximum number of records to return (optional)
    Returns:
        DataFrame with records
    """
    query = """
    SELECT DISTINCT srx_accession
    FROM srx_metadata
    INNER JOIN screcounter_star_results ON srx_metadata.srx_accession = screcounter_star_results.sample
    AND is_illumina = 'yes'
    AND is_single_cell = 'yes'
    AND is_paired_end = 'yes'
    AND lib_prep = '10x_Genomics'
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    return pd.read_sql(query, conn)['srx_accession'].tolist()

def parse_sra_xml_dates(xml_content):
    """
    Parse SRA XML content and extract all available dates.
    
    Args:
        xml_content: XML content as string or bytes
    
    Returns:
        dict: Dictionary containing all found dates
    """
    
    # Handle bytes input
    if isinstance(xml_content, bytes):
        xml_content = xml_content.decode('utf-8')
    
    # Parse the XML
    root = ET.fromstring(xml_content)
    
    dates_info = {}
    
    # Extract experiment accession for reference
    experiment = root.find('.//EXPERIMENT')
    if experiment is not None:
        dates_info['experiment_accession'] = experiment.get('accession', 'Unknown')
    
    # Extract submission information (though submission_date might not be present)
    submission = root.find('.//SUBMISSION')
    if submission is not None:
        dates_info['submission_accession'] = submission.get('accession', 'Unknown')
        dates_info['submission_broker'] = submission.get('broker_name', 'Unknown')
        # Check if submission_date exists (it might not always be present)
        if 'submission_date' in submission.attrib:
            dates_info['submission_date'] = submission.get('submission_date')
        else:
            dates_info['submission_date'] = 'Not available in XML'
    
    # Extract RUN dates (these are often the most useful)
    runs = root.findall('.//RUN')
    run_dates = []
    
    for run in runs:
        run_info = {
            'run_accession': run.get('accession', 'Unknown'),
            'published_date': run.get('published', 'Not available'),
            'is_public': run.get('is_public', 'Unknown')
        }
        run_dates.append(run_info)
    
    dates_info['runs'] = run_dates
    
    # Extract file dates from SRAFiles
    sra_files = root.findall('.//SRAFile')
    file_dates = []
    
    for sra_file in sra_files:
        file_info = {
            'filename': sra_file.get('filename', 'Unknown'),
            'date': sra_file.get('date', 'Not available'),
            'size': sra_file.get('size', 'Unknown')
        }
        file_dates.append(file_info)
    
    dates_info['file_dates'] = file_dates
    
    return dates_info

def find_earliest_date(dates_info):
    """
    Find the earliest date from all available dates.
    This is often the closest to the actual submission date.
    """
    all_dates = []
    
    # Collect all date strings
    for run in dates_info.get('runs', []):
        if run['published_date'] != 'Not available':
            all_dates.append(('published_date', run['published_date'], run['run_accession']))
    
    for file_info in dates_info.get('file_dates', []):
        if file_info['date'] != 'Not available':
            all_dates.append(('file_date', file_info['date'], file_info['filename']))
    
    if not all_dates:
        return None
    
    # Parse dates and find earliest
    parsed_dates = []
    for date_type, date_str, identifier in all_dates:
        try:
            # Parse the datetime string
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            parsed_dates.append((dt, date_type, date_str, identifier))
        except ValueError:
            continue
    
    if parsed_dates:
        earliest = min(parsed_dates, key=lambda x: x[0])
        return {
            'earliest_date': earliest[2],
            'date_type': earliest[1],
            'source': earliest[3],
            'parsed_datetime': earliest[0]
        }
    
    return None

def get_srx_submission_date_biopython(srx_accession):
    """Get submission/run date using Biopython's Entrez module.
    
    Attempts to extract the run_date from RUN element first,
    then falls back to submission_date from SUBMISSION element if available.
    """
    # Search for the SRX accession
    handle = Entrez.esearch(db="sra", term=srx_accession)
    search_results = Entrez.read(handle)
    handle.close()
    
    if not search_results['IdList']:
        return None
    
    uid = search_results['IdList'][0]
    
    # Fetch detailed record
    handle = Entrez.efetch(db="sra", id=uid, retmode="xml")
    xml_data = handle.read()
    handle.close()

    dates = parse_sra_xml_dates(xml_data)

    earliest_date = find_earliest_date(dates)
    return earliest_date['parsed_datetime'].strftime('%Y-%m-%d')

def main(args):
    # Set your email (required by NCBI)
    Entrez.email = os.getenv("EMAIL", "blank@gmail.com")
    Entrez.api_key = os.getenv("NCBI_API_KEY")

    # Set the database tenant, 
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant
    if not os.getenv("DYNACONF"):
        print("Warning: No tenant specified. Using the 'test' tenant", file=sys.stderr)
        os.environ["DYNACONF"] = "test"

    # get the accessions
    if args.use_db:
        with db_connect() as conn:
            args.accessions = get_records(conn, limit=args.limit)

    results = {'srx_accession': [], 'date': []}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing accessions", total=len(args.accessions))
        for accession in args.accessions:
            date = get_srx_submission_date_biopython(accession)
            results['srx_accession'].append(accession)
            results['date'].append(date)
            progress.update(task, completed=len(results['srx_accession']))
            time.sleep(0.33)

    # save the results
    pd.DataFrame(results).to_csv(args.output, index=False)
    console.print(f"[green]Results saved to {args.output}[/green]")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    args = parse_args()
    main(args)

