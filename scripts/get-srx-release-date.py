#!/usr/bin/env python
import os
import sys
import argparse
import time
from datetime import datetime
from typing import Annotated, List, Callable
import xml.etree.ElementTree as ET
from Bio import Entrez
from google.cloud import bigquery
import pandas as pd
from SRAgent.db.connect import db_connect
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from functools import wraps

console = Console()


def retry_with_backoff(retries: int, backoff_base: int = 2) -> Callable:
    """Decorator to retry a function with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        console.print(f"[red]Final attempt failed for {func.__name__}: {e}[/red]")
                        raise
                    wait_time = backoff_base ** attempt
                    console.print(f"[yellow]{func.__name__} attempt {attempt + 1}/{retries} failed: {e}[/yellow]")
                    console.print(f"[yellow]Retrying in {wait_time} seconds...[/yellow]")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description="Obtain the release dates for SRA experiment accessions",
        formatter_class=CustomFormatter,
        epilog="""
        Examples:
          # Get submission date for SRX1234567 and ERX11876752
          python get-srx-date.py --accessions SRX1234567 ERX11876752

          # Get submission date for all SRX accessions in the database
          python get-srx-date.py --use-db

          # Get submission date for all SRX accessions in the database, limited to 10
          python get-srx-date.py --use-db --limit 10
          
          # Process 100k accessions in batches of 2000 with 3 retries
          python get-srx-date.py --use-db --batch-size 5000 --retries 3
        """
    )
    parser.add_argument(
        "--accessions", type=str, nargs="+", default=None,
        help="SRX accessions to use; can be a CSV file with a 'srx_accession' column"
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
        default="test",
        help="Database tenant to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of accessions to process per BigQuery batch"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for failed BigQuery or Entrez queries"
    )
    return parser.parse_args()

def get_records(conn, limit: int = None) -> list[str]:
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

def parse_sra_xml_dates(xml_content: str) -> dict:
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

def find_earliest_date(dates_info: dict) -> dict:
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

def get_srx_submission_date_biopython(srx_accession: str, retries: int = 3) -> str:
    """Get submission/run date using Biopython's Entrez module.
    Attempts to extract the run_date from RUN element first,
    then falls back to submission_date from SUBMISSION element if available.
    Args:
        srx_accession: SRA experiment accession
        retries: Number of retries for failed Entrez queries
    Returns:
        str: Release date in YYYY-MM-DD format
    """
    @retry_with_backoff(retries)
    def _fetch_with_retry(srx_accession):
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
        
        return xml_data
    
    xml_data = _fetch_with_retry(srx_accession)
    if xml_data is None:
        return None

    dates = parse_sra_xml_dates(xml_data)
    earliest_date = find_earliest_date(dates)
    
    if earliest_date:
        return earliest_date['parsed_datetime'].strftime('%Y-%m-%d')
    return None

def join_accs(accessions: list[str]) -> str:
    """
    Join a list of accessions into a string.
    Args:
        accessions: list of accessions
    Returns:
        str: comma separated string of accessions
    """
    return ', '.join([f"'{acc}'" for acc in accessions])

def get_study_metadata(
    experiment_accessions: list[str],
    client: bigquery.Client,
    limit: int = 100,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Get study-level metadata for a list of SRA experimentaccessions.
    The metadata fields returned:
    - sra_study: SRA study accession (the query accession)
    - bioproject: BioProject accession (parent of study)
    - experiments: Comma-separated list of associated experiment accessions (SRX)
    - earliest_release_date: Earliest release date among experiments in the study
    - latest_release_date: Latest release date among experiments in the study
    """
    query = f"""
    WITH distinct_values AS (
        SELECT DISTINCT
            m.sra_study,
            m.bioproject,
            m.experiment,
            m.releasedate
        FROM `nih-sra-datastore.sra.metadata` as m
        WHERE m.experiment IN ({join_accs(experiment_accessions)})
        LIMIT {limit}
    )
    SELECT 
        sra_study,
        bioproject,
        STRING_AGG(experiment, ',') as experiments,
        MIN(releasedate) as earliest_release_date,
        MAX(releasedate) as latest_release_date
    FROM distinct_values
    GROUP BY sra_study, bioproject
    """
    
    @retry_with_backoff(retries)
    def _execute_query():
        # Execute query and convert to DataFrame
        query_job = client.query(query)
        df = query_job.to_dataframe()
        if df.empty:
            console.print(f"[red]No bigquery results found...[/red]")
        return df
    
    return _execute_query()

def batch_list(lst: list[str], batch_size: int) -> list[list[str]]:
    """Yield successive batches from list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main(args: argparse.Namespace):
    # Set your email (required by NCBI)
    Entrez.email = os.getenv("EMAIL", "blank@gmail.com")
    Entrez.api_key = os.getenv("NCBI_API_KEY")

    # Set the database tenant, 
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant
    if not os.getenv("DYNACONF"):
        console.print("[yellow]Warning: No tenant specified. Using the 'test' tenant[/yellow]")
        os.environ["DYNACONF"] = "test"
    if not args.tenant:
        args.tenant = os.getenv("DYNACONF")

    # get the accessions
    if args.use_db:
        console.print(f"[yellow]Database tenant: {args.tenant}[/yellow]")
        with console.status("[bold green]Getting accessions from the database...") as status:
            with db_connect() as conn:
                args.accessions = get_records(conn, limit=args.limit)
        console.print(f"[cyan]Obtained {len(args.accessions)} accessions from the database[/cyan]")
    elif len(args.accessions) == 1 and os.path.exists(args.accessions[0]):
        console.print(f"[yellow]Reading accessions from {args.accessions[0]}[/yellow]")
        args.accessions = pd.read_csv(args.accessions[0])['srx_accession'].tolist()
    else:
        console.print(f"[yellow]Using {len(args.accessions)} accessions from command line[/yellow]")
    
    # Initialize BigQuery client
    client = bigquery.Client()
    
    # Process accessions in batches
    total_accessions = len(args.accessions)
    num_batches = (total_accessions + args.batch_size - 1) // args.batch_size
    
    console.print(f"[cyan]Processing {total_accessions} accessions in {num_batches} batches of {args.batch_size}[/cyan]")
    console.print(f"[cyan]Retry attempts configured: {args.retries}[/cyan]")
    
    # Collect all batch results
    all_batch_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        batch_task = progress.add_task("Processing BigQuery batches...", total=num_batches)
        
        for batch_idx, batch in enumerate(batch_list(args.accessions, args.batch_size)):
            progress.update(batch_task, description=f"Processing batch {batch_idx + 1}/{num_batches}")
            
            # Query BigQuery for this batch
            try:
                df_batch = get_study_metadata(batch, client=client, retries=args.retries)
                all_batch_results.append(df_batch)
                time.sleep(0.5)
            except Exception as e:
                console.print(f"[red]Error processing batch {batch_idx + 1}: {e}[/red]")
                continue
            
            progress.update(batch_task, advance=1)
    
    # Combine all batch results
    if all_batch_results:
        df_bq = pd.concat(all_batch_results, ignore_index=True)
        console.print(f"[green]Successfully processed {len(all_batch_results)} batches[/green]")
    else:
        console.print("[red]No successful batch queries[/red]")
        df_bq = pd.DataFrame()
    
    ## explode comma separated experiments
    if not df_bq.empty:
        df_bq['experiments'] = df_bq['experiments'].str.split(',')
        df_bq = df_bq.explode('experiments')

        ## filter to only the accessions that lack a date
        df_bq_no_date = df_bq[df_bq['earliest_release_date'].isna()]
        ## find accession not in the bigquery results
        acc_no_date = df_bq_no_date['experiments'].tolist()
        acc_no_date += list(set(args.accessions) - set(df_bq['experiments'].tolist()))
    else:
        acc_no_date = args.accessions

    ## status on the number of accessions
    console.print(f"[yellow]Accessions lacking dates: {len(acc_no_date)}[/yellow]")

    # run the entrez queries
    results = {'srx_accession': [], 'release_date': []}
    if acc_no_date:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running entrez queries on remaining accessions...", total=len(acc_no_date))
            for accession in acc_no_date:
                try:
                    date = get_srx_submission_date_biopython(accession, retries=args.retries)
                    results['srx_accession'].append(accession)
                    results['release_date'].append(date)
                except Exception as e:
                    console.print(f"[red]Error processing {accession}: {e}[/red]")
                    results['srx_accession'].append(accession)
                    results['release_date'].append(None)
                
                progress.update(task, completed=len(results['srx_accession']))
                time.sleep(0.33)

    # combine the results
    if not df_bq.empty:
        df_bq_dates = df_bq[~df_bq['earliest_release_date'].isna()].drop(
            columns=['sra_study', 'bioproject', 'latest_release_date']
        ).rename(columns={'earliest_release_date': 'release_date', 'experiments': 'srx_accession'})
    else:
        # if no results, create an empty dataframe
        df_bq_dates = pd.DataFrame(columns=['srx_accession', 'release_date'])
    
    # Ensure consistent dtypes before concatenation
    df_entrez = pd.DataFrame(results)
    ## if no results, create an empty dataframe
    if df_entrez.empty:
        df_entrez = pd.DataFrame(columns=['srx_accession', 'release_date'])
    
    # Convert dates to datetime
    for df in [df_bq_dates, df_entrez]:
        if not df.empty:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    # concatenate the results
    df_results = pd.concat([df_bq_dates, df_entrez], ignore_index=True)

    # convert release_date to YYYY-MM-DD, handling NA values
    df_results['release_date'] = df_results['release_date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None
    )

    # filter out records not in args.accessions
    df_results = df_results[df_results['srx_accession'].isin(args.accessions)].sort_values(by='release_date')

    # for duplicate srx_accession, keep the earliest release_date
    df_results = df_results.drop_duplicates(subset='srx_accession', keep='first')

    # save the results
    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    df_results.to_csv(args.output, index=False)
    console.print(f"[green]Results saved to {args.output}[/green]")
    console.print(f"[cyan]Total records processed: {len(df_results)}[/cyan]")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    args = parse_args()
    main(args)
