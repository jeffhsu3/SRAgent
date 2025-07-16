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
        description="Obtain the organism information for SRA experiment accessions",
        formatter_class=CustomFormatter,
        epilog="""
        Examples:
          # Get organism for SRX1234567 and ERX11876752
          python get-srx-organism.py --accessions SRX1234567 ERX11876752

          # Get organism for filtered SRX accessions in the database
          python get-srx-organism.py --use-db

          # Get organism for ALL SRX accessions in the database (not just filtered)
          python get-srx-organism.py --use-db --all-accessions

          # Get organism for all SRX accessions in the database, limited to 10
          python get-srx-organism.py --use-db --limit 10
          
          # Process 100k accessions in batches of 2000 with 3 retries
          python get-srx-organism.py --use-db --batch-size 5000 --retries 3
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
        default="srx_organisms.csv",
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
        "--all-accessions",
        default=False,
        action="store_true",
        help="Process all accessions, not just the ones with tissue information"
    )
    parser.add_argument(
        "--just-star-results",
        default=False,
        action="store_true",
        help="Process only the accessions that have STAR results"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for failed BigQuery or Entrez queries"
    )
    return parser.parse_args()

def get_records(conn, limit: int = None, all_accessions: bool = False, just_star_results: bool = False) -> list[str]:
    """
    Get SRX metadata records.
    
    Args:
        conn: Database connection
        limit: Maximum number of records to return (optional)
        all_accessions: If True, get all accessions; if False, only get filtered ones
    Returns:
        DataFrame with records
    """
    if all_accessions:
        if just_star_results:
            query = """
            SELECT DISTINCT srx_accession
            FROM srx_metadata
            INNER JOIN screcounter_star_results ON srx_metadata.srx_accession = screcounter_star_results.sample
            AND is_illumina = 'yes'
            AND is_single_cell = 'yes'
            AND is_paired_end = 'yes'
            AND lib_prep = '10x_Genomics'
            """
        else:
            query = """
            SELECT DISTINCT srx_accession
            FROM srx_metadata
            WHERE srx_accession IS NOT NULL
            """
    else:
        if just_star_results:
            query = """
            SELECT DISTINCT srx_accession
            FROM srx_metadata
            INNER JOIN screcounter_star_results ON srx_metadata.srx_accession = screcounter_star_results.sample
            AND is_illumina = 'yes'
            AND is_single_cell = 'yes'
            AND is_paired_end = 'yes'
            AND lib_prep = '10x_Genomics'
            WHERE srx_accession IS NOT NULL
            """
        else:
            query = """
            SELECT DISTINCT srx_accession
            FROM srx_metadata
            WHERE srx_accession IS NOT NULL
            """
    if limit:
        query += f" LIMIT {limit}"
    
    return pd.read_sql(query, conn)['srx_accession'].tolist()

def parse_sra_xml_organism(xml_content: str) -> dict:
    """
    Parse SRA XML content and extract organism information.
    Args:
        xml_content: XML content as string or bytes
    Returns:
        dict: Dictionary containing organism information
    """
    
    # Handle bytes input
    if isinstance(xml_content, bytes):
        xml_content = xml_content.decode('utf-8')
    
    # Parse the XML
    root = ET.fromstring(xml_content)
    
    organism_info = {}
    
    # Extract experiment accession for reference
    experiment = root.find('.//EXPERIMENT')
    if experiment is not None:
        organism_info['experiment_accession'] = experiment.get('accession', 'Unknown')
    
    # Extract organism information from SAMPLE
    sample = root.find('.//SAMPLE')
    if sample is not None:
        organism_info['sample_accession'] = sample.get('accession', 'Unknown')
        
        # Look for organism name in SAMPLE_NAME
        sample_name = sample.find('.//SAMPLE_NAME')
        if sample_name is not None:
            scientific_name = sample_name.find('.//SCIENTIFIC_NAME')
            common_name = sample_name.find('.//COMMON_NAME')
            
            if scientific_name is not None:
                organism_info['scientific_name'] = scientific_name.text
            if common_name is not None:
                organism_info['common_name'] = common_name.text
        
        # Look for organism in sample attributes
        sample_attributes = sample.findall('.//SAMPLE_ATTRIBUTE')
        for attr in sample_attributes:
            tag = attr.find('.//TAG')
            value = attr.find('.//VALUE')
            if tag is not None and value is not None:
                tag_text = tag.text.lower() if tag.text else ""
                if 'organism' in tag_text or 'species' in tag_text:
                    organism_info[f'attr_{tag.text}'] = value.text
    
    # Extract organism information from STUDY if available
    study = root.find('.//STUDY')
    if study is not None:
        study_desc = study.find('.//STUDY_DESCRIPTION')
        if study_desc is not None and study_desc.text:
            organism_info['study_description'] = study_desc.text[:200]  # First 200 chars
    
    return organism_info

def get_srx_organism_biopython(srx_accession: str, retries: int = 3) -> dict:
    """Get organism information using Biopython's Entrez module.
    Args:
        srx_accession: SRA experiment accession
        retries: Number of retries for failed Entrez queries
    Returns:
        dict: Organism information
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

    organism_info = parse_sra_xml_organism(xml_data)
    
    # Return the most relevant organism information
    result = {'srx_accession': srx_accession}
    if 'scientific_name' in organism_info:
        result['organism'] = organism_info['scientific_name']
    elif 'common_name' in organism_info:
        result['organism'] = organism_info['common_name']
    else:
        # Look for organism in attributes
        for key, value in organism_info.items():
            if key.startswith('attr_') and 'organism' in key.lower():
                result['organism'] = value
                break
        else:
            result['organism'] = None
    
    result['sample_accession'] = organism_info.get('sample_accession')
    
    return result

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
    Get organism metadata for a list of SRA experiment accessions.
    The metadata fields returned:
    - experiment: SRA experiment accession
    - organism: organism name
    - sample_acc: sample accession
    """
    query = f"""
    SELECT DISTINCT
        m.experiment,
        m.organism,
        m.sample_acc
    FROM `nih-sra-datastore.sra.metadata` as m
    WHERE m.experiment IN ({join_accs(experiment_accessions)})
    AND m.organism IS NOT NULL
    LIMIT {limit}
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
        if args.all_accessions:
            console.print(f"[yellow]Getting ALL accessions from the database[/yellow]")
        else:
            console.print(f"[yellow]Getting filtered accessions from the database[/yellow]")
        with console.status("[bold green]Getting accessions from the database...") as status:
            with db_connect() as conn:
                args.accessions = get_records(conn, limit=args.limit, all_accessions=args.all_accessions)
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
    
    # Find accessions that lack organism information
    if not df_bq.empty:
        # Filter to only the accessions that lack organism information
        df_bq_no_organism = df_bq[df_bq['organism'].isna()]
        # Find accessions not in the bigquery results
        acc_no_organism = df_bq_no_organism['experiment'].tolist()
        acc_no_organism += list(set(args.accessions) - set(df_bq['experiment'].tolist()))
    else:
        acc_no_organism = args.accessions

    # Status on the number of accessions
    console.print(f"[yellow]Accessions lacking organism information: {len(acc_no_organism)}[/yellow]")

    # Run the entrez queries for missing organism information
    entrez_results = []
    if acc_no_organism:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running entrez queries on remaining accessions...", total=len(acc_no_organism))
            for accession in acc_no_organism:
                try:
                    organism_data = get_srx_organism_biopython(accession, retries=args.retries)
                    if organism_data:
                        entrez_results.append(organism_data)
                    else:
                        entrez_results.append({
                            'srx_accession': accession,
                            'organism': None,
                            'sample_accession': None
                        })
                except Exception as e:
                    console.print(f"[red]Error processing {accession}: {e}[/red]")
                    entrez_results.append({
                        'srx_accession': accession,
                        'organism': None,
                        'sample_accession': None
                    })
                
                progress.update(task, completed=len(entrez_results))
                time.sleep(0.33)

    # Combine the results
    if not df_bq.empty:
        df_bq_organisms = df_bq[~df_bq['organism'].isna()].rename(
            columns={'experiment': 'srx_accession', 'sample_acc': 'sample_accession'}
        )[['srx_accession', 'organism',  'sample_accession']]
    else:
        # If no results, create an empty dataframe
        df_bq_organisms = pd.DataFrame(columns=['srx_accession', 'organism', 'sample_accession'])
    
    # Create DataFrame from Entrez results
    df_entrez = pd.DataFrame(entrez_results)
    if df_entrez.empty:
        df_entrez = pd.DataFrame(columns=['srx_accession', 'organism',  'sample_accession'])
    
    # Concatenate the results
    df_results = pd.concat([df_bq_organisms, df_entrez], ignore_index=True)

    # Filter out records not in args.accessions
    df_results = df_results[df_results['srx_accession'].isin(args.accessions)]

    # For duplicate srx_accession, keep the first occurrence (BigQuery results have priority)
    df_results = df_results.drop_duplicates(subset='srx_accession', keep='first')

    # Sort by organism for better readability
    df_results = df_results.sort_values(by='organism', na_position='last')

    # Save the results
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
