#!/usr/bin/env python3
"""
scRNA-seq dbGaP Cell Count Analyzer

A command-line tool to search dbGaP for single-cell RNA sequencing datasets
and extract cell count information using OpenAI LLM with structured outputs.
"""

import os
import re
import time
import argparse
import sys
from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import pandas as pd
from Bio import Entrez
import xml.etree.ElementTree as ET

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Make sure to set environment variables manually.")


class CellCount(BaseModel):
    """Individual cell count mention with context"""
    count: int = Field(description="The numerical cell count")
    context: str = Field(description="The context or description of this cell count")
    count_type: str = Field(description="Type of count: 'total', 'per_sample', 'after_qc', 'before_qc', 'analyzed', 'sequenced', 'other'")
    confidence: float = Field(description="Confidence in this extraction (0.0 to 1.0)", ge=0.0, le=1.0)


class CellCountExtraction(BaseModel):
    """Complete cell count extraction results"""
    cell_counts: List[CellCount] = Field(description="List of all cell counts found in the text")
    has_single_cell_data: bool = Field(description="Whether the text indicates this is single-cell data")
    technology_platform: Optional[str] = Field(description="Sequencing technology/platform mentioned (e.g., '10X Genomics', 'Drop-seq', 'Smart-seq')")
    summary: str = Field(description="Brief summary of the cell count information found")


class StudyCellAnalysis(BaseModel):
    """Overall analysis of cell counts for a study"""
    primary_cell_count: Optional[int] = Field(description="The most likely primary/total cell count for the study")
    all_cell_counts: List[int] = Field(description="All unique cell counts mentioned")
    is_single_cell_study: bool = Field(description="Whether this appears to be a single-cell study")
    technology_platform: Optional[str] = Field(description="Primary technology platform used")
    confidence_score: float = Field(description="Overall confidence in the analysis (0.0 to 1.0)", ge=0.0, le=1.0)
    notes: str = Field(description="Additional notes about the cell count analysis")


class LLMCellCountExtractor:
    """Extract cell count information using OpenAI LLM with structured outputs"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.1, max_tokens=2000):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in analyzing scientific text to extract cell count information from single-cell RNA sequencing studies.

Your task is to:
1. Identify all mentions of cell counts in the text
2. Determine the context and type of each count
3. Assess whether this is single-cell data
4. Identify the technology platform used
5. Provide confidence scores for your extractions

Pay attention to:
- Different ways numbers might be written (1,000 vs 1000 vs 1k vs one thousand)
- Context clues about what type of count it is
- Quality control filtering steps
- Multiple samples or batches
- Technology-specific terminology

Be conservative with confidence scores - only high confidence (>0.8) for very clear mentions."""),
            ("human", """Please analyze the following scientific text and extract all cell count information:

Text: {text}

Extract all cell counts with their context and provide a structured analysis.""")
        ])
        
        # Set up structured output
        self.parser = PydanticOutputParser(pydantic_object=CellCountExtraction)
        self.chain = self.prompt | self.llm.with_structured_output(CellCountExtraction)
    
    def extract_cell_counts(self, text: str, max_text_length: int = 8000) -> CellCountExtraction:
        """Extract cell counts from text using LLM"""
        if not text or len(text.strip()) < 10:
            return CellCountExtraction(
                cell_counts=[],
                has_single_cell_data=False,
                technology_platform=None,
                summary="No meaningful text to analyze"
            )
        
        try:
            # Truncate very long texts to avoid token limits
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            result = self.chain.invoke({"text": text})
            return result
            
        except Exception as e:
            print(f"Error in LLM extraction: {e}")
            return CellCountExtraction(
                cell_counts=[],
                has_single_cell_data=False,
                technology_platform=None,
                summary=f"Error during extraction: {str(e)}"
            )
    
    def analyze_study_cells(self, extraction: CellCountExtraction) -> StudyCellAnalysis:
        """Analyze extraction results to determine primary cell count"""
        
        if not extraction.cell_counts:
            return StudyCellAnalysis(
                primary_cell_count=None,
                all_cell_counts=[],
                is_single_cell_study=extraction.has_single_cell_data,
                technology_platform=extraction.technology_platform,
                confidence_score=0.0,
                notes="No cell counts found in text"
            )
        
        # Extract all counts and sort by confidence
        all_counts = [cc.count for cc in extraction.cell_counts]
        high_confidence_counts = [cc for cc in extraction.cell_counts if cc.confidence > 0.7]
        
        # Determine primary count (highest confidence total/analyzed count)
        primary_count = None
        max_confidence = 0
        
        for cc in extraction.cell_counts:
            if cc.count_type in ['total', 'analyzed', 'sequenced'] and cc.confidence > max_confidence:
                primary_count = cc.count
                max_confidence = cc.confidence
        
        # If no primary count found, use highest confidence count
        if primary_count is None and high_confidence_counts:
            primary_count = max(high_confidence_counts, key=lambda x: x.confidence).count
        
        # Calculate overall confidence
        avg_confidence = sum(cc.confidence for cc in extraction.cell_counts) / len(extraction.cell_counts)
        
        return StudyCellAnalysis(
            primary_cell_count=primary_count,
            all_cell_counts=sorted(list(set(all_counts))),
            is_single_cell_study=extraction.has_single_cell_data,
            technology_platform=extraction.technology_platform,
            confidence_score=avg_confidence,
            notes=f"Found {len(extraction.cell_counts)} cell count mentions. {extraction.summary}"
        )


def parse_study_xml_for_cells(root):
    """Parse XML to extract detailed study information including potential cell counts"""
    
    studies = {}
    
    for study in root.findall('.//Study'):
        study_id = study.get('id', '')
        if not study_id:
            # Try to get ID from other attributes
            study_id = study.get('accession', '')
        
        study_info = {
            'detailed_description': '',
            'methods': '',
            'design': '',
            'documents': [],
            'variables': []
        }
        
        # Get detailed description
        desc_elem = study.find('.//Description')
        if desc_elem is not None and desc_elem.text:
            study_info['detailed_description'] = desc_elem.text
        
        # Get methods/design information
        method_elem = study.find('.//Method')
        if method_elem is not None and method_elem.text:
            study_info['methods'] = method_elem.text
            
        design_elem = study.find('.//StudyDesign')
        if design_elem is not None and design_elem.text:
            study_info['design'] = design_elem.text
        
        # Get document information
        for doc in study.findall('.//Document'):
            doc_name = doc.get('name', '')
            doc_desc = doc.find('.//Description')
            if doc_desc is not None and doc_desc.text:
                study_info['documents'].append(f"{doc_name}: {doc_desc.text}")
        
        # Get variable information (might contain cell count variables)
        for var in study.findall('.//Variable'):
            var_name = var.get('var_name', '')
            var_desc = var.find('.//Description')
            if var_desc is not None and var_desc.text:
                if any(term in var_name.lower() or term in var_desc.text.lower() 
                       for term in ['cell', 'count', 'number']):
                    study_info['variables'].append(f"{var_name}: {var_desc.text}")
        
        studies[study_id] = study_info
    
    return studies


def get_comprehensive_study_data(study_ids, cell_extractor, batch_size=20, verbose=False):
    """Get comprehensive study data using LLM for cell count extraction"""
    
    all_study_data = []
    
    for i in range(0, len(study_ids), batch_size):
        batch_ids = study_ids[i:i+batch_size]
        if verbose:
            print(f"Processing batch {i//batch_size + 1}: {len(batch_ids)} studies")
        
        study_data = {}
        
        # Get summaries
        try:
            handle = Entrez.esummary(db="gap", id=",".join(batch_ids))
            summaries = Entrez.read(handle)
            handle.close()
            
            for summary in summaries:
                study_data[summary['Id']] = {
                    'id': summary['Id'],
                    'accession': summary.get('Accession', ''),
                    'title': summary.get('Title', ''),
                    'description': summary.get('Description', ''),
                    'study_types': summary.get('StudyTypes', ''),
                    'molecular_data_types': summary.get('MolecularDataTypes', ''),
                    'organism': summary.get('OrganismName', ''),
                }
        except Exception as e:
            print(f"Error getting summaries: {e}")
        
        # Get full XML records
        try:
            handle = Entrez.efetch(db="gap", id=",".join(batch_ids), rettype="xml")
            xml_data = handle.read()
            handle.close()
            
            root = ET.fromstring(xml_data)
            xml_details = parse_study_xml_for_cells(root)
            
            for study_id, xml_info in xml_details.items():
                if study_id in study_data:
                    study_data[study_id].update(xml_info)
                    
        except Exception as e:
            print(f"Error getting XML records: {e}")
        
        # Use LLM to extract cell counts
        for study_id, data in study_data.items():
            if verbose:
                print(f"  Analyzing study {data['accession']} with LLM...")
            
            # Combine all relevant text
            all_text = ' '.join(filter(None, [
                data.get('title', ''),
                data.get('description', ''),
                data.get('detailed_description', ''),
                data.get('methods', ''),
                data.get('design', ''),
                ' '.join(data.get('documents', []))
            ]))
            
            # Extract cell counts using LLM
            extraction = cell_extractor.extract_cell_counts(all_text)
            analysis = cell_extractor.analyze_study_cells(extraction)
            
            # Add LLM results to study data
            data.update({
                'llm_extraction': extraction,
                'cell_analysis': analysis,
                'primary_cell_count': analysis.primary_cell_count,
                'all_cell_counts': analysis.all_cell_counts,
                'is_single_cell_study': analysis.is_single_cell_study,
                'technology_platform': analysis.technology_platform,
                'confidence_score': analysis.confidence_score,
                'analysis_notes': analysis.notes
            })
        
        all_study_data.extend(study_data.values())
        
        # Rate limiting
        time.sleep(2)
    
    return all_study_data


def search_scrna_studies(search_terms, max_results_per_term=100, verbose=False):
    """Search for scRNA-seq studies on dbGaP"""
    
    if verbose:
        print("Searching for scRNA-seq studies...")
    
    all_study_ids = set()
    
    for term in search_terms:
        if verbose:
            print(f"  Searching: '{term}'")
        
        handle = Entrez.esearch(db="gap", term=term, retmax=max_results_per_term)
        results = Entrez.read(handle)
        handle.close()
        
        all_study_ids.update(results['IdList'])
        if verbose:
            print(f"    Found {results['Count']} studies")
    
    if verbose:
        print(f"Total unique studies found: {len(all_study_ids)}")
    
    return list(all_study_ids)


def analyze_llm_results(studies):
    """Analyze LLM extraction results"""
    
    results = {
        'high_confidence_counts': [],
        'medium_confidence_counts': [],
        'low_confidence_counts': [],
        'single_cell_studies': [],
        'technology_platforms': {},
        'statistics': {}
    }
    
    all_primary_counts = []
    
    for study in studies:
        if not study.get('cell_analysis'):
            continue
            
        analysis = study['cell_analysis']
        
        study_summary = {
            'accession': study['accession'],
            'title': study['title'][:100] + '...' if len(study['title']) > 100 else study['title'],
            'primary_cell_count': analysis.primary_cell_count,
            'all_counts': analysis.all_cell_counts,
            'confidence': analysis.confidence_score,
            'platform': analysis.technology_platform,
            'notes': analysis.notes
        }
        
        # Categorize by confidence
        if analysis.confidence_score >= 0.8:
            results['high_confidence_counts'].append(study_summary)
        elif analysis.confidence_score >= 0.5:
            results['medium_confidence_counts'].append(study_summary)
        else:
            results['low_confidence_counts'].append(study_summary)
        
        # Track single-cell studies
        if analysis.is_single_cell_study:
            results['single_cell_studies'].append(study_summary)
        
        # Track technology platforms
        if analysis.technology_platform:
            platform = analysis.technology_platform
            results['technology_platforms'][platform] = results['technology_platforms'].get(platform, 0) + 1
        
        # Collect primary counts for statistics
        if analysis.primary_cell_count:
            all_primary_counts.append(analysis.primary_cell_count)
    
    # Calculate statistics
    if all_primary_counts:
        results['statistics'] = {
            'total_studies_with_counts': len(all_primary_counts),
            'min_cells': min(all_primary_counts),
            'max_cells': max(all_primary_counts),
            'median_cells': sorted(all_primary_counts)[len(all_primary_counts)//2],
            'mean_cells': sum(all_primary_counts) / len(all_primary_counts),
            'high_confidence_studies': len(results['high_confidence_counts']),
            'single_cell_studies': len(results['single_cell_studies'])
        }
    
    return results


def export_results(results, output_file):
    """Export LLM analysis results to CSV"""
    
    all_data = []
    
    # Combine all categories
    for category, studies in [
        ('high_confidence', results['high_confidence_counts']),
        ('medium_confidence', results['medium_confidence_counts']),
        ('low_confidence', results['low_confidence_counts'])
    ]:
        for study in studies:
            all_data.append({
                'accession': study['accession'],
                'title': study['title'],
                'primary_cell_count': study['primary_cell_count'],
                'all_cell_counts': ', '.join(map(str, study['all_counts'])) if study['all_counts'] else '',
                'confidence_category': category,
                'confidence_score': study['confidence'],
                'technology_platform': study['platform'],
                'notes': study['notes']
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Results exported to {output_file}")
    
    return df


def display_results(analysis, verbose=False):
    """Display analysis results"""
    
    print(f"\n=== Results Summary ===")
    if analysis['statistics']:
        stats = analysis['statistics']
        print(f"Studies with cell counts: {stats['total_studies_with_counts']}")
        print(f"High confidence studies: {stats['high_confidence_studies']}")
        print(f"Single-cell studies identified: {stats['single_cell_studies']}")
        if stats['total_studies_with_counts'] > 0:
            print(f"Cell count range: {stats['min_cells']:,} - {stats['max_cells']:,}")
            print(f"Median cell count: {stats['median_cells']:,}")
            print(f"Mean cell count: {stats['mean_cells']:,.0f}")
    else:
        print("No studies with cell counts found.")
    
    if analysis['technology_platforms']:
        print(f"\nTechnology platforms found:")
        for platform, count in sorted(analysis['technology_platforms'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {platform}: {count} studies")
    
    if verbose and analysis['high_confidence_counts']:
        print(f"\nTop high-confidence studies:")
        for i, study in enumerate(analysis['high_confidence_counts'][:10], 1):
            cell_count = f"{study['primary_cell_count']:,}" if study['primary_cell_count'] else "Unknown"
            print(f"  {i}. {study['accession']}: {cell_count} cells ({study['confidence']:.2f})")
            print(f"     {study['title']}")
            if study['platform']:
                print(f"     Platform: {study['platform']}")
            print()


def setup_environment():
    """Setup environment variables and validate configuration"""
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    if not os.getenv("NCBI_EMAIL"):
        raise ValueError("NCBI_EMAIL environment variable is not set")
    
    # Set Entrez email
    Entrez.email = os.getenv("NCBI_EMAIL")


def main():
    """Main function with CLI argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Search dbGaP for scRNA-seq datasets and extract cell count information using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Search parameters
    parser.add_argument(
        "--search-terms",
        nargs="+",
        default=["single cell RNA sequencing", "10X Genomics", "scRNA-seq"],
        help="Search terms for finding scRNA-seq studies"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum results per search term"
    )
    
    parser.add_argument(
        "--max-studies",
        type=int,
        default=50,
        help="Maximum number of studies to analyze (0 for no limit)"
    )
    
    # LLM parameters
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use for cell count extraction"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM generation"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum tokens for LLM response"
    )
    
    # Processing parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for processing studies"
    )
    
    # Output parameters
    parser.add_argument(
        "--output",
        "-o",
        default="scrna_dbgap_analysis.csv",
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search for studies but don't run LLM analysis"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup environment
        setup_environment()
        
        # Initialize LLM extractor
        if not args.dry_run:
            if args.verbose:
                print(f"Initializing LLM extractor with model: {args.model}")
            cell_extractor = LLMCellCountExtractor(
                model_name=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
        
        # Search for studies
        study_ids = search_scrna_studies(
            args.search_terms,
            args.max_results,
            args.verbose
        )
        
        if not study_ids:
            print("No studies found matching search criteria.")
            return
        
        # Limit number of studies if specified
        if args.max_studies > 0:
            study_ids = study_ids[:args.max_studies]
            if args.verbose:
                print(f"Limited to first {len(study_ids)} studies")
        
        if args.dry_run:
            print(f"Dry run: Found {len(study_ids)} studies to analyze")
            print("Study IDs:", study_ids[:10], "..." if len(study_ids) > 10 else "")
            return
        
        # Get comprehensive study data with LLM analysis
        print(f"\nAnalyzing {len(study_ids)} studies...")
        studies = get_comprehensive_study_data(
            study_ids,
            cell_extractor,
            args.batch_size,
            args.verbose
        )
        
        # Analyze results
        if args.verbose:
            print("\nAnalyzing extraction results...")
        analysis = analyze_llm_results(studies)
        
        # Display results
        display_results(analysis, args.verbose)
        
        # Export results
        if args.output:
            print(f"\nExporting results...")
            export_results(analysis, args.output)
        
        print(f"\nAnalysis complete!")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()