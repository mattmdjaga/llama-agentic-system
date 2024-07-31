import sqlite3
from typing import Dict, List, Union
from llama_agentic_system.tools.custom import SingleMessageCustomTool
from llama_models.llama3_1.api.datatypes import ToolParamDefinition
import pandas as pd
from collections import defaultdict


def extract_gene_interactions() -> pd.DataFrame:
    """
    Extracts gene interactions from the SQLite database.

    Returns:
        pd.DataFrame: A DataFrame containing gene interactions and associated diseases.
    """
    conn = sqlite3.connect("../gene_pathway_db.sqlite")
    query = """
    SELECT p1.gene_id AS gene1, p2.gene_id AS gene2, p1.disease
    FROM pathways p1
    JOIN pathways p2 ON p1.entry_id = p2.entry_id
    WHERE p1.gene_id != p2.gene_id
    """
    interactions = pd.read_sql_query(query, conn)
    conn.close()
    return interactions


def build_interaction_network(interactions: pd.DataFrame) -> Dict[str, List[tuple]]:
    """
    Builds an interaction network from the gene interactions DataFrame.

    Args:
        interactions (pd.DataFrame): A DataFrame containing gene interactions and associated diseases.

    Returns:
        Dict[str, List[tuple]]: A dictionary representing the interaction network.
    """
    network = defaultdict(list)
    for _, row in interactions.iterrows():
        network[row["gene1"]].append((row["gene2"], row["disease"]))
        network[row["gene2"]].append((row["gene1"], row["disease"]))
    return network


def analyze_network(
    network: Dict[str, List[tuple]], gene: str
) -> Union[Dict[str, List[str]], str]:
    """
    Analyzes the interaction network to find downstream interactions for a given gene.

    Args:
        network (Dict[str, List[tuple]]): The interaction network.
        gene (str): The gene ID to analyze.

    Returns:
        Union[Dict[str, List[str]], str]: A dictionary of diseases and associated interacting genes, or an error message.
    """
    if gene not in network:
        return f"No interactions found for gene {gene}"

    interactions = network[gene]
    analysis_result = {}

    for neighbor, disease in interactions:
        if disease not in analysis_result:
            analysis_result[disease] = []
        analysis_result[disease].append(neighbor)

    return analysis_result


class DownstreamAnalysisTool(SingleMessageCustomTool):
    """Tool to perform downstream analysis on gene interactions"""

    def __init__(self) -> None:
        interactions = extract_gene_interactions()
        self.network = build_interaction_network(interactions)

    def get_name(self) -> str:
        """Returns the name of the tool."""
        return "perform_downstream_analysis"

    def get_description(self) -> str:
        """Returns the description of the tool."""
        return "Analyze gene interactions and their effects on disease pathways"

    def get_params_definition(self) -> Dict[str, ToolParamDefinition]:
        """Returns the parameter definitions for the tool."""
        return {
            "gene_id": ToolParamDefinition(
                param_type="str",
                description="The gene ID for which to perform downstream analysis",
                required=True,
            )
        }

    async def run_impl(self, gene_id: str) -> Union[Dict[str, List[str]], str]:
        """
        Executes the downstream analysis.

        Args:
            gene_id (str): The gene ID to analyze.

        Returns:
            Union[Dict[str, List[str]], str]: The analysis result.
        """
        analysis_result = analyze_network(self.network, gene_id)
        return analysis_result


class GeneDiseaseAssociationTool(SingleMessageCustomTool):
    """Tool to get diseases associated with a given gene"""

    def get_name(self) -> str:
        """Returns the name of the tool."""
        return "get_gene_disease_associations"

    def get_description(self) -> str:
        """Returns the description of the tool."""
        return "Retrieve diseases associated with a given gene"

    def get_params_definition(self) -> Dict[str, ToolParamDefinition]:
        """Returns the parameter definitions for the tool."""
        return {
            "gene_id": ToolParamDefinition(
                param_type="str",
                description="The gene ID for which to get the associated diseases",
                required=True,
            )
        }

    async def run_impl(self, gene_id: str) -> List[str]:
        """
        Executes the retrieval of diseases associated with a given gene.

        Args:
            gene_id (str): The gene ID to analyze.

        Returns:
            List[str]: A list of diseases associated with the gene.
        """
        conn = sqlite3.connect("../gene_pathway_db.sqlite")
        c = conn.cursor()

        query = """
        SELECT disease
        FROM pathways
        WHERE gene_id = ?
        """
        c.execute(query, (gene_id,))
        results = c.fetchall()
        conn.close()

        diseases = [row[0] for row in results]
        return diseases


class GeneGoTermsTool(SingleMessageCustomTool):
    """Tool to get GO terms associated with a given gene"""

    def get_name(self) -> str:
        """Returns the name of the tool."""
        return "get_gene_go_terms"

    def get_description(self) -> str:
        """Returns the description of the tool."""
        return "Retrieve GO terms associated with a given gene"

    def get_params_definition(self) -> Dict[str, ToolParamDefinition]:
        """Returns the parameter definitions for the tool."""
        return {
            "gene_id": ToolParamDefinition(
                param_type="str",
                description="The gene ID for which to get the associated GO terms",
                required=True,
            )
        }

    async def run_impl(self, gene_id: str) -> List[str]:
        """
        Executes the retrieval of GO terms associated with a given gene.

        Args:
            gene_id (str): The gene ID to analyze.

        Returns:
            List[str]: A list of GO terms associated with the gene.
        """
        conn = sqlite3.connect("../gene_pathway_db.sqlite")
        c = conn.cursor()

        query = """
        SELECT go_id
        FROM gene_go_associations
        WHERE gene_id = ?
        """
        c.execute(query, (gene_id,))
        results = c.fetchall()
        conn.close()

        go_terms = [row[0] for row in results]
        return go_terms
