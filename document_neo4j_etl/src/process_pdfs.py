import os
import logging
from neo4j import GraphDatabase
import PyPDF2
import openai
import networkx as nx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to the directory containing PDF files
PDF_DIRECTORY_PATH = os.getenv("PDF_DIRECTORY_PATH")

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
# Configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

def process_pdfs_and_update_neo4j():
    """Load structured data from PDF files into Neo4j"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    LOGGER.info("Processing PDFs and updating Neo4j with extracted data")
    _process_pdfs_and_update_neo4j(driver)

def _process_pdfs_and_update_neo4j(driver):
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY_PATH) if f.endswith('.pdf')]
    documents = []

    for pdf_file in pdf_files:
        file_path = os.path.join(PDF_DIRECTORY_PATH, pdf_file)
        reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        documents.append(text)

    chunks = _split_documents_into_chunks(documents)
    elements = _extract_elements_from_chunks(chunks)
    summaries = _summarize_elements(elements)
    graph = _build_graph_from_summaries(summaries)

    _update_neo4j_with_graph(driver, graph)

def _split_documents_into_chunks(documents, chunk_size=600, overlap_size=100):
    chunks = []
    for document in documents:
        for i in range(0, len(document), chunk_size - overlap_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

def _extract_elements_from_chunks(chunks):
    elements = []
    for index, chunk in enumerate(chunks):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract entities and relationships from the following text."},
                {"role": "user", "content": chunk}
            ]
        )
        entities_and_relations = response.choices[0].message["content"]
        elements.append(entities_and_relations)
    return elements


def _summarize_elements(elements):
    summaries = []
    for index, element in enumerate(elements):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize the following entities and relationships in a structured format. Use '->' to represent relationships, after the 'Relationships:' word."},
                {"role": "user", "content": element}
            ]
        )
        summary = response.choices[0].message["content"]
        summaries.append(summary)
    return summaries

def _build_graph_from_summaries(summaries):
    G = nx.Graph()
    for summary in summaries:
        lines = summary.split("\n")
        entities_section = False
        relationships_section = False
        entities = []
        for line in lines:
            if line.startswith("### Entities:") or line.startswith("**Entities:**"):
                entities_section = True
                relationships_section = False
                continue
            elif line.startswith("### Relationships:") or line.startswith("**Relationships:**"):
                entities_section = False
                relationships_section = True
                continue
            if entities_section and line.strip():
                if line[0].isdigit() and line[1] == ".":
                    line = line.split(".", 1)[1].strip()
                entity = line.strip()
                entity = entity.replace("**", "")
                entities.append(entity)
                G.add_node(entity)
            elif relationships_section and line.strip():
                parts = line.split("->")
                if len(parts) >= 2:
                    source = parts[0].strip()
                    target = parts[-1].strip()
                    relation = " -> ".join(parts[1:-1]).strip()
                    G.add_edge(source, target, label=relation)
    return G

def _update_neo4j_with_graph(driver, G):
    with driver.session(database="neo4j") as session:
        for node in G.nodes:
            session.run("MERGE (n:Entity {name: $name})", name=node)
        for edge in G.edges(data=True):
            session.run(
                "MATCH (a:Entity {name: $source}), (b:Entity {name: $target}) "
                "MERGE (a)-[r:RELATION {label: $label}]->(b)",
                source=edge[0],
                target=edge[1],
                label=edge[2]['label']
            )

if __name__ == "__main__":
    process_pdfs_and_update_neo4j()
