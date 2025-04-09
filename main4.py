import os
import fitz  # PyMuPDF
import streamlit as st
from py2neo import Graph, Node, Relationship
from dotenv import load_dotenv
import json
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
import hashlib

# --- Load environment variables ---
load_dotenv()

# --- Neo4j Setup ---
graph = Graph(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    secure=True,
    verify=False
)

# --- Groq LLM Setup ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

# --- Utility Functions ---
def clear_neo4j_database():
    graph.run("MATCH (n) DETACH DELETE n")

def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc])

def get_pdf_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_structured_data_from_llm(resume_text):
    prompt = f"""
You are an intelligent assistant that converts resumes into structured JSON.

Given the following resume:

{resume_text}

Extract structured data in this JSON format:

{{
  "name": "<Full Name>",
  "skills": ["Skill1", "Skill2"],
  "education": ["Degree in Field from University"],
  "projects": ["Project title or short description"],
  "experience": ["Job title at Company - short description"],
  "certifications": ["Certification Name"]
}}

Only return pure JSON. Do not include explanations or any other text.
"""
    try:
        response = llm([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ])
        return json.loads(response.content)
    except Exception as e:
        print("Error parsing LLM output:", e)
        return {}

def add_candidate_to_neo4j(data, resume_text):
    name = data.get("name", "Unknown")
    content_hash = get_pdf_hash(resume_text)

    candidate = Node("Candidate", name=name, content=resume_text, content_hash=content_hash)
    graph.merge(candidate, "Candidate", "content_hash")

    for label, rel, items in [
        ("Skill", "HAS_SKILL", data.get("skills", [])),
        ("Education", "HAS_EDUCATION", data.get("education", [])),
        ("Project", "HAS_PROJECT", data.get("projects", [])),
        ("Experience", "HAS_EXPERIENCE", data.get("experience", [])),
        ("Certification", "HAS_CERTIFICATION", data.get("certifications", []))
    ]:
        for item in items:
            node = Node(label, name=item)
            graph.merge(node, label, "name")
            graph.merge(Relationship(candidate, rel, node))

# --- Streamlit App ---
st.set_page_config(page_title="Resume Graph Search", layout="wide")
st.title("Resume Graph Search with LLM + Neo4j")

st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if st.sidebar.button("Clear Graph DB"):
    clear_neo4j_database()
    st.session_state.processed_files.clear()
    st.success("Graph database cleared!")

if uploaded_files:
    with st.spinner("Processing resumes..."):
        for file in uploaded_files:
            if file.name in st.session_state.processed_files:
                continue

            resume_text = extract_text_from_pdf(file)
            structured_data = get_structured_data_from_llm(resume_text)

            if structured_data:
                add_candidate_to_neo4j(structured_data, resume_text)
                st.session_state.processed_files.add(file.name)

        st.success(f"{len(st.session_state.processed_files)} unique resumes processed and inserted into Neo4j.")

# --- Search Section ---
st.sidebar.header("Search Candidates by Skill")
skill_query = st.sidebar.text_input("Enter skill to search", value="Python")

if skill_query:
    with st.spinner("Searching graph..."):
        cypher = """
        MATCH (c:Candidate)-[r:HAS_SKILL]->(s:Skill {name: $skill}) RETURN c.name, c.content
        """
        results = graph.run(cypher, skill=skill_query).data()

        st.subheader(f"Candidates with skill: {skill_query}")
        if results:
            for res in results:
                st.markdown(f"**{res['c.name']}**\n")
                # with st.expander("Show resume content"):
                #     st.code(res['c.content'], language="text")
        else:
            st.warning("No candidates found for the selected skill.")
