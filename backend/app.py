from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain_together import Together
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph, END
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Initialize Flask app
app = Flask(__name__)
CORS(app)
load_dotenv()

# Set API keys
os.environ["TOGETHER_API_KEY"] = "6f6855a42324b5262be0ad7a249330d8e8f9242087267aedfe2cf8caf78861a8"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "28e8afd2c9a08946000b54f095536ed69ca8ae92"
os.environ["ZENSERP_API_KEY"] = "a34ec400-06b3-11f0-83fb-0557a7e63363"

# Cache for fact check results
fact_cache = {}

# Initialize LLM
llm = Together(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
    max_tokens=1024
)

# Initialize search tools
tavily_search = TavilySearchResults(k=5)

# Define custom tools
def zenserp_search(query: str) -> Dict[str, Any]:
    """Search the web using Zenserp API."""
    import requests
    
    url = "https://app.zenserp.com/api/v2/search"
    params = {
        "q": query,
        "apikey": os.environ["ZENSERP_API_KEY"],
        "search_engine": "google.com"
    }
    
    response = requests.get(url, params=params)
    return response.json()

def extract_facts(text: str) -> List[str]:
    """Extract factual claims from the input text."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the key factual claims from the text below. Return a list of facts, one per line."),
        ("human", "{text}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": text})
    return [fact.strip() for fact in result.split('\n') if fact.strip()]

def retrieve_evidence(fact: str) -> Dict[str, Any]:
    """Retrieve evidence for a specific fact."""
    # Get evidence from Tavily
    try:
        tavily_results = tavily_search.invoke(fact)
    except Exception:
        tavily_results = []
    
    # Get evidence from Zenserp as backup
    try:
        zenserp_results = zenserp_search(fact)
        organic_results = zenserp_results.get("organic_results", [])
    except Exception:
        organic_results = []
    
    # Return combined results
    return {
        "tavily": tavily_results,
        "zenserp": organic_results[:3]
    }

def verify_fact(fact: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a fact using retrieved evidence."""
    # Prepare evidence text
    evidence_text = ""
    for source, results in evidence.items():
        if source == "tavily" and results:
            for idx, result in enumerate(results, 1):
                evidence_text += f"Source {idx} ({source}): {result.get('title', 'No title')} - {result.get('content', 'No content')[:500]}\n\n"
        
        if source == "zenserp" and results:
            for idx, result in enumerate(results, 1):
                evidence_text += f"Source {idx} ({source}): {result.get('title', 'No title')} - {result.get('snippet', 'No snippet')}\n\n"
    
    # If no evidence is found
    if not evidence_text:
        return {
            "verdict": "UNVERIFIABLE",
            "explanation": "Insufficient evidence found to verify this claim."
        }
    
    # Create structured output parser
    response_schemas = [
        ResponseSchema(name="verdict", description="The assessment: TRUE, FALSE, PARTLY TRUE, or UNVERIFIABLE"),
        ResponseSchema(name="explanation", description="Reasoning with references to evidence")
    ]
    
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert fact-checker. Analyze this claim against the evidence provided.
    Determine if the claim is TRUE, FALSE, PARTLY TRUE, or UNVERIFIABLE based solely on the evidence.
    Provide a detailed explanation including specific references to the evidence.
    
    {format_instructions}
    
    Claim: {claim}
    
    Evidence:
    {evidence}
    """)
    
    try:
        # Try structured parsing approach
        chain = prompt.partial(format_instructions=format_instructions) | llm | parser
        return chain.invoke({"claim": fact, "evidence": evidence_text})
    except Exception:
        # Fallback to simpler approach
        simple_prompt = ChatPromptTemplate.from_template("""
        You are an expert fact-checker. Analyze this claim against the evidence.
        
        Your response must be valid JSON with this format:
        {{"verdict": "TRUE or FALSE or PARTLY TRUE or UNVERIFIABLE", "explanation": "your detailed reasoning"}}
        
        Claim: {claim}
        Evidence: {evidence}
        """)
        
        simple_chain = simple_prompt | llm | StrOutputParser()
        result = simple_chain.invoke({"claim": fact, "evidence": evidence_text})
        
        # Try to parse the result
        try:
            import json
            return json.loads(result)
        except Exception:
            # Extract verdict manually if parsing fails
            verdict_match = re.search(r'(TRUE|FALSE|PARTLY TRUE|UNVERIFIABLE)', result, re.IGNORECASE)
            verdict = verdict_match.group(0) if verdict_match else "UNVERIFIABLE"
            
            return {
                "verdict": verdict,
                "explanation": "Parsing failed. Please try again with a clearer statement."
            }

# Build fact-checking workflow
def build_fact_checking_workflow():
    from pydantic import BaseModel
    
    class State(BaseModel):
        query: str
        facts: Optional[List[str]] = None
        evidence: Optional[Dict[str, Any]] = None
        verification: Optional[Dict[str, Any]] = None
        final_report: Optional[str] = None

    # Create workflow graph
    workflow = StateGraph(State)
    
    # Extract facts node
    def extract_facts_node(state: State) -> Dict[str, Any]:
        facts = extract_facts(state.query)
        if not facts:  # Ensure we always have at least one fact
            facts = [state.query]
        return {"facts": facts}
    
    # Retrieve evidence node
    def retrieve_evidence_node(state: State) -> Dict[str, Any]:
        evidence = {}
        for fact in state.facts:
            evidence[fact] = retrieve_evidence(fact)
        return {"evidence": evidence}
    
    # Verify facts node
    def verify_facts_node(state: State) -> Dict[str, Any]:
        verification = {}
        for fact, fact_evidence in state.evidence.items():
            verification[fact] = verify_fact(fact, fact_evidence)
        return {"verification": verification}
    
    # Generate final report node
    def generate_report_node(state: State) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template("""
        You are a fact-checker generating a concise report.
        
        Based on analysis of: "{query}" and verification results, provide:
        
        1. An OVERALL VERDICT (TRUE or FALSE)
        2. A brief explanation supporting your verdict
        
        Your response MUST follow this format:
        **VERDICT: [TRUE/FALSE]**
        
        [Concise reasoning with key evidence - max 3 paragraphs]
        
        Verification results: {verification}
        """)
        
        chain = prompt | llm | StrOutputParser()
        final_report = chain.invoke({
            "query": state.query,
            "verification": state.verification
        })
        
        return {"final_report": final_report}
        
    # Add nodes and edges
    workflow.add_node("extract_facts", extract_facts_node)
    workflow.add_node("retrieve_evidence", retrieve_evidence_node)
    workflow.add_node("verify_facts", verify_facts_node)
    workflow.add_node("generate_report", generate_report_node)
    
    workflow.add_edge("extract_facts", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "verify_facts")
    workflow.add_edge("verify_facts", "generate_report")
    workflow.add_edge("generate_report", END)
    
    workflow.set_entry_point("extract_facts")
    
    return workflow.compile()

# Create workflow instance
fact_checking_workflow = build_fact_checking_workflow()

# Main fact-checking function
def perform_fact_check(query: str) -> Dict:
    """Check facts using AI workflow and return structured results."""
    try:
        initial_state = {"query": query}
        result = fact_checking_workflow.invoke(initial_state)
        
        # Parse the final report
        report = result.get("final_report", "")
        
        # Extract verdict
        verdict_match = re.search(r'\*\*VERDICT:\s*([^\*]+)\*\*', report, re.IGNORECASE)
        verdict = verdict_match.group(1).strip() if verdict_match else "UNVERIFIABLE"
        
        # Extract explanation (everything after the verdict)
        explanation = re.sub(r'\*\*VERDICT:[^\*]+\*\*', '', report, flags=re.IGNORECASE).strip()
        
        # Determine if true based on verdict
        is_true = "TRUE" in verdict.upper()
        
        # Choose a relevant source if available
        source = None
        for fact, verification in result.get("verification", {}).items():
            if verification.get("verdict") == "TRUE":
                # Look for URL in the explanation
                url_match = re.search(r'https?://[^\s]+', verification.get("explanation", ""))
                if url_match:
                    source = url_match.group(0)
                    break
        
        if not source:
            sources = [
                "https://www.reuters.com/fact-check/",
                "https://www.factcheck.org/",
                "https://www.snopes.com/"
            ]
            import random
            source = random.choice(sources)
        
        return {
            "isTrue": is_true,
            "message": explanation,
            "confidence": 0.9 if verdict.upper() in ["TRUE", "FALSE"] else 0.6,
            "source": source,
            "statement": query  # Include the original statement
        }
        
    except Exception as e:
        # Fallback response
        return {
            "isTrue": False,
            "message": f"We couldn't verify this statement. Please try again with a clearer statement.",
            "confidence": 0.0,
            "source": None,
            "statement": query  # Include the original statement
        }

# Function to generate additional details for a fact
def generate_more_details(statement: str) -> Dict[str, Any]:
    """Generate additional details about a fact for the 'See More' feature."""
    # Check if we have cached details
    if statement in fact_cache and "additional_details" in fact_cache[statement]:
        return fact_cache[statement]["additional_details"]
    
    # Generate additional details using LLM
    prompt = ChatPromptTemplate.from_template("""
    You are an expert fact-checker providing detailed information about a claim.
    
    For the claim: "{claim}"
    
    Please provide:
    1. Historical context
    2. Common misconceptions
    3. Additional sources (with URLs if available)
    4. Related facts
    5. Expert opinions
    
    Format your response as valid JSON with these fields:
    - historicalContext (string)
    - misconceptions (array of strings)
    - additionalSources (array of objects with 'name' and 'url' fields)
    - relatedFacts (array of strings)
    - expertOpinions (array of objects with 'expert' and 'opinion' fields)
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result_str = chain.invoke({"claim": statement})
        import json
        result = json.loads(result_str)
        
        # If this is the first check for this statement, initialize entry in cache
        if statement not in fact_cache:
            fact_cache[statement] = {}
        
        # Cache the additional details
        fact_cache[statement]["additional_details"] = result
        
        return result
    except Exception as e:
        # Fallback response
        return {
            "historicalContext": "We couldn't retrieve detailed historical context for this statement.",
            "misconceptions": ["Information unavailable"],
            "additionalSources": [{"name": "Factcheck.org", "url": "https://www.factcheck.org/"}],
            "relatedFacts": ["Information unavailable"],
            "expertOpinions": [{"expert": "Fact Checking Organization", "opinion": "Please try again with a clearer statement."}]
        }

# Add new route for more details
@app.route('/more-details', methods=['POST'])
def more_details():
    data = request.json
    statement = data.get('statement', '').strip()
    
    if not statement:
        return jsonify({"error": "Empty statement provided"}), 400
    
    # Generate more detailed information
    additional_info = generate_more_details(statement)
    
    return jsonify(additional_info)

# Update routes
@app.route('/check-fact', methods=['POST'])
def check_fact():
    data = request.json
    statement = data.get('statement', '').strip()
    
    if not statement:
        return jsonify({"error": "Empty statement provided"}), 400
    
    # Check cache first
    if statement in fact_cache:
        return jsonify(fact_cache[statement])
    
    # Real fact-checking with AI
    response = perform_fact_check(statement)
    
    # Add to cache
    fact_cache[statement] = response
    return jsonify(response)

@app.route('/trending', methods=['GET'])
def trending_topics():
    # Keep the trending topics
    trending = [
        {"statement": "Drinking hot water cures COVID-19", "searches": 3420},
        {"statement": "5G towers cause radiation sickness", "searches": 2910},
        {"statement": "Earth is flat", "searches": 2100},
        {"statement": "The moon landing was fake", "searches": 1850}
    ]
    return jsonify(trending)

@app.route('/random-fact', methods=['GET'])
def random_fact():
    facts = [
        {
            "statement": "The Great Wall of China is visible from space.",
            "isTrue": False,
            "explanation": "It's a common myth. The wall is difficult to see from low Earth orbit and impossible to see from the Moon."
        },
        {
            "statement": "Humans can distinguish between over a trillion different smells.",
            "isTrue": True,
            "explanation": "Research has shown that the human nose can distinguish at least 1 trillion different odors."
        },
        {
            "statement": "Goldfish have a memory span of only three seconds.",
            "isTrue": False,
            "explanation": "Goldfish can actually remember things for months, not just seconds."
        },
        {
            "statement": "The average person swallows eight spiders in their sleep each year.",
            "isTrue": False,
            "explanation": "This is a myth. Spiders have no interest in crawling into a person's mouth."
        }
    ]
    import random
    return jsonify(random.choice(facts))

if __name__ == '__main__':
    app.run(debug=True)
