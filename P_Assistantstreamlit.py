import streamlit as st
import time
import os
from typing import List, Optional, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

# Set the page configuration at the very beginning
st.set_page_config(page_title="Agentic RAG System", page_icon="📊", layout="wide")

# Streamlit UI for selecting LLM provider
st.sidebar.title("LLM Configuration")
llm_provider = st.sidebar.selectbox("Choose LLM Provider", ["Groq", "OpenAI"], index=0)
api_key = st.sidebar.text_input("Enter API Key", type="password")

# Import LLM (using a try/except to handle potential import errors gracefully)
try:
    if llm_provider == "Groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=api_key)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4", api_key=api_key)
except (ImportError, Exception) as e:
    st.error(f"Error initializing LLM: {str(e)}")
    st.warning("Please ensure you have the required packages installed and a valid API key.")
    
    # Fallback for development/testing without the actual LLM
    class MockLLM:
        def invoke(self, messages):
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            return MockResponse(f"Mock response for: {messages[-1].content[:50]}...")
    llm = MockLLM()

# Define models
class Analyst(BaseModel):
    name: str = Field(description="Name of the analyst.")
    role: str = Field(default="Senior Analyst", description="Role of the analyst in the context of the topic.")
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    summary: str = Field(default="", description="The analyst's summary of the topic.")
    
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: Optional[str]
    analysts: List[Analyst]
    combined_summary: Optional[str]

# Core functions
def determine_description(affiliation: str) -> str:
    """Generate a description for an analyst based on their affiliation"""
    messages = [
        SystemMessage(content="You are a helpful assistant that creates detailed descriptions for analysts."),
        HumanMessage(content=f"Create a detailed description for a Senior Analyst from {affiliation}. Describe their expertise, perspective, and what they would focus on when analyzing businesses or topics. Keep it under 100 words.")
    ]
    response = llm.invoke(messages)
    return response.content.strip()

def generate_analyst_summary(analyst: Analyst, topic: str) -> str:
    """Generate a summary from an analyst on the given topic"""
    messages = [
        SystemMessage(content=f"You are {analyst.name}, a {analyst.role} from {analyst.affiliation}. {analyst.description}"),
        HumanMessage(content=f"Write a detailed analysis on the topic: {topic}. Focus on aspects relevant to your background and expertise. Keep the summary between 200-300 words.")
    ]
    response = llm.invoke(messages)
    return response.content.strip()

def create_analysts(state: GenerateAnalystsState):
    """Create analysts based on the affiliations in the state"""
    # Get existing analysts or create empty list
    analysts = state.get("analysts", [])
    
    # Check if we're adding a new analyst based on feedback
    if state.get("human_analyst_feedback"):
        feedback = state["human_analyst_feedback"]
        
        # Check if feedback contains a specific affiliation to add
        if isinstance(feedback, dict) and "affiliation" in feedback:
            new_affiliation = feedback["affiliation"]
            
            # Generate a name for the new analyst
            new_name = f"Analyst {len(analysts) + 1}"
            
            # Create description for the new analyst
            new_description = determine_description(new_affiliation)
            
            # Create and add the new analyst
            new_analyst = Analyst(
                name=new_name,
                affiliation=new_affiliation,
                description=new_description
            )
            
            # Generate a summary for the new analyst
            new_analyst.summary = generate_analyst_summary(new_analyst, state["topic"])
            
            # Add to the existing analysts
            analysts.append(new_analyst)
            
            # Update the state
            state["analysts"] = analysts
            state["max_analysts"] = len(analysts)
    
    # If initial creation (no analysts yet)
    elif not analysts and "affiliations" in state:
        # Get affiliations from state (set by Streamlit UI)
        affiliations = state["affiliations"]
        
        # Create analysts
        for idx, affiliation in enumerate(affiliations):
            name = f"Analyst {idx + 1}"
            description = determine_description(affiliation)
            analyst = Analyst(
                name=name, 
                affiliation=affiliation, 
                description=description
            )
            
            # Generate a summary for the analyst
            if "topic" in state:
                analyst.summary = generate_analyst_summary(analyst, state["topic"])
            
            analysts.append(analyst)
        
        # Update state
        state["analysts"] = analysts
        state["max_analysts"] = len(analysts)
    
    return state

def human_feedback(state: GenerateAnalystsState):
    """Placeholder for human feedback - will be interrupted"""
    return state

def should_continue(state: GenerateAnalystsState):
    """Determine the next node based on human feedback"""
    feedback = state.get("human_analyst_feedback", "")
    
    # If feedback is a dict with "continue" key set to False, go to editor
    if isinstance(feedback, dict) and feedback.get("continue") is False:
        return "editor"
    
    # Otherwise, continue adding analysts
    return "create_analysts"

def editor(state: GenerateAnalystsState):
    """Combine all analyst summaries and create a final report"""
    analysts = state["analysts"]
    topic = state["topic"]
    
    # Collect all summaries
    summaries = [f"## {analyst.name} ({analyst.affiliation}) Summary:\n\n{analyst.summary}\n\n" 
                 for analyst in analysts]
    
    all_summaries = "\n".join(summaries)
    
    # Create a prompt for the editor to synthesize
    messages = [
        SystemMessage(content="You are a skilled editor responsible for synthesizing multiple analyst perspectives into a cohesive report."),
        HumanMessage(content=f"""
        As the editor, synthesize the following analyst summaries on the topic: {topic}
        
        {all_summaries}
        
        Create a comprehensive report that includes:
        1. An executive summary
        2. Key findings from across all analysts
        3. Points of consensus
        4. Different perspectives
        5. Final recommendations
        
        Format your response in Markdown with a title.
        """)
    ]
    
    # Generate the combined summary
    response = llm.invoke(messages)
    combined_summary = response.content.strip()
    
    # Update state with the combined summary
    state["combined_summary"] = combined_summary
    
    return state

# Create LangGraph
def create_graph():
    # Initialize the state graph
    builder = StateGraph(GenerateAnalystsState)
    
    # Add nodes
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("editor", editor)
    
    # Add edges
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", should_continue, 
                                {"create_analysts": "create_analysts", "editor": "editor"})
    builder.add_edge("editor", END)
    
    # Compile the graph
    memory = MemorySaver()
    graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
    
    return graph

# Streamlit UI
st.title("🤖 Multi-Analyst Report Generator")
st.subheader("Generate comprehensive reports with multiple perspectives")

# Initialize session state
if "graph" not in st.session_state:
    st.session_state.graph = create_graph()
    st.session_state.thread_id = f"thread_{int(time.time())}"
    st.session_state.thread = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.setup_complete = False
    st.session_state.analysts = []
    st.session_state.report_generated = False
    st.session_state.report_content = ""

# Step 1: Initial setup
if not st.session_state.setup_complete:
    with st.form("setup_form"):
        st.subheader("Step 1: Setup Your Analysis Team")
        
        # Topic input
        topic = st.text_input("Enter the topic to analyze:", placeholder="e.g., The future of AI in healthcare")
        
        # Get affiliations
        affiliations_input = st.text_area("Enter the affiliations of analysts (one per line):", 
                                          placeholder="e.g.,\nHarvard Medical School\nGoogle AI Research\nWorld Health Organization")
        
        submitted = st.form_submit_button("Create Initial Team")
        
        if submitted:
            if not topic or not affiliations_input:
                st.error("Please enter both a topic and at least one affiliation.")
            else:
                # Process affiliations
                affiliations = [aff.strip() for aff in affiliations_input.split('\n') if aff.strip()]
                
                if not affiliations:
                    st.error("Please enter at least one valid affiliation.")
                else:
                    # Show loading state
                    with st.spinner("Creating analysis team..."):
                        # Initialize state
                        initial_state = {
                            "topic": topic,
                            "max_analysts": 0,
                            "human_analyst_feedback": "",
                            "analysts": [],
                            "affiliations": affiliations  # Add affiliations to state
                        }
                        
                        # Run initial flow to create analysts
                        for event in st.session_state.graph.stream(
                            initial_state, 
                            st.session_state.thread, 
                            stream_mode="values"
                        ):
                            # Update analysts in session state if available
                            if "analysts" in event:
                                st.session_state.analysts = event["analysts"]
                        
                        st.session_state.setup_complete = True
                        st.session_state.topic = topic
                        st.rerun()

# Step 2: Display analysts and allow adding more
elif st.session_state.setup_complete and not st.session_state.report_generated:
    st.subheader("Step 2: Review Your Analysis Team")
    
    # Display analysts in cards
    col1, col2 = st.columns(2)
    
    for i, analyst in enumerate(st.session_state.analysts):
        with (col1 if i % 2 == 0 else col2):
            with st.expander(f"{analyst.name} - {analyst.affiliation}", expanded=True):
                st.markdown(f"**Role:** {analyst.role}")
                st.markdown(f"**Description:** {analyst.description}")
                st.markdown(f"**Summary Preview:**")
                st.markdown(analyst.summary[:150] + "...")
    
    # Option to add more analysts
    with st.form("add_analyst_form"):
        st.subheader("Add Another Analyst")
        new_affiliation = st.text_input("Enter the affiliation for a new analyst:", placeholder="e.g., Stanford Business School")
        
        col1, col2 = st.columns(2)
        with col1:
            add_analyst = st.form_submit_button("Add Analyst")
        with col2:
            generate_report = st.form_submit_button("Generate Final Report")
        
        if add_analyst:
            if not new_affiliation:
                st.error("Please enter an affiliation for the new analyst.")
            else:
                with st.spinner("Adding new analyst..."):
                    # Update with feedback to add a new analyst
                    feedback = {"affiliation": new_affiliation, "continue": True}
                    st.session_state.graph.update_state(
                        st.session_state.thread, 
                        {"human_analyst_feedback": feedback}, 
                        as_node="human_feedback"
                    )
                    
                    # Process the stream with the new feedback
                    for event in st.session_state.graph.stream(
                        None, 
                        st.session_state.thread, 
                        stream_mode="values"
                    ):
                        # Update analysts in session state if available
                        if "analysts" in event:
                            st.session_state.analysts = event["analysts"]
                    
                    st.rerun()
        
        if generate_report:
            with st.spinner("Generating final report... This may take a minute..."):
                # Set feedback to generate report
                feedback = {"continue": False}
                st.session_state.graph.update_state(
                    st.session_state.thread, 
                    {"human_analyst_feedback": feedback}, 
                    as_node="human_feedback"
                )
                
                # Process the final stream
                for event in st.session_state.graph.stream(
                    None, 
                    st.session_state.thread, 
                    stream_mode="values"
                ):
                    # Check for combined summary
                    if "combined_summary" in event:
                        st.session_state.report_content = event["combined_summary"]
                
                st.session_state.report_generated = True
                st.rerun()

# Step 3: Display final report
elif st.session_state.report_generated:
    st.subheader("Step 3: Final Report")
    
    # Display the report
    st.markdown(st.session_state.report_content)
    
    # Download button
    report_filename = f"report_{st.session_state.topic.replace(' ', '_')[:30]}.md"
    
    # Function to create download link
    def get_binary_file_downloader_html(report_content, file_name, button_text):
        # Write report to file
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        # Read file as bytes
        with open(file_name, 'rb') as f:
            import base64
            file_bytes = f.read()
            b64 = base64.b64encode(file_bytes).decode()
            
        # HTML component for download
        href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">{button_text}</a>'
        return href
    
    # Create and display download button
    download_button_str = get_binary_file_downloader_html(
        st.session_state.report_content, 
        report_filename, 
        'Download Report as Markdown'
    )
    st.markdown(download_button_str, unsafe_allow_html=True)
    
    # Option to start over
    if st.button("Start a New Report"):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with LangGraph and Streamlit")