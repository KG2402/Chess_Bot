import streamlit as st
import os
import re
from dotenv import load_dotenv
from typing import List, Dict

# LangChain imports
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()


# CONFIGURATION

class Config:
    """Configuration settings"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = "Llama-3.1-8B-Instant"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1024
    MEMORY_K = 10  
    
    PAGE_TITLE = "Chess Q&A Chatbot"
    PAGE_ICON = "♟️"
    
    GREETING_MESSAGE = """👋 Hello! I'm your Chess Q&A Bot!
    
Ask me anything about chess - rules, openings, strategies, famous players, tournaments, and chess history!"""


# GUARDRAILS: Implements chess-only content filtering 

class ChessGuardrails:
    
    CHESS_KEYWORDS = [
        'chess', 'checkmate', 'stalemate', 'draw', 'resign',
        'pawn', 'knight', 'bishop', 'rook', 'queen', 'king',
        'castling', 'en passant', 'promotion', 'capture',
        'opening', 'middlegame', 'endgame', 'gambit', 'defense',
        'fork', 'pin', 'skewer', 'sacrifice', 'tactic', 'strategy',
        'grandmaster', 'fide', 'elo', 'rating', 'tournament',
        'carlsen', 'kasparov', 'fischer', 'karpov', 'tal',
        'move', 'board', 'square', 'piece', 'position', 'game'
    ]
    
    @staticmethod
    def is_chess_related(text: str) -> bool:
        """Check if query is chess-related"""
        text_lower = text.lower()
        
        # Allow greetings and introductions
        greeting_patterns = [
            r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)',
            r'(i am|i\'m|my name is|this is|call me)\s+[a-zA-Z]+',
            r'^(thank you|thanks|bye|goodbye)',
        ]
        
        for pattern in greeting_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check chess keywords
        for keyword in ChessGuardrails.CHESS_KEYWORDS:
            if keyword in text_lower:
                return True
        
        # Check chess notation patterns
        patterns = [
            r'\b[a-h][1-8]\b',           # e4, d5
            r'\b[KQRBN][a-h]?[1-8]?\b',  # Nf3, Qd4
            r'\bO-O(-O)?\b',             # Castling
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    @staticmethod
    def get_rejection_message() -> str:
        """Return rejection message for non-chess queries"""
        return """🚫 I'm sorry, but I can only answer questions related to chess.

Please ask me about:
• Chess rules and regulations
• Opening strategies and defenses
• Famous players and games
• Chess tactics and strategies
• Tournament history
• Anything else chess-related!"""

# PERSONALIZATION: Name extraction

class PersonalizationHelper:
    """Handles user name extraction and management"""
    
    @staticmethod
    def extract_name(text: str) -> str:
        """Extract user name from introduction"""
        text_lower = text.lower()
        patterns = [
            r"(?:i am|i'm|my name is|this is|call me)\s+([a-zA-Z]+)",
            r"^(?:hi|hello|hey),?\s+(?:i am|i'm)\s+([a-zA-Z]+)",
            r"^([a-zA-Z]+)\s+here",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).capitalize()
                excluded = ['chess', 'here', 'there', 'player', 'learning']
                if name.lower() not in excluded:
                    return name
        return None
    
    @staticmethod
    def is_greeting(text: str) -> bool:
        """Check if message is a simple greeting"""
        text_lower = text.lower().strip()
        simple_greetings = [
            'hi', 'hello', 'hey', 'greetings', 
            'good morning', 'good afternoon', 'good evening',
            'howdy', 'hiya', 'yo'
        ]
        return text_lower in simple_greetings
    
    @staticmethod
    def get_greeting_response(user_name: str = None) -> str:
        """Generate friendly greeting response"""
        if user_name:
            return f"Hello again, {user_name}! 👋 How can I help you with chess today?"
        else:
            return "Hello! 👋 Welcome to the Chess Q&A Bot! Feel free to introduce yourself or ask me any chess-related questions!"


# LANGCHAIN LLM and Memory


def initialize_langchain():
    """Initialize LangChain components"""
    
    # Validate API key
    if not Config.GROQ_API_KEY:
        st.error("""
        ⚠️ **API Key Not Found!**
        
        Please create a `.env` file with:
        ```
        GROQ_API_KEY=your-api-key-here
        ```
        
        Get FREE API key from: https://console.groq.com/keys
        """)
        st.stop()
    

    llm = ChatGroq(
        groq_api_key=Config.GROQ_API_KEY,
        model_name=Config.MODEL_NAME,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS
    )
    
    memory = ConversationBufferWindowMemory(
        k=Config.MEMORY_K,
        return_messages=True,
        memory_key="chat_history"
    )
    
    return llm, memory

def create_chess_prompt(user_name: str = None) -> ChatPromptTemplate:
    """Create LangChain prompt template"""
    
    system_message = """You are an expert chess assistant with deep knowledge of chess rules, strategies, openings, endgames, famous players, and chess history.

Your responses should be:
- Accurate and factual about chess
- Clear and concise (2-4 sentences typically, longer for complex topics)
- Educational and friendly in tone
- Well-structured with proper formatting
- Only about chess topics
- Personalized with user's name when available

When explaining moves, use standard algebraic notation (e.g., e4, Nf3, O-O).
When discussing players, include relevant context like nationality and era.
Provide examples when explaining tactics or strategies."""
    
    if user_name:
        system_message += f"\n\nThe user's name is {user_name}. Use their name naturally in your responses when appropriate."
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    return prompt

# SESSION STATE INITIALIZATION

def initialize_session_state():
    """Initialize Streamlit session state"""
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add greeting message
        st.session_state.messages.append({
            'role': 'assistant',
            'content': Config.GREETING_MESSAGE
        })
    
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 0
    
    if 'llm' not in st.session_state:
        st.session_state.llm, st.session_state.memory = initialize_langchain()
    
    if 'session_start_time' not in st.session_state:
        from datetime import datetime
        st.session_state.session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def export_conversation_history() -> str:
    """Export conversation history as text"""
    from datetime import datetime
    
    history_text = f"""
# Chess Q&A Chatbot - Conversation History
# Session Started: {st.session_state.session_start_time}
# User: {st.session_state.user_name or "Anonymous"}
# Total Questions: {st.session_state.total_questions}
# Export Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'='*60}

"""
    
    for idx, msg in enumerate(st.session_state.messages, 1):
        role = msg['role'].upper()
        content = msg['content']
        history_text += f"{idx}. [{role}]\n{content}\n\n"
    
    return history_text

# UI css  STYLING

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTextInput > div > div > input {
            font-size: 16px;
            border-radius: 10px;
        }
        .stChatMessage {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with information"""
    with st.sidebar:
        st.header("ℹ️ About This Bot")
        st.write("""
        **Powered by:**
        - 🤖 LangChain Framework
        - ⚡ Groq API (Llama 3.1 8B)
        - 🎨 Streamlit UI
        
        **📚 Topics I Cover:**
        - Chess rules and regulations
        - Opening strategies and theory
        - Famous players and games
        - Chess history and tournaments
        - Tactics and strategic concepts
        - Endgame techniques
        """)
        
        st.markdown("---")
        
        st.header("🔧 Features")
        st.write("✅ **Guardrails** - Chess-only filter")
        st.write("✅ **Memory** - Remembers context")
        st.write("✅ **Greeting** - Auto welcome message")
        st.write("✅ **Personalization** - Remembers your name")
        st.write("✅ **History** - Full conversation saved")
        
        st.markdown("---")
        
        st.header("📊 Your Session")
        
        # Display user info
        if st.session_state.user_name:
            st.success(f"👤 **User**: {st.session_state.user_name}")
        else:
            st.info("👤 **User**: Not introduced yet")
        
        st.metric("Questions Asked", st.session_state.total_questions)
        st.metric("Total Messages", len(st.session_state.messages))
        
        st.markdown("---")
        
        # Conversation History Expander
        with st.expander("📜 View Conversation History"):
            if len(st.session_state.messages) > 1:
                for idx, msg in enumerate(st.session_state.messages[1:], 1):  # Skip greeting
                    role_icon = "🧑" if msg['role'] == 'user' else "🤖"
                    st.text(f"{role_icon} {msg['role'].title()}: {msg['content'][:50]}...")
            else:
                st.text("No messages yet. Start chatting!")
        
        st.markdown("---")
        
        # Download conversation history
        if len(st.session_state.messages) > 1:
            history_text = export_conversation_history()
            st.download_button(
                label="💾 Download Chat History",
                data=history_text,
                file_name=f"chess_chat_{st.session_state.user_name or 'anonymous'}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state.messages = [{
                'role': 'assistant',
                'content': Config.GREETING_MESSAGE
            }]
            st.session_state.user_name = None
            st.session_state.total_questions = 0
            st.session_state.memory.clear()
            st.success("✅ Chat cleared! Session reset.")
            st.rerun()
        
        st.markdown("---")
        st.caption("💡 **Tip**: I remember your name and all our conversation!")

# CHAT LOGIC

def generate_response(user_input: str) -> str:
    """Generate response using LangChain"""
    
    try:
        # Create prompt with current user name
        prompt = create_chess_prompt(st.session_state.user_name)
        
        # Create LangChain conversation chain
        conversation = LLMChain(
            llm=st.session_state.llm,
            prompt=prompt,
            memory=st.session_state.memory,
            verbose=False
        )
        
        # Generate response
        response = conversation.predict(input=user_input)
        
        return response.strip()
    
    except Exception as e:
        error_msg = str(e).lower()
        
        if 'api key' in error_msg or 'unauthorized' in error_msg:
            return "🔑 **Authentication Error**: Invalid API key. Please check your .env file."
        elif 'rate limit' in error_msg:
            return "⏱️ **Rate Limit**: Too many requests. Please wait a moment."
        else:
            return f"⚠️ **Error**: {str(e)}"

#  APPLICATION

def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Apply styling
    apply_custom_css()
    
    # Header
    st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
    st.markdown("""
        <p style='font-size: 18px; color: #666;'>
        Powered by LangChain 🦜🔗 and Llama 3.1 8B ⚡
        </p>
    """, unsafe_allow_html=True)
    
    # Feature badges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("🛡️ **Guardrails**")
    with col2:
        st.markdown("🧠 **Memory**")
    with col3:
        st.markdown("👋 **Greeting**")
    with col4:
        st.markdown("👤 **Personalization**")
    
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # User input
    user_input = st.chat_input("Ask me anything about chess...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add to messages
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Extract name if introduced
        if not st.session_state.user_name:
            name = PersonalizationHelper.extract_name(user_input)
            if name:
                st.session_state.user_name = name
        
        # Check if simple greeting
        if PersonalizationHelper.is_greeting(user_input):
            greeting_response = PersonalizationHelper.get_greeting_response(st.session_state.user_name)
            
            with st.chat_message("assistant"):
                st.markdown(greeting_response)
            
            st.session_state.messages.append({
                'role': 'assistant',
                'content': greeting_response
            })
        
        # Check guardrails for non-greeting messages
        elif not ChessGuardrails.is_chess_related(user_input):
            # Reject non-chess queries
            rejection_msg = ChessGuardrails.get_rejection_message()
            
            with st.chat_message("assistant"):
                st.markdown(rejection_msg)
            
            st.session_state.messages.append({
                'role': 'assistant',
                'content': rejection_msg
            })
        
        else:
            # Generate AI response for chess questions
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = generate_response(user_input)
                st.markdown(response)
            
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response
            })
            
            # Increment question counter
            st.session_state.total_questions += 1

if __name__ == "__main__":
    main()