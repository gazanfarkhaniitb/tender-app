import streamlit as st

# Import your pages here
import page1
import page2

# Dictionary to store the pages
PAGES = {
    "Summary": page1,
    "ChatBOT": page2
}

# Main Page Content
def main_page():
    st.title("AI-Powered Chatbot for Analyzing Tenders")
    
    st.write("""
    This app is an AI-powered chatbot built for analyzing tenders: 
        
    1. Company-specific solution which could give a similarity Score to prioritize tenders. Extremely helpful for all organizations to be ahead of their competitors by analyzing and making decisions faster and better.
    2. Offline Suite available for Enterprises. Benefits: NO INFORMATION is collected or saved. Complete secrecy and control within the company. 

    The Beta version contains 2 functions: Summary and ChatBOT.
    - Provides quick summary of the document and work to be done
    - Ask questions to the document and it answers it from within the document

    Note: AI can make mistakes. Please check important information again. This is just a demo shared for research purposes. 

    Please Enjoy. Made with ❤️ by GAK and Ashar. 

    **About Creators:**
    - Ashar Mirza: IIT Delhi
    - Gazanfar Khan: IIT Bombay
    """)

# Sidebar for navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", ["Main Page"] + list(PAGES.keys()))

if selection == "Main Page":
    main_page()
else:
    page = PAGES[selection]
    page.app()
