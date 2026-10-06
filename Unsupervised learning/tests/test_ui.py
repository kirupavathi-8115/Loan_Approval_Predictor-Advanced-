import re
import pytest
from playwright.sync_api import Page, expect

def test_streamlit_dashboard(page: Page):
    print("Navigating to Streamlit Dashboard...")
    page.goto("http://localhost:8501")
    
    print("Waiting for page title...")
    expect(page).to_have_title(re.compile("Customer Personas"))
    
    print("Checking for dashboard heading...")
    expect(page.locator("text=Customer Persona Dashboard")).to_be_visible(timeout=15000)
    
    print("Checking for Discovered Personas section...")
    expect(page.locator("text=Discovered Personas")).to_be_visible()
    
    print("Test passed successfully.")
