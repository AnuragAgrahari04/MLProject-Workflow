import sys

# We are trying to import from our 'src' package.
# If this line runs without an error, the setup is successful.
from src.logger import logging

def run_test():
    """
    A simple function to test if the logger from the src package works.
    """
    try:
        logging.info("SUCCESS: The 'src' folder is working correctly and modules can be imported.")
        print("✅ Test successful! Check the new log file in the 'logs' directory.")
    except Exception as e:
        print(f"❌ Test Failed: An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_test()