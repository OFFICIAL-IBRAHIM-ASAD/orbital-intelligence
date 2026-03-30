import ee

def setup_gee():
    """
    Authenticates and initializes the Google Earth Engine API.
    """
    PROJECT_ID = 'orbital-intelligence-thesis' 
    
    try:
        print(f"Attempting to initialize Earth Engine with project: '{PROJECT_ID}'...")
        
        # Initialize directly with the project ID
        ee.Initialize(project=PROJECT_ID)
        
        print("Success: Google Earth Engine API is initialized and ready.")
        
    except Exception as e:
        print(f"Initialization failed. Attempting to authenticate...")
        try:
            # Bug fixed: Authenticate() does not take the project ID as an argument here.
            # It just refreshes the local token.
            ee.Authenticate()
            
            # Then we initialize with the project ID
            ee.Initialize(project=PROJECT_ID)
            print("Success: Google Earth Engine API is initialized and ready.")
        except Exception as deep_e:
            print(f"Critical Error: {deep_e}")
            print("Please ensure you registered the project at the URL provided in the console.")

if __name__ == "__main__":
    setup_gee()