import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection string from environment variable
mongo_connection_string ="MONGO_CONNECTION_STRING"

# Debugging: print the connection string to ensure it's loaded correctly
print(f"Using connection string: {mongo_connection_string}")

def test_mongo_connection():
    try:
        # Initialize MongoDB client
        client = MongoClient(mongo_connection_string)
        
        # List available databases
        databases = client.list_database_names()
        
        # Print the list of databases
        print("Successfully connected to MongoDB!")
        print("Here are the databases available in your MongoDB server:")
        for db in databases:
            print(f"- {db}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_mongo_connection()
