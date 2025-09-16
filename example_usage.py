#!/usr/bin/env python3

"""
Example usage of the Protein Happi async search API.

This script demonstrates how to:
1. Submit a search job
2. Poll for job completion
3. Retrieve and display results
"""

import requests
import time
import json

# API configuration
BASE_URL = "http://localhost:8001"  # Update with your server URL

def submit_search_job(sequences, max_hits=5):
    """Submit a search job to the API."""
    
    # Prepare the request data
    data = {
        "query_sequences": [
            {"id": seq_id, "sequence": sequence}
            for seq_id, sequence in sequences.items()
        ],
        "max_hits": max_hits,
        "best_hit_only": False,
        "return_query_embeddings": False,
        "return_hit_embeddings": False
    }
    
    # Submit the job
    response = requests.post(f"{BASE_URL}/search", json=data)
    response.raise_for_status()
    
    return response.json()

def get_job_status(job_id):
    """Get the current status of a job."""
    response = requests.get(f"{BASE_URL}/job/{job_id}")
    response.raise_for_status()
    return response.json()

def wait_for_completion(job_id, poll_interval=1, timeout=60):
    """Poll for job completion."""
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status_data = get_job_status(job_id)
        status = status_data["status"]
        
        print(f"Job {job_id}: {status}")
        
        if status == "completed":
            return status_data
        elif status == "failed":
            print(f"Job failed: {status_data.get('error_message', 'Unknown error')}")
            return status_data
        
        time.sleep(poll_interval)
    
    print(f"Job {job_id}: Timed out after {timeout} seconds")
    return None

def display_results(results):
    """Display search results in a readable format."""
    
    if not results or not results.get("results"):
        print("No results available")
        return
    
    hits = results["results"]["hits"]
    
    print(f"\n🔍 Search Results ({len(hits)} queries)")
    print("=" * 50)
    
    for hit_group in hits:
        query_id = hit_group["query_id"]
        total_hits = hit_group["total_hits"]
        best_hit = hit_group["best_hit"]
        
        print(f"\n📋 Query: {query_id}")
        print(f"   Total hits: {total_hits}")
        print(f"   Best hit: {best_hit['id']} (score: {best_hit['score']:.3f})")
        
        # Show additional hits if available
        if hit_group.get("hits"):
            print(f"   Other hits:")
            for hit in hit_group["hits"][:3]:  # Show top 3 additional hits
                print(f"     - {hit['id']} (score: {hit['score']:.3f})")

def main():
    """Main example function."""
    
    # Example protein sequences
    sequences = {
        "insulin_human": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        "lysozyme_chicken": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    }
    
    try:
        print("🚀 Submitting search job...")
        
        # Submit the job
        job_response = submit_search_job(sequences, max_hits=10)
        job_id = job_response["job_id"]
        
        print(f"✅ Job submitted successfully!")
        print(f"   Job ID: {job_id}")
        print(f"   Status: {job_response['status']}")
        print(f"   Submitted at: {time.ctime(job_response['submitted_at'])}")
        
        # Wait for completion
        print(f"\n⏳ Waiting for job completion...")
        final_status = wait_for_completion(job_id)
        
        if final_status and final_status["status"] == "completed":
            print(f"\n✅ Job completed successfully!")
            print(f"   Duration: {final_status['completed_at'] - final_status['submitted_at']:.3f} seconds")
            
            # Display results
            display_results(final_status)
        
    except requests.RequestException as e:
        print(f"❌ API request failed: {e}")
    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted by user")

if __name__ == "__main__":
    main()