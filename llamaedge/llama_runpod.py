# This is llama_runpod.py
import runpod
import requests
import json
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s     :%(lineno)-4d %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def handler(job):
    """
    Handler for RunPod that processes OpenAI-compatible requests
    """
    try:
        # Log the received job for debugging
        logger.info(f"Received job: {json.dumps(job)}")
        
        # Get the input data from the job
        job_input = job.get("input", {})
        
        # Check for different input formats:
        # 1. Direct format: { "messages": [...], "model": "..." }
        # 2. OpenAI wrapper format: { "openai_input": { "messages": [...], "model": "..." }, "openai_route": "..." }
        
        if "openai_input" in job_input and "openai_route" in job_input:
            # This is format #2 with openai_input wrapper
            openai_route = job_input.get("openai_route")
            request_payload = job_input.get("openai_input", {})
            
            # Extract route type from openai_route
            if "/chat/completions" in openai_route:
                endpoint_type = "chat/completions"
            elif "/completions" in openai_route:
                endpoint_type = "completions"
            else:
                endpoint_type = "chat/completions"  # Default
        else:
            # Assume direct format (#1)
            request_payload = job_input
            # Determine endpoint type from presence of "messages" or "prompt"
            endpoint_type = "chat/completions" if "messages" in request_payload else "completions"

        # Update the model name if needed (e.g., convert GPT to Llama)
        if "model" in request_payload and request_payload["model"].startswith("gpt"):
            request_payload["model"] = "llama-3-8b"  # Default Llama model
        
        # Ensure stream is disabled for serverless
        request_payload["stream"] = False
        
        # Set up the endpoint URL
        base_url = "http://127.0.0.1:4000"
        endpoint = f"{base_url}/v1/{endpoint_type}"
        
        # Log the request we're about to make
        logger.info(f"Making request to: {endpoint}")
        logger.info(f"Request payload: {json.dumps(request_payload)}")
        
        # Make the API call to the local Llama server
        response = requests.post(
            url=endpoint,
            headers={"Content-Type": "application/json"},
            json=request_payload,
            timeout=120
        )
        
        # Log response status
        logger.info(f"Response status: {response.status_code}")
        
        # Check for successful response
        if response.status_code == 200:
            # Parse and return the JSON response
            response_data = response.json()
            logger.info(f"Response data: {json.dumps(response_data)[:200]}...")  # Log first 200 chars
            return response_data
        else:
            # Handle error responses
            error_msg = f"API request failed with status {response.status_code}"
            logger.error(f"{error_msg}: {response.text}")
            return {
                "error": error_msg,
                "details": response.text
            }
            
    except Exception as e:
        # Log and return any exceptions
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {
            "error": "Exception in handler",
            "details": str(e)
        }

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})