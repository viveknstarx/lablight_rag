import logging
import requests
import os
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import socket
import json
# Setup logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('requests').setLevel(logging.DEBUG)

# Configurations
USER_END_URL = os.getenv("USER_END_URL", "http://0.0.0.0:9000/")
# API_ANSWER_UPDATE = "rag_studio/try_it_out/webhooks/questions/{question_id}/answer/"

# Payload creation functions
def q_and_a_payload(rag_answer, without_rag):
    return {
        "answer":{
        "with_rag": rag_answer,
        "without_rag": without_rag}
    }

# REST call function with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5), retry=retry_if_exception_type(socket.timeout))
def make_rest_call_userbackend(question_id, json_payload):
    API_ANSWER_UPDATE = f"rag_studio/try_it_out/webhooks/questions/{question_id}/answer/"
    print(json.dumps(json_payload))
    headers = {'Content-Type': 'application/json'}
    url = USER_END_URL + API_ANSWER_UPDATE
    try:
        response = requests.post(url, headers=headers, json=json_payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        logging.info(f"Response received: {result}")
        return result
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None
    
# @retry(stop=stop_after_attempt(3), wait=wait_fixed(5), retry=retry_if_exception_type(socket.timeout))
# def make_rest_call_userbackend_db(endpoint, json_payload):
#     print(json.dumps(json_payload))
#     headers = {'Content-Type': 'application/json'}
#     url = USER_END_URL + endpoint
#     try:
#         response = requests.post(url, headers=headers, json=json_payload)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         result = response.json()
#         logging.info(f"Response received: {result}")
#         return result
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Request failed: {e}")
#         return None


# if __name__ == "__main__":
    # question_id = "6688e039916778f130294b75"
    # rag_answer = "Example RAG answer"
    # without_rag = "Example without RAG answer"

    # payload = q_and_a_payload(rag_answer, without_rag)
    # response = make_rest_call_userbackend(question_id, payload)
    
    # logging.info(f"Final response: {json.dumps(payload,indent=2)}")