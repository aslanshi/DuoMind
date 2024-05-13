import os
import openai
from openai import AzureOpenAI
import json
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

from LLM_CONFIG import (
    AZURE_API_URL,
    AUZRE_API_VERSION,
    AZURE_DEPLOY_NAME,
)
from index import retrieve_image, RetrieveImage

def mask_extra_property(messages: List[Dict]) -> List[Dict]:
    """Mask the response_id property while sending request to the API"""
    return [{key: d[key] for key in d if key !="response_id"} for d in messages]

## Set up API services

system_prompt = {
    "role": "system",
    "content": "If you're given an description of an image, respond only the following: Here is the photo you're looking for:"
}

azure_oa_client = AzureOpenAI(
    azure_endpoint=AZURE_API_URL,
    azure_deployment=AZURE_DEPLOY_NAME,
    api_version=AUZRE_API_VERSION,
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

azure_model = AzureChatOpenAI(
    azure_endpoint=AZURE_API_URL,
    deployment_name=AZURE_DEPLOY_NAME,
    openai_api_version=AUZRE_API_VERSION,
    openai_api_key=os.getenv("AZURE_OPENAI_KEY")
)
available_functions = {
    "RetrieveImage": retrieve_image,
}

## UI

st.title("DuoMind")
st.caption("Search through your precious memories")

if "response_id" not in st.session_state:
    st.session_state["response_id"] = 0

# need this check otherwise state variable would be overwritten when the next loop starts
if "images" not in st.session_state:
    st.session_state["images"] = {str(st.session_state.response_id): None}

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "What do you have in mind today?", "response_id": st.session_state.response_id}
    ]
    # st.session_state["messages"] = {
    #     str(st.session_state.response_id): {"role": "assistant", "content": "What do you have in mind today?"}
    # }

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    if msg["role"] == "assistant":
        response_id = msg["response_id"]
        if st.session_state.images[str(response_id)] is not None:
            st.image(st.session_state.images[str(response_id)], width=400)
# for response_id, msg in st.session_state.messages.items():
#     st.chat_message(msg["role"]).write(msg["content"])
#     if msg["role"] == "assistant":
#         if st.session_state.images[response_id] is not None:
#             st.image(st.session_state.images[response_id], width=400)


if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    # st.session_state.response_id += 1
    # st.session_state.messages.update(
    #     {str(st.session_state.response_id): {"role": "user", "content": prompt}}
    # )
    st.chat_message("user").write(prompt)

    # check function calls
    result = azure_model.invoke(prompt, functions=[convert_pydantic_to_openai_function(RetrieveImage)])
    
    stream_response = azure_oa_client.chat.completions.create(
        model="gpt-4", # or AZURE_DEPLOY_NAME or not needed
        messages=[system_prompt] + mask_extra_property(st.session_state.messages), # prepend system message but not showing in the app
        stream=True
    )
    # stream_response = azure_oa_client.chat.completions.create(
    #     model=MODEL_NAME,
    #     messages=[system_prompt] + list(st.session_state.messages.values()), # prepend system message but not showing in the app
    #     stream=True
    # )

    st.session_state.response_id += 1

    try:
        # Azure chat model return additional kwargs even no function called
        # leaving try-except here in case we swicth to other APIs
        if len(result.additional_kwargs) != 0:
            st.chat_message("assistant").write("Something magical is happening, please wait...")
            # st.write("Something magical is happening, please wait...")
            function_call_results = result.additional_kwargs["function_call"]
            function_name = function_call_results["name"]
            function_args = json.loads(function_call_results["arguments"])

            func_response = available_functions[function_name](**function_args)
            image_path, score = func_response
            # cosine distance: lower -> higher similarity
            if score < 0.4:
                # TODO: can prepend wait message here:
                full_response = st.chat_message("assistant").write_stream(stream_response)
                st.image(image_path, width=400)
                st.session_state.images.update({str(st.session_state.response_id): image_path})
            else:
                full_response = "Sorry, I could not find the photo you're looking for."
                st.chat_message("assistant").write(full_response)
                # st.write("Sorry, I could not find the photo you're looking for.")
                st.session_state.images.update({str(st.session_state.response_id): None})
        else:
            full_response = st.chat_message("assistant").write_stream(stream_response)
            st.session_state.images.update({str(st.session_state.response_id): None})
    
    except (AttributeError, KeyError):
        full_response = st.chat_message("assistant").write_stream(stream_response)
        st.session_state.images.update({str(st.session_state.response_id): None})

    
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "response_id": st.session_state.response_id}
    )
    # st.write(st.session_state.images)
    # st.session_state.messages.update(
    #     {str(st.session_state.response_id): {"role": "assistant", "content": full_response}}
    # )
