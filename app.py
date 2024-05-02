import os
import openai
from openai import AzureOpenAI
import json
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

from LLM_CONFIG import (
    API_URL,
    MODEL_NAME,
    AZURE_API_URL,
    AUZRE_API_VERSION,
    AZURE_DEPLOY_NAME,
)
from index import retrieve_image, RetrieveImage

system_prompt = {
    "role": "system",
    "content": "If you're given an description of an image, respond the following: Here is the photo you're looking for:"
}

# openai.base_url = API_URL
# openai.api_key = os.getenv("MGA_API_TOKEN")

azure_oa_client = AzureOpenAI(
    azure_endpoint=AZURE_API_URL,
    azure_deployment=AZURE_DEPLOY_NAME,
    api_version=AUZRE_API_VERSION,
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

# model = ChatOpenAI(model="gpt-4-1106-preview", base_url=API_URL, api_key=os.getenv("MGA_API_TOKEN"))
azure_model = AzureChatOpenAI(
    azure_endpoint=AZURE_API_URL,
    deployment_name=AZURE_DEPLOY_NAME,
    openai_api_version=AUZRE_API_VERSION,
    openai_api_key=os.getenv("AZURE_OPENAI_KEY")
)
available_functions = {
    "RetrieveImage": retrieve_image,
}

# with open("photo_caption_embed_mapping.json", "rb") as f:
#     image_data = json.load(f)

st.title("DuoMind")
st.caption("Search through your precious memories")
# st.write("Streamlit version", st.__version__)

if "response_id" not in st.session_state:
    st.session_state["response_id"] = 0

# need this check otherwise state variable would be overwritten when the next loop starts
if "images" not in st.session_state:
    st.session_state["images"] = {str(st.session_state.response_id): None}

if "messages" not in st.session_state:
    # st.session_state["messages"] = [
    #     {"role": "assistant", "content": "What do you have in mind today?", "response_id": st.session_state.response_id}
    # ]
    st.session_state["messages"] = {
        str(st.session_state.response_id): {"role": "assistant", "content": "What do you have in mind today?"}
    }

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])
#     if msg["role"] == "assistant":
#         response_id = msg["response_id"]
#         if st.session_state.images[str(response_id)] is not None:
#             st.image(st.session_state.images[str(response_id)], width=400)
for response_id, msg in st.session_state.messages.items():
    st.chat_message(msg["role"]).write(msg["content"])
    if msg["role"] == "assistant":
        if st.session_state.images[response_id] is not None:
            st.image(st.session_state.images[response_id], width=400)


if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    st.session_state.response_id += 1
    # st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.update(
        {str(st.session_state.response_id): {"role": "user", "content": prompt}}
    )
    st.chat_message("user").write(prompt)
    
    stream_response = azure_oa_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[system_prompt] + list(st.session_state.messages.values()), # prepend system message but not showing in the app
        stream=True
    )

    full_response = st.chat_message("assistant").write_stream(stream_response)
    st.session_state.response_id += 1
    # st.session_state.messages.append(
    #     {"role": "assistant", "content": full_response, "response_id": st.session_state.response_id}
    # )
    st.session_state.messages.update(
        {str(st.session_state.response_id): {"role": "assistant", "content": full_response}}
    )

    # result = model.invoke(prompt, functions=[convert_pydantic_to_openai_function(RetrieveImage)])
    result = azure_model.invoke(prompt, functions=[convert_pydantic_to_openai_function(RetrieveImage)])
    st.write(st.session_state.images)
    try:
        # Azure chat model return additional kwargs even no function called
        if len(result.additional_kwargs) != 0:
            function_call_results = result.additional_kwargs["function_call"]
            function_name = function_call_results["name"]
            function_args = json.loads(function_call_results["arguments"])

            func_response = available_functions[function_name](**function_args)
            image_path, score = func_response
            # cosine distance: lower -> higher similarity
            if score < 0.4:
                st.image(image_path, width=400)
                st.session_state.images.update({str(st.session_state.response_id): image_path})
            else:
                st.write("Sorry, I could not find the photo you're looking for.")
                st.session_state.images.update({str(st.session_state.response_id): None})
        else:
            st.session_state.images.update({str(st.session_state.response_id): None})
    except (AttributeError, KeyError):
        st.session_state.images.update({str(st.session_state.response_id): None})

    # image_path, score = retrieve_image(prompt)
    # # cosine distance: lower -> higher similarity
    # if score < 0.4:
    #     st.image(image_data["paths"][0], width=400)
    #     st.session_state.images.update({str(st.session_state.response_id): image_data["paths"][0]})
    # else:
    #     st.write("Sorry, I could not find the photo you're looking for.")
    #     st.session_state.images.update({str(st.session_state.response_id): None})
