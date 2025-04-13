# OpenAI API Endpoints

## Responses

This section details the API endpoints related to generating and managing model responses.

### Create a model response
- **Method:** `POST`
- **Endpoint:** `https://api.openai.com/v1/responses`
- **Description:** Creates a new model response based on provided text or image inputs. The model can generate text or JSON outputs and can be configured to call custom code or use built-in tools (web search, file search) to incorporate external data.

### Retrieve a model response
- **Method:** `GET`
- **Endpoint:** `https://api.openai.com/v1/responses/{response_id}`
- **Description:** Retrieves a specific model response using its unique ID (`response_id`).

### Delete a model response
- **Method:** `DELETE`
- **Endpoint:** `https://api.openai.com/v1/responses/{response_id}`
- **Description:** Deletes a model response identified by its `response_id`.

### List input items for a response
- **Method:** `GET`
- **Endpoint:** `https://api.openai.com/v1/responses/{response_id}/input_items`
- **Description:** Returns a list of the input items that were used to generate a specific model response (identified by `response_id`).

## Chat Completions

This section outlines the API endpoints for interacting with chat-based model completions. OpenAI recommends exploring the "Responses" endpoint for new projects, as it leverages the latest platform features. Parameter support may vary depending on the model used, with newer reasoning models having specific parameter considerations (refer to the reasoning guide for details on unsupported parameters).

### Create a chat completion
- **Method:** `POST`
- **Endpoint:** `https://api.openai.com/v1/chat/completions`
- **Description:** Creates a new model response for a given chat conversation. Refer to the text generation, vision, and audio guides for more information.

### Get chat messages
- **Method:** `GET`
- **Endpoint:** `https://api.openai.com/v1/chat/completions/{completion_id}/messages`
- **Description:** Retrieves the messages associated with a stored chat completion, identified by its `completion_id`. Only chat completions created with the `store` parameter set to `true` can be accessed.

### List Chat Completions
- **Method:** `GET`
- **Endpoint:** `https://api.openai.com/v1/chat/completions`
- **Description:** Lists all stored chat completions. Only chat completions that were created with the `store` parameter set to `true` will be included in the list.

### Update chat completion
- **Method:** `POST`
- **Endpoint:** `https://api.openai.com/v1/chat/completions/{completion_id}`
- **Description:** Modifies a stored chat completion identified by its `completion_id`. Currently, the only supported modification is updating the `metadata` field. This endpoint only works for chat completions created with the `store` parameter set to `true`.

### Delete chat completion
- **Method:** `DELETE`
- **Endpoint:** `https://api.openai.com/v1/chat/completions/{completion_id}`
- **Description:** Deletes a stored chat completion identified by its `completion_id`. This operation is only applicable to chat completions created with the `store` parameter set to `true`.

# Gemini API Endpoints

This section lists the API endpoints for interacting with the Gemini models.

### List available models
- **Method:** `GET`
- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models`
- **Description:** Retrieves a list of the available Gemini models.

### Generate content
- **Method:** `POST`
- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/{model=models/*}:generateContent`
- **Description:** Generates content based on the provided input using the specified Gemini model. The `{model=models/*}` placeholder indicates that you need to replace `*` with the specific model name or identifier.

### Stream generate content
- **Method:** `POST`
- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/{model=models/*}:streamGenerateContent`
- **Description:** Generates content in a streaming fashion using the specified Gemini model. This allows for receiving partial results as they are generated. Replace `*` in the placeholder with the model identifier.

### Count tokens
- **Method:** `POST`
- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/{model=models/*}:countTokens`
- **Description:** Counts the number of tokens in the input content for the specified Gemini model. Replace `*` in the placeholder with the model identifier.

### Embed content
- **Method:** `POST`
- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/{model=models/*}:embedContent`
- **Description:** Generates embeddings for the input content using the specified Gemini model. Replace `*` in the placeholder with the model identifier.

### Batch embed contents
- **Method:** `POST`
- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/{model=models/*}:batchEmbedContents`
- **Description:** Generates embeddings for a batch of input content using the specified Gemini model. Replace `*` in the placeholder with the model identifier.